"""
Training utilities and training loop for the multimodal model.

Single-objective: predict next-day TSLA return, evaluate on reconstructed price MAE.
Uses ReduceLROnPlateau on val MAE (dollar space).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import joblib
import sys
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import DETERMINISTIC_TRAINING, RANDOM_SEED, TRAINING_CONFIG, MODELS_DIR, MODEL_VERSION
from src.models.fusion import create_model


class StockDataset(Dataset):
    """PyTorch Dataset for next-day return prediction."""

    def __init__(
        self,
        X_price: np.ndarray,
        X_sentiment: Optional[np.ndarray],
        y_reg: np.ndarray,
    ):
        self.X_price = torch.FloatTensor(X_price)
        self.X_sentiment = torch.FloatTensor(X_sentiment) if X_sentiment is not None else None
        self.y_reg = torch.FloatTensor(y_reg)

    def __len__(self):
        return len(self.y_reg)

    def __getitem__(self, idx):
        item = {
            'X_price': self.X_price[idx],
            'y_reg': self.y_reg[idx],
        }
        if self.X_sentiment is not None:
            item['X_sentiment'] = self.X_sentiment[idx]
        return item


class Trainer:
    """Trainer for the multimodal next-day return model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = TRAINING_CONFIG['learning_rate'],
        random_seed: int = RANDOM_SEED,
    ):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.random_seed = random_seed

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        self.scheduler = None  # set in fit()
        self.criterion = nn.SmoothL1Loss()

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'lr': [],
        }

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            X_price = batch['X_price'].to(self.device)
            X_sentiment = batch.get('X_sentiment')
            if X_sentiment is not None:
                X_sentiment = X_sentiment.to(self.device)
            y_reg = batch['y_reg'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_price, X_sentiment)
            loss = self.criterion(outputs['regression'], y_reg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {'loss': total_loss / n_batches}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, return_scaler=None, close_prices=None) -> dict:
        """Evaluate. When return_scaler and close_prices are given, MAE/RMSE/MAPE are in dollars."""
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_targets = []

        for batch in dataloader:
            X_price = batch['X_price'].to(self.device)
            X_sentiment = batch.get('X_sentiment')
            if X_sentiment is not None:
                X_sentiment = X_sentiment.to(self.device)
            y_reg = batch['y_reg'].to(self.device)

            outputs = self.model(X_price, X_sentiment)
            loss = self.criterion(outputs['regression'], y_reg)
            total_loss += loss.item()

            all_predictions.extend(outputs['regression'].cpu().numpy())
            all_targets.extend(y_reg.cpu().numpy())

        n_batches = len(dataloader)
        predictions_scaled = np.array(all_predictions)
        targets_scaled = np.array(all_targets)

        if return_scaler is not None:
            pred_returns = return_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            actual_returns = return_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()
        else:
            pred_returns = predictions_scaled
            actual_returns = targets_scaled

        if close_prices is not None:
            if len(close_prices) != len(pred_returns):
                min_len = min(len(close_prices), len(pred_returns))
                close_prices = close_prices[:min_len]
                pred_returns = pred_returns[:min_len]
                actual_returns = actual_returns[:min_len]

            pred_prices = close_prices * (1 + pred_returns)
            actual_prices = close_prices * (1 + actual_returns)

            rmse = float(np.sqrt(np.mean((pred_prices - actual_prices) ** 2)))
            mae = float(np.mean(np.abs(pred_prices - actual_prices)))
            mape = float(np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100)
        else:
            rmse = float(np.sqrt(np.mean((pred_returns - actual_returns) ** 2)))
            mae = float(np.mean(np.abs(pred_returns - actual_returns)))
            mape = None

        return {
            'loss': total_loss / n_batches,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
        }

    def fit(
        self,
        train_data: dict,
        val_data: dict,
        epochs: int = TRAINING_CONFIG['epochs'],
        batch_size: int = TRAINING_CONFIG['batch_size'],
        return_scaler=None,
    ) -> dict:
        """Train for the configured number of epochs with ReduceLROnPlateau on val MAE."""
        train_dataset = StockDataset(
            train_data['X_price'], train_data['X_sentiment'], train_data['y_reg']
        )
        val_dataset = StockDataset(
            val_data['X_price'], val_data['X_sentiment'], val_data['y_reg']
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Validation MAE in dollar space steers the LR schedule.
        val_close_prices = val_data.get('close_prices')

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=TRAINING_CONFIG['lr_factor'],
            patience=TRAINING_CONFIG['lr_patience'],
        )

        best_val_mae = float('inf')

        # Save an initial checkpoint so best_model.pt always exists.
        self.save_model('best_model.pt')

        print(f"Training on {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Epochs: {epochs}")
        print("-" * 50)

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(
                val_loader,
                return_scaler=return_scaler,
                close_prices=val_close_prices,
            )

            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_metrics['mae'])

            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['lr'].append(current_lr)

            improved = val_metrics['mae'] < best_val_mae
            if improved:
                best_val_mae = val_metrics['mae']
                self.save_model('best_model.pt')

            marker = " *" if improved else ""
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val MAE: ${val_metrics['mae']:.2f} | "
                f"LR: {current_lr:.2e}{marker}"
            )

        self.save_model('final_model.pt')
        print(f"Best val MAE: ${best_val_mae:.2f}")

        return self.history

    def save_model(self, filename: str):
        path = MODELS_DIR / filename
        torch.save({
            'model_version': MODEL_VERSION,
            'random_seed': self.random_seed,
            'deterministic_training': DETERMINISTIC_TRAINING,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)

    def load_model(self, filename: str):
        path = MODELS_DIR / filename
        checkpoint = torch.load(path, map_location=self.device)
        version = checkpoint.get('model_version')
        if version != MODEL_VERSION:
            raise RuntimeError(
                f"Checkpoint at {path} has model_version={version!r}, "
                f"expected {MODEL_VERSION!r}. Retrain required."
            )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"Model loaded from {path}")


def _update_preprocessing_metadata(**updates):
    """Persist training-mode metadata alongside preprocessing metadata."""
    metadata_path = MODELS_DIR / 'preprocessing_metadata.pkl'
    if not metadata_path.exists():
        return

    metadata = joblib.load(metadata_path)
    metadata.update(updates)
    joblib.dump(metadata, metadata_path)


def train_model(
    splits: dict,
    return_scaler=None,
    use_cross_attention: bool = True,
    training_mode: str = "current",
    use_sentiment: bool = True,
    random_seed: int = RANDOM_SEED,
) -> tuple:
    """Train the multimodal next-day return model and evaluate on the test set."""
    ts_input_size = splits['train']['X_price'].shape[2]
    sentiment_input_size = (
        splits['train']['X_sentiment'].shape[2]
        if splits['train']['X_sentiment'] is not None
        else 0
    )

    print(f"Time-series features: {ts_input_size}")
    print(f"Sentiment features: {sentiment_input_size}")
    print(f"Use sentiment: {use_sentiment and sentiment_input_size > 0}")
    print(f"Use cross-attention: {use_cross_attention and sentiment_input_size > 0}")

    model = create_model(
        ts_input_size=ts_input_size,
        sentiment_input_size=sentiment_input_size,
        use_cross_attention=use_cross_attention,
    )

    _update_preprocessing_metadata(
        training_mode=training_mode,
        time_series_encoder="transformer",
        use_sentiment=use_sentiment and sentiment_input_size > 0,
        use_cross_attention=use_cross_attention and sentiment_input_size > 0,
        random_seed=random_seed,
        deterministic_training=DETERMINISTIC_TRAINING,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(model, random_seed=random_seed)

    history = trainer.fit(
        splits['train'],
        splits['val'],
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        return_scaler=return_scaler,
    )

    # Evaluate the best validation checkpoint, not the final epoch weights.
    trainer.load_model('best_model.pt')

    test_dataset = StockDataset(
        splits['test']['X_price'],
        splits['test']['X_sentiment'],
        splits['test']['y_reg'],
    )
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'])

    close_prices = splits['test'].get('close_prices', None)

    if close_prices is not None:
        print(f"Close prices available: {len(close_prices)} samples, range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
    else:
        print("Warning: close_prices not found in splits['test']")

    test_metrics = trainer.evaluate(
        test_loader,
        return_scaler=return_scaler,
        close_prices=close_prices,
    )

    print("\n" + "=" * 50)
    print("Test Set Results (best validation checkpoint):")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  RMSE: ${test_metrics['rmse']:.2f}")
    print(f"  MAE: ${test_metrics['mae']:.2f}")
    if test_metrics.get('mape') is not None:
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print("=" * 50)

    return model, trainer, history


if __name__ == "__main__":
    n_samples = 500
    seq_len = 30
    ts_features = 50
    sentiment_features = 8

    dummy_splits = {
        'train': {
            'X_price': np.random.randn(n_samples, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples).astype(np.float32),
        },
        'val': {
            'X_price': np.random.randn(n_samples // 4, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples // 4, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples // 4).astype(np.float32),
        },
        'test': {
            'X_price': np.random.randn(n_samples // 4, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples // 4, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples // 4).astype(np.float32),
        },
    }

    model, trainer, history = train_model(dummy_splits)
