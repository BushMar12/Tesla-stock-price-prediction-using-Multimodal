"""
Training utilities and training loop for the multimodal model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import joblib
import sys
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import TRAINING_CONFIG, MODELS_DIR, PREDICTION_HORIZONS
from src.models.fusion import create_model


class StockDataset(Dataset):
    """PyTorch Dataset for stock prediction"""
    
    def __init__(
        self,
        X_price: np.ndarray,
        X_sentiment: Optional[np.ndarray],
        y_reg: np.ndarray,
        y_cls: np.ndarray,
        y_multi_reg: Optional[np.ndarray] = None
    ):
        self.X_price = torch.FloatTensor(X_price)
        self.X_sentiment = torch.FloatTensor(X_sentiment) if X_sentiment is not None else None
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_cls = torch.LongTensor(y_cls)
        self.y_multi_reg = torch.FloatTensor(y_multi_reg) if y_multi_reg is not None else None
    
    def __len__(self):
        return len(self.y_reg)
    
    def __getitem__(self, idx):
        item = {
            'X_price': self.X_price[idx],
            'y_reg': self.y_reg[idx],
            'y_cls': self.y_cls[idx]
        }
        
        if self.X_sentiment is not None:
            item['X_sentiment'] = self.X_sentiment[idx]
        
        if self.y_multi_reg is not None:
            item['y_multi_reg'] = self.y_multi_reg[idx]
        
        return item


class CombinedLoss(nn.Module):
    """Combined loss for regression (single + multi-day) and classification"""
    
    def __init__(
        self,
        regression_weight: float = TRAINING_CONFIG['regression_weight'],
        classification_weight: float = TRAINING_CONFIG['classification_weight'],
        multi_day_weight: float = 0.05
    ):
        super().__init__()
        
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.multi_day_weight = multi_day_weight
        
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()  # More robust to outliers
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: dict, y_reg: torch.Tensor, y_cls: torch.Tensor,
                y_multi_reg: torch.Tensor = None) -> dict:
        """
        Calculate combined loss.
        
        Returns:
            Dict with individual and combined losses
        """
        reg_loss = self.huber_loss(outputs['regression'], y_reg)
        cls_loss = self.ce_loss(outputs['classification'], y_cls)
        
        total_loss = (self.regression_weight * reg_loss + 
                      self.classification_weight * cls_loss)
        
        # Multi-day regression loss
        multi_reg_loss = torch.tensor(0.0, device=y_reg.device)
        if y_multi_reg is not None and 'multi_regression' in outputs:
            # Mask NaN values in multi-day targets
            valid_mask = ~torch.isnan(y_multi_reg)
            if valid_mask.any():
                multi_reg_loss = self.huber_loss(
                    outputs['multi_regression'][valid_mask],
                    y_multi_reg[valid_mask]
                )
                total_loss = total_loss + self.multi_day_weight * multi_reg_loss
        
        return {
            'total': total_loss,
            'regression': reg_loss,
            'classification': cls_loss,
            'multi_regression': multi_reg_loss
        }


class Trainer:
    """Trainer class for the multimodal model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = TRAINING_CONFIG['learning_rate']
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
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Scheduler will be set in fit() once we know the number of steps
        self.scheduler = None
        
        self.criterion = CombinedLoss()
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_reg_loss': [],
            'val_reg_loss': [],
            'train_cls_loss': [],
            'val_cls_loss': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0
        n_batches = 0
        
        for batch in dataloader:
            X_price = batch['X_price'].to(self.device)
            X_sentiment = batch.get('X_sentiment')
            if X_sentiment is not None:
                X_sentiment = X_sentiment.to(self.device)
            y_reg = batch['y_reg'].to(self.device)
            y_cls = batch['y_cls'].to(self.device)
            y_multi_reg = batch.get('y_multi_reg')
            if y_multi_reg is not None:
                y_multi_reg = y_multi_reg.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(X_price, X_sentiment)
            losses = self.criterion(outputs, y_reg, y_cls, y_multi_reg)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Step the OneCycleLR scheduler per batch (if active)
            if self.scheduler is not None:
                self.scheduler.step()
            
            total_loss += losses['total'].item()
            total_reg_loss += losses['regression'].item()
            total_cls_loss += losses['classification'].item()
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'reg_loss': total_reg_loss / n_batches,
            'cls_loss': total_cls_loss / n_batches
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, return_scaler=None, close_prices=None) -> dict:
        """
        Evaluate the model.
        
        Args:
            dataloader: DataLoader with evaluation data
            return_scaler: Scaler to inverse transform returns
            close_prices: Close prices for reconstructing predicted prices
        """
        self.model.eval()
        
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
        for batch in dataloader:
            X_price = batch['X_price'].to(self.device)
            X_sentiment = batch.get('X_sentiment')
            if X_sentiment is not None:
                X_sentiment = X_sentiment.to(self.device)
            y_reg = batch['y_reg'].to(self.device)
            y_cls = batch['y_cls'].to(self.device)
            y_multi_reg = batch.get('y_multi_reg')
            if y_multi_reg is not None:
                y_multi_reg = y_multi_reg.to(self.device)
            
            outputs = self.model(X_price, X_sentiment)
            losses = self.criterion(outputs, y_reg, y_cls, y_multi_reg)
            
            total_loss += losses['total'].item()
            total_reg_loss += losses['regression'].item()
            total_cls_loss += losses['classification'].item()
            
            # Classification accuracy
            _, predicted = torch.max(outputs['classification'], 1)
            total += y_cls.size(0)
            correct += (predicted == y_cls).sum().item()
            
            all_predictions.extend(outputs['regression'].cpu().numpy())
            all_targets.extend(y_reg.cpu().numpy())
        
        n_batches = len(dataloader)
        accuracy = correct / total
        
        # Convert to arrays
        predictions_scaled = np.array(all_predictions)
        targets_scaled = np.array(all_targets)
        
        # Inverse transform to actual returns if scaler provided
        if return_scaler is not None:
            pred_returns = return_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
            actual_returns = return_scaler.inverse_transform(targets_scaled.reshape(-1, 1)).flatten()
        else:
            pred_returns = predictions_scaled
            actual_returns = targets_scaled
        
        # Reconstruct prices if close_prices provided
        if close_prices is not None:
            # Handle length mismatch (close_prices should match predictions length)
            if len(close_prices) != len(pred_returns):
                print(f"Warning: close_prices length ({len(close_prices)}) != predictions length ({len(pred_returns)})")
                # Truncate or use available data
                min_len = min(len(close_prices), len(pred_returns))
                close_prices = close_prices[:min_len]
                pred_returns = pred_returns[:min_len]
                actual_returns = actual_returns[:min_len]
            
            pred_prices = close_prices * (1 + pred_returns)
            actual_prices = close_prices * (1 + actual_returns)
            
            # Calculate metrics on actual price scale
            rmse = np.sqrt(np.mean((pred_prices - actual_prices) ** 2))
            mae = np.mean(np.abs(pred_prices - actual_prices))
            mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
        else:
            # Fallback: metrics on returns (not scaled)
            rmse = np.sqrt(np.mean((pred_returns - actual_returns) ** 2))
            mae = np.mean(np.abs(pred_returns - actual_returns))
            mape = None
        
        return {
            'loss': total_loss / n_batches,
            'reg_loss': total_reg_loss / n_batches,
            'cls_loss': total_cls_loss / n_batches,
            'accuracy': accuracy,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def fit(
        self,
        train_data: dict,
        val_data: dict,
        epochs: int = TRAINING_CONFIG['epochs'],
        batch_size: int = TRAINING_CONFIG['batch_size']
    ) -> dict:
        """
        Train the model.
        
        Args:
            train_data: Dict with training data
            val_data: Dict with validation data
            epochs: Number of epochs
            batch_size: Batch size
        
        Returns:
            Training history
        """
        # Create datasets
        train_dataset = StockDataset(
            train_data['X_price'],
            train_data['X_sentiment'],
            train_data['y_reg'],
            train_data['y_cls'],
            train_data.get('y_multi_reg')
        )
        
        val_dataset = StockDataset(
            val_data['X_price'],
            val_data['X_sentiment'],
            val_data['y_reg'],
            val_data['y_cls'],
            val_data.get('y_multi_reg')
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Set up OneCycleLR scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        best_val_loss = float('inf')
        
        print(f"Training on {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print("-" * 50)
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Save history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_reg_loss'].append(train_metrics['reg_loss'])
            self.history['val_reg_loss'].append(val_metrics['reg_loss'])
            self.history['train_cls_loss'].append(train_metrics['cls_loss'])
            self.history['val_cls_loss'].append(val_metrics['cls_loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model('best_model.pt')
        
        # Save final model
        self.save_model('final_model.pt')
        
        return self.history
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        path = MODELS_DIR / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, filename: str):
        """Load model checkpoint"""
        path = MODELS_DIR / filename
        checkpoint = torch.load(path, map_location=self.device)
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
    use_sentiment: bool = True
) -> tuple:
    """
    Train the multimodal model.
    
    Args:
        splits: Dict with train/val/test data splits
        return_scaler: Scaler for inverse transforming returns to actual values
        use_cross_attention: Whether the fusion model should use cross-attention
        training_mode: Human-readable training mode saved with metadata
        use_sentiment: Whether this run intentionally uses sentiment features
    
    Returns:
        Tuple of (trained model, trainer, history)
    """
    # Get input sizes
    ts_input_size = splits['train']['X_price'].shape[2]
    sentiment_input_size = (splits['train']['X_sentiment'].shape[2] 
                           if splits['train']['X_sentiment'] is not None else 0)
    
    print(f"Time-series features: {ts_input_size}")
    print(f"Sentiment features: {sentiment_input_size}")
    print(f"Use sentiment: {use_sentiment and sentiment_input_size > 0}")
    print(f"Use cross-attention: {use_cross_attention and sentiment_input_size > 0}")
    
    # Create model
    model = create_model(
        ts_input_size=ts_input_size,
        sentiment_input_size=sentiment_input_size,
        use_cross_attention=use_cross_attention
    )

    _update_preprocessing_metadata(
        training_mode=training_mode,
        use_sentiment=use_sentiment and sentiment_input_size > 0,
        use_cross_attention=use_cross_attention and sentiment_input_size > 0,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = Trainer(model)
    
    # Train
    history = trainer.fit(
        splits['train'],
        splits['val'],
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    # Evaluate on test set
    test_dataset = StockDataset(
        splits['test']['X_price'],
        splits['test']['X_sentiment'],
        splits['test']['y_reg'],
        splits['test']['y_cls'],
        splits['test'].get('y_multi_reg')
    )
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'])
    
    # Get close prices for reconstruction if available
    close_prices = splits['test'].get('close_prices', None)
    
    # Debug: check if close_prices exists
    if close_prices is not None:
        print(f"Close prices available: {len(close_prices)} samples, range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
    else:
        print("Warning: close_prices not found in splits['test']")
    
    if return_scaler is not None:
        print(f"Return scaler data range: [{return_scaler.data_min_[0]:.4f}, {return_scaler.data_max_[0]:.4f}]")
    
    test_metrics = trainer.evaluate(
        test_loader, 
        return_scaler=return_scaler,
        close_prices=close_prices
    )
    
    print("\n" + "=" * 50)
    print("Test Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  RMSE: ${test_metrics['rmse']:.2f}")
    print(f"  MAE: ${test_metrics['mae']:.2f}")
    if 'mape' in test_metrics and test_metrics['mape'] is not None:
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  Direction Accuracy: {test_metrics['accuracy']*100:.1f}%")
    print("=" * 50)
    
    return model, trainer, history


if __name__ == "__main__":
    # Test with dummy data
    n_samples = 500
    seq_len = 20
    ts_features = 50
    sentiment_features = 8
    n_horizons = len(PREDICTION_HORIZONS)
    
    dummy_splits = {
        'train': {
            'X_price': np.random.randn(n_samples, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples).astype(np.float32) * 100 + 200,
            'y_multi_reg': np.random.randn(n_samples, n_horizons).astype(np.float32),
            'y_cls': np.random.randint(0, 2, n_samples)
        },
        'val': {
            'X_price': np.random.randn(n_samples//4, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples//4, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples//4).astype(np.float32) * 100 + 200,
            'y_multi_reg': np.random.randn(n_samples//4, n_horizons).astype(np.float32),
            'y_cls': np.random.randint(0, 2, n_samples//4)
        },
        'test': {
            'X_price': np.random.randn(n_samples//4, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples//4, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples//4).astype(np.float32) * 100 + 200,
            'y_multi_reg': np.random.randn(n_samples//4, n_horizons).astype(np.float32),
            'y_cls': np.random.randint(0, 2, n_samples//4)
        }
    }
    
    model, trainer, history = train_model(dummy_splits)
