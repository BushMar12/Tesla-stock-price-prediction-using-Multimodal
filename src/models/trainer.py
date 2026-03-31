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
from config import TRAINING_CONFIG, MODELS_DIR
from src.models.fusion import create_model


class StockDataset(Dataset):
    """PyTorch Dataset for stock prediction"""
    
    def __init__(
        self,
        X_price: np.ndarray,
        X_sentiment: Optional[np.ndarray],
        y_reg: np.ndarray,
        y_cls: np.ndarray
    ):
        self.X_price = torch.FloatTensor(X_price)
        self.X_sentiment = torch.FloatTensor(X_sentiment) if X_sentiment is not None else None
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_cls = torch.LongTensor(y_cls)
    
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
        
        return item


class CombinedLoss(nn.Module):
    """Combined loss for regression and classification"""
    
    def __init__(
        self,
        regression_weight: float = TRAINING_CONFIG['regression_weight'],
        classification_weight: float = TRAINING_CONFIG['classification_weight']
    ):
        super().__init__()
        
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: dict, y_reg: torch.Tensor, y_cls: torch.Tensor) -> dict:
        """
        Calculate combined loss.
        
        Returns:
            Dict with individual and combined losses
        """
        reg_loss = self.mse_loss(outputs['regression'], y_reg)
        cls_loss = self.ce_loss(outputs['classification'], y_cls)
        
        total_loss = (self.regression_weight * reg_loss + 
                      self.classification_weight * cls_loss)
        
        return {
            'total': total_loss,
            'regression': reg_loss,
            'classification': cls_loss
        }


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = TRAINING_CONFIG['early_stopping_patience'], min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class Trainer:
    """Trainer class for the multimodal model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        learning_rate: float = TRAINING_CONFIG['learning_rate']
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.criterion = CombinedLoss()
        self.early_stopping = EarlyStopping()
        
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
            
            self.optimizer.zero_grad()
            
            outputs = self.model(X_price, X_sentiment)
            losses = self.criterion(outputs, y_reg, y_cls)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
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
    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate the model"""
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
            
            outputs = self.model(X_price, X_sentiment)
            losses = self.criterion(outputs, y_reg, y_cls)
            
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
        
        # Calculate RMSE
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))
        mae = np.mean(np.abs(predictions - targets))
        
        return {
            'loss': total_loss / n_batches,
            'reg_loss': total_reg_loss / n_batches,
            'cls_loss': total_cls_loss / n_batches,
            'accuracy': accuracy,
            'rmse': rmse,
            'mae': mae
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
            train_data['y_cls']
        )
        
        val_dataset = StockDataset(
            val_data['X_price'],
            val_data['X_sentiment'],
            val_data['y_reg'],
            val_data['y_cls']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
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
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
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
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"RMSE: {val_metrics['rmse']:.2f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model('best_model.pt')
            
            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"Early stopping at epoch {epoch+1}")
                break
        
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


def train_model(splits: dict) -> tuple:
    """
    Train the multimodal model.
    
    Args:
        splits: Dict with train/val/test data splits
    
    Returns:
        Tuple of (trained model, trainer, history)
    """
    # Get input sizes
    ts_input_size = splits['train']['X_price'].shape[2]
    sentiment_input_size = (splits['train']['X_sentiment'].shape[2] 
                           if splits['train']['X_sentiment'] is not None else 0)
    
    print(f"Time-series features: {ts_input_size}")
    print(f"Sentiment features: {sentiment_input_size}")
    
    # Create model
    model = create_model(
        ts_input_size=ts_input_size,
        sentiment_input_size=sentiment_input_size
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
        splits['test']['y_cls']
    )
    test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'])
    
    test_metrics = trainer.evaluate(test_loader)
    print("\n" + "=" * 50)
    print("Test Set Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.2f}")
    print(f"  MAE: {test_metrics['mae']:.2f}")
    print(f"  Direction Accuracy: {test_metrics['accuracy']:.4f}")
    print("=" * 50)
    
    return model, trainer, history


if __name__ == "__main__":
    # Test with dummy data
    n_samples = 500
    seq_len = 60
    ts_features = 50
    sentiment_features = 8
    
    dummy_splits = {
        'train': {
            'X_price': np.random.randn(n_samples, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples).astype(np.float32) * 100 + 200,
            'y_cls': np.random.randint(0, 3, n_samples)
        },
        'val': {
            'X_price': np.random.randn(n_samples//4, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples//4, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples//4).astype(np.float32) * 100 + 200,
            'y_cls': np.random.randint(0, 3, n_samples//4)
        },
        'test': {
            'X_price': np.random.randn(n_samples//4, seq_len, ts_features).astype(np.float32),
            'X_sentiment': np.random.randn(n_samples//4, seq_len, sentiment_features).astype(np.float32),
            'y_reg': np.random.randn(n_samples//4).astype(np.float32) * 100 + 200,
            'y_cls': np.random.randint(0, 3, n_samples//4)
        }
    }
    
    model, trainer, history = train_model(dummy_splits)
