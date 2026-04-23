"""
Standalone regression models: LSTM, GRU, Transformer, and XGBoost for stock price prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODEL_CONFIG


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMRegressor(nn.Module):
    """LSTM-based regression model for price prediction"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.model_name = "LSTM"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layers
        lstm_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_projection(x)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last output
        if self.bidirectional:
            last_out = torch.cat([lstm_out[:, -1, :self.hidden_size], 
                                  lstm_out[:, 0, self.hidden_size:]], dim=1)
        else:
            last_out = lstm_out[:, -1, :]
        
        # Predict
        return self.fc(last_out).squeeze(-1)


class GRURegressor(nn.Module):
    """GRU-based regression model for price prediction"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.model_name = "GRU"
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layers
        gru_output_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_projection(x)
        
        # GRU
        gru_out, h_n = self.gru(x)
        
        # Use last output
        if self.bidirectional:
            last_out = torch.cat([gru_out[:, -1, :self.hidden_size], 
                                  gru_out[:, 0, self.hidden_size:]], dim=1)
        else:
            last_out = gru_out[:, -1, :]
        
        # Predict
        return self.fc(last_out).squeeze(-1)


class TransformerRegressor(nn.Module):
    """Transformer-based regression model for price prediction"""
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = "Transformer"
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=200, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Use last token embedding for prediction
        last_out = x[:, -1, :]
        
        return self.fc(last_out).squeeze(-1)


class XGBoostRegressor:
    """XGBoost-based regression model for price prediction"""
    
    def __init__(self, **kwargs):
        self.model_name = "XGBoost"
        self.model = None
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 400),
            'max_depth': kwargs.get('max_depth', 4),
            'learning_rate': kwargs.get('learning_rate', 0.05),
            'subsample': kwargs.get('subsample', 0.8),
            'random_state': kwargs.get('random_state', 42),
            'n_jobs': kwargs.get('n_jobs', 1),
        }

    @staticmethod
    def _prepare_features(X: np.ndarray) -> np.ndarray:
        """Flatten sequence data and hand XGBoost a native-friendly array."""
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        X = np.ascontiguousarray(X, dtype=np.float32)
        if not np.isfinite(X).all():
            raise ValueError("XGBoost input contains NaN or infinite values")
        return X

    @staticmethod
    def _prepare_target(y: np.ndarray) -> np.ndarray:
        y = np.ascontiguousarray(np.asarray(y).reshape(-1), dtype=np.float32)
        if not np.isfinite(y).all():
            raise ValueError("XGBoost target contains NaN or infinite values")
        return y
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the XGBoost model"""
        from xgboost import XGBRegressor

        X = self._prepare_features(X)
        y = self._prepare_target(y)

        self.model = XGBRegressor(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            subsample=self.params["subsample"],
            random_state=self.params["random_state"],
            n_jobs=self.params["n_jobs"],
            verbosity=0,
        )
        self.model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X = self._prepare_features(X)
        return self.model.predict(X)
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (for compatibility with PyTorch models)"""
        return self.predict(X)


class MultiModelRegressor:
    """Wrapper to train and compare multiple regression models"""
    
    def __init__(self, input_size: int, sequence_length: int = 60):
        self.input_size = input_size
        self.sequence_length = sequence_length
        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Initialize models (now includes Transformer)
        self.models = {
            'LSTM': LSTMRegressor(input_size).to(self.device),
            'GRU': GRURegressor(input_size).to(self.device),
            'Transformer': TransformerRegressor(input_size).to(self.device),
            'XGBoost': XGBoostRegressor()
        }
        
        self.trained = {name: False for name in self.models}
        self.metrics = {}
    
    def train_pytorch_model(
        self, 
        model: nn.Module, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3
    ) -> dict:
        """Train a PyTorch model for the full configured epoch count."""
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.SmoothL1Loss()  # Huber loss — matches the fusion model
        
        # OneCycleLR for baselines too
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 10, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}
        
        print(f"  {'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Status'}")
        print(f"  {'-'*50}")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_t)
                val_loss = criterion(val_predictions, y_val_t).item()
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Check for improvement
            status = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                status = "✓ Best"
            
            # Print progress every 10 epochs or first/last
            if epoch % 10 == 0 or epoch == epochs - 1 or status:
                print(f"  {epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {status}")
        
        print(f"  {'-'*50}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        
        return history
    
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        plot_history: bool = True
    ):
        """Train all models"""
        self.histories = {}
        
        print("\n" + "="*60)
        print("Training LSTM...")
        print("="*60)
        self.histories['LSTM'] = self.train_pytorch_model(
            self.models['LSTM'], X_train, y_train, X_val, y_val, epochs
        )
        self.trained['LSTM'] = True
        
        print("\n" + "="*60)
        print("Training GRU...")
        print("="*60)
        self.histories['GRU'] = self.train_pytorch_model(
            self.models['GRU'], X_train, y_train, X_val, y_val, epochs
        )
        self.trained['GRU'] = True
        
        print("\n" + "="*60)
        print("Training Transformer...")
        print("="*60)
        self.histories['Transformer'] = self.train_pytorch_model(
            self.models['Transformer'], X_train, y_train, X_val, y_val, epochs
        )
        self.trained['Transformer'] = True
        
        print("\n" + "="*60)
        print("Training XGBoost...")
        print("="*60)
        self.models['XGBoost'].fit(X_train, y_train)
        self.trained['XGBoost'] = True
        print("  XGBoost training complete (no epochs - gradient boosting)")
        
        print("\n✅ All models trained!")
        
        # Plot training history
        if plot_history:
            self.plot_training_history()
    
    def plot_training_history(self):
        """Plot training and validation loss curves"""
        import matplotlib.pyplot as plt
        
        n_models = len(self.histories)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, history) in enumerate(self.histories.items()):
            ax = axes[idx]
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax.set_title(f'{name} Training History', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (Huber)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Mark best epoch
            best_epoch = np.argmin(history['val_loss']) + 1
            best_val = min(history['val_loss'])
            ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best: Epoch {best_epoch}')
            ax.scatter([best_epoch], [best_val], color='g', s=100, zorder=5)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("\n📊 Training history plot saved to 'training_history.png'")
    
    def evaluate_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        return_scaler=None,
        close_prices: np.ndarray = None
    ) -> dict:
        """
        Evaluate all models and return comparison metrics.
        
        Args:
            X_test: Test features
            y_test: Test targets (scaled returns)
            return_scaler: Scaler to inverse transform returns
            close_prices: Today's close prices for reconstructing predicted prices
        """
        results = {}
        
        for name, model in self.models.items():
            if not self.trained[name]:
                continue
            
            # Get predictions (in scaled return space)
            if name == 'XGBoost':
                predictions_scaled = model.predict(X_test)
            else:
                model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X_test).to(self.device)
                    predictions_scaled = model(X_t).cpu().numpy()
            
            # Inverse transform to actual returns
            if return_scaler is not None:
                pred_returns = return_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
                actual_returns = return_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            else:
                pred_returns = predictions_scaled
                actual_returns = y_test
            
            # Reconstruct prices: predicted_price = close_price * (1 + predicted_return)
            if close_prices is not None:
                pred_prices = close_prices * (1 + pred_returns)
                actual_prices = close_prices * (1 + actual_returns)
                
                # Calculate metrics on actual price scale
                mse = np.mean((pred_prices - actual_prices) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(pred_prices - actual_prices))
                mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
            else:
                # Fallback: metrics on returns (less interpretable)
                mse = np.mean((pred_returns - actual_returns) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(pred_returns - actual_returns))
                mape = np.nan
                pred_prices = pred_returns
                actual_prices = actual_returns
            
            # Directional accuracy: did we predict the right direction?
            actual_direction = np.sign(actual_returns)
            pred_direction = np.sign(pred_returns)
            dir_acc = np.mean(actual_direction == pred_direction) * 100
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'Directional_Accuracy': dir_acc,
                'predictions': pred_prices,
                'pred_returns': pred_returns,
                'predictions_scaled': predictions_scaled
            }
        
        self.metrics = results
        return results
    
    def predict_all(self, X: np.ndarray, return_scaler=None, current_price: float = None) -> dict:
        """
        Get predictions from all models.
        
        Args:
            X: Input features
            return_scaler: Scaler to inverse transform predicted returns
            current_price: Today's close price for reconstructing predicted price
        
        Returns:
            Dict with predicted prices (or returns if no current_price provided)
        """
        predictions = {}
        
        for name, model in self.models.items():
            if not self.trained[name]:
                continue
            
            if name == 'XGBoost':
                pred_scaled = model.predict(X)
            else:
                model.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X).to(self.device)
                    pred_scaled = model(X_t).cpu().numpy()
            
            # Get scalar value
            pred_scaled = float(pred_scaled[0]) if len(pred_scaled.shape) > 0 and pred_scaled.shape[0] == 1 else pred_scaled
            
            # Inverse transform to actual return
            if return_scaler is not None:
                pred_return = return_scaler.inverse_transform([[pred_scaled]])[0, 0]
            else:
                pred_return = pred_scaled
            
            # Reconstruct price: predicted_price = current_price * (1 + predicted_return)
            if current_price is not None:
                pred_price = current_price * (1 + pred_return)
                predictions[name] = pred_price
            else:
                predictions[name] = pred_return
        
        return predictions
    
    def save_models(self, save_dir: Path):
        """Save all trained models"""
        import joblib
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch models
        for name in ['LSTM', 'GRU', 'Transformer']:
            if self.trained[name]:
                torch.save(
                    self.models[name].state_dict(),
                    save_dir / f'{name.lower()}_regressor.pt'
                )
        
        # Save XGBoost model
        if self.trained['XGBoost']:
            joblib.dump(self.models['XGBoost'].model, save_dir / 'xgboost_regressor.pkl')
        
        # Save metadata
        joblib.dump({
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'trained': self.trained,
            'metrics': self.metrics
        }, save_dir / 'multi_model_metadata.pkl')
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: Path):
        """Load all trained models"""
        import joblib
        
        save_dir = Path(save_dir)
        
        # Load metadata
        metadata = joblib.load(save_dir / 'multi_model_metadata.pkl')
        self.input_size = metadata['input_size']
        self.sequence_length = metadata['sequence_length']
        self.trained = metadata['trained']
        self.metrics = metadata.get('metrics', {})
        
        # Reinitialize models with correct input size
        self.models = {
            'LSTM': LSTMRegressor(self.input_size).to(self.device),
            'GRU': GRURegressor(self.input_size).to(self.device),
            'Transformer': TransformerRegressor(self.input_size).to(self.device),
            'XGBoost': XGBoostRegressor()
        }
        
        # Load PyTorch models
        for name in ['LSTM', 'GRU', 'Transformer']:
            if self.trained.get(name, False):
                model_path = save_dir / f'{name.lower()}_regressor.pt'
                if model_path.exists():
                    self.models[name].load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    self.models[name].eval()
        
        # Load XGBoost model
        if self.trained.get('XGBoost', False):
            xgb_path = save_dir / 'xgboost_regressor.pkl'
            if xgb_path.exists():
                self.models['XGBoost'].model = joblib.load(xgb_path)
        
        print(f"Models loaded from {save_dir}")


if __name__ == "__main__":
    # Test the models
    batch_size = 32
    seq_len = 60
    input_size = 50
    
    # Test LSTM
    lstm = LSTMRegressor(input_size=input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    out = lstm(x)
    print(f"LSTM - Input: {x.shape}, Output: {out.shape}")
    
    # Test GRU
    gru = GRURegressor(input_size=input_size)
    out = gru(x)
    print(f"GRU - Input: {x.shape}, Output: {out.shape}")
    
    # Test Transformer
    transformer = TransformerRegressor(input_size=input_size)
    out = transformer(x)
    print(f"Transformer - Input: {x.shape}, Output: {out.shape}")
    
    # Test XGBoost
    xgb = XGBoostRegressor()
    X_np = x.numpy()
    y_np = np.random.randn(batch_size)
    xgb.fit(X_np, y_np)
    pred = xgb.predict(X_np)
    print(f"XGBoost - Input: {X_np.shape}, Output: {pred.shape}")
