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
from config import MODEL_CONFIG, TRAINING_CONFIG, MODELS_DIR


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
        # Defaults tuned for noisy daily-return targets: small eta + many
        # estimators give the booster room to find signal.
        self.params = {
            'n_estimators': kwargs.get('n_estimators', 2000),
            'max_depth': kwargs.get('max_depth', 4),
            'learning_rate': kwargs.get('learning_rate', 0.01),
            'subsample': kwargs.get('subsample', 0.8),
            'colsample_bytree': kwargs.get('colsample_bytree', 0.8),
            'min_child_weight': kwargs.get('min_child_weight', 5),
            'reg_lambda': kwargs.get('reg_lambda', 1.0),
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
    
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the XGBoost model and record validation metrics when provided."""
        from xgboost import XGBRegressor

        X = self._prepare_features(X)
        y = self._prepare_target(y)

        kwargs = dict(
            n_estimators=self.params["n_estimators"],
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            subsample=self.params["subsample"],
            colsample_bytree=self.params["colsample_bytree"],
            min_child_weight=self.params["min_child_weight"],
            reg_lambda=self.params["reg_lambda"],
            random_state=self.params["random_state"],
            n_jobs=self.params["n_jobs"],
            verbosity=0,
        )

        if X_val is not None and y_val is not None:
            X_val = self._prepare_features(X_val)
            y_val = self._prepare_target(y_val)
            kwargs["eval_metric"] = "mae"
            self.model = XGBRegressor(**kwargs)
            self.model.fit(X, y, eval_set=[(X, y), (X_val, y_val)], verbose=False)
        else:
            self.model = XGBRegressor(**kwargs)
            self.model.fit(X, y)
        return self

    def training_history(self) -> dict:
        """Return XGBoost train/validation MAE history when available."""
        if self.model is None:
            return {}
        try:
            evals_result = self.model.evals_result()
        except Exception:
            return {}

        train_mae = evals_result.get("validation_0", {}).get("mae", [])
        val_mae = evals_result.get("validation_1", {}).get("mae", [])
        return {
            "train_mae": [float(v) for v in train_mae],
            "val_mae": [float(v) for v in val_mae],
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X = self._prepare_features(X)
        return self.model.predict(X)
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (for compatibility with PyTorch models)"""
        return self.predict(X)


class MultiModelRegressor:
    """Wrapper to train and compare multiple regression models"""
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int = 60,
        random_seed: int = 42,
        device: torch.device | str | None = None,
    ):
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.random_seed = random_seed
        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
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
            'XGBoost': XGBoostRegressor(random_state=random_seed)
        }
        
        self.trained = {name: False for name in self.models}
        self.metrics = {}
    
    def train_pytorch_model(
        self,
        model: nn.Module,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = TRAINING_CONFIG['epochs'],
        batch_size: int = TRAINING_CONFIG['batch_size'],
        lr: float = TRAINING_CONFIG['learning_rate'],
        return_scaler=None,
        val_close_prices: np.ndarray = None,
        checkpoint_dir: Path = None,
    ) -> dict:
        """Train a PyTorch baseline for the configured epochs with ReduceLROnPlateau.

        Val MAE is computed in dollar space when return_scaler and val_close_prices
        are provided; otherwise it falls back to scaled-return MAE.
        """
        from torch.utils.data import TensorDataset, DataLoader

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.random_seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.SmoothL1Loss()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=TRAINING_CONFIG['lr_factor'],
            patience=TRAINING_CONFIG['lr_patience'],
        )

        best_val_mae = float('inf')
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'lr': []}

        def _val_mae_dollars():
            model.eval()
            with torch.no_grad():
                pred_scaled = model(X_val_t).cpu().numpy()
            if return_scaler is not None:
                pred_returns = return_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                actual_returns = return_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            else:
                pred_returns = pred_scaled
                actual_returns = y_val
            if val_close_prices is not None:
                pred_prices = val_close_prices * (1 + pred_returns)
                actual_prices = val_close_prices * (1 + actual_returns)
                return float(np.mean(np.abs(pred_prices - actual_prices))), float(
                    np.mean((pred_returns - actual_returns) ** 2)
                ) ** 0.5
            return float(np.mean(np.abs(pred_returns - actual_returns))), None

        def _val_loss():
            model.eval()
            with torch.no_grad():
                pred = model(X_val_t)
                return float(criterion(pred, y_val_t).item())

        print(f"  {'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Val MAE':<15} {'LR':<12} {'Status'}")
        print(f"  {'-' * 78}")

        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            val_loss = _val_loss()
            val_mae, _ = _val_mae_dollars()
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_mae)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mae'].append(val_mae)
            history['lr'].append(current_lr)

            improved = val_mae < best_val_mae
            status = ""
            if improved:
                best_val_mae = val_mae
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                status = "Best"

            if epoch % 10 == 0 or epoch == epochs - 1 or status:
                mae_str = f"${val_mae:.2f}" if val_close_prices is not None else f"{val_mae:.6f}"
                print(f"  {epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {mae_str:<15} {current_lr:<12.2e} {status}")

        # Restore best weights
        model.load_state_dict(best_state)

        # Persist best checkpoint
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, checkpoint_dir / f'{model_name.lower()}_best.pt')

        print(f"  {'-' * 78}")
        if val_close_prices is not None:
            print(f"  Best Val MAE: ${best_val_mae:.2f}")
        else:
            print(f"  Best Val MAE: {best_val_mae:.6f}")

        return history
    
    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = TRAINING_CONFIG['epochs'],
        plot_history: bool = True,
        return_scaler=None,
        val_close_prices: np.ndarray = None,
        checkpoint_dir: Path = None,
    ):
        """Train all baselines for the configured epochs."""
        self.histories = {}

        for name in ['LSTM', 'GRU', 'Transformer']:
            print("\n" + "=" * 60)
            print(f"Training {name}...")
            print("=" * 60)
            self.histories[name] = self.train_pytorch_model(
                self.models[name],
                model_name=name,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                return_scaler=return_scaler,
                val_close_prices=val_close_prices,
                checkpoint_dir=checkpoint_dir,
            )
            self.trained[name] = True

        print("\n" + "=" * 60)
        print("Training XGBoost...")
        print("=" * 60)
        self.models['XGBoost'].fit(X_train, y_train, X_val=X_val, y_val=y_val)
        self.trained['XGBoost'] = True
        print("  XGBoost training complete")

        print("\nAll models trained!")

        if plot_history:
            self.plot_training_history()
    
    def plot_training_history(self, save_path=None):
        """Plot per-model training curves.

        PyTorch models (LSTM/GRU/Transformer) put train/val loss (scaled SmoothL1) on
        the left axis and val MAE ($) on a twin right axis, since the two
        quantities differ by several orders of magnitude. XGBoost reports MAE on
        scaled returns for both train and val on a single axis.
        """
        import matplotlib.pyplot as plt

        if not self.histories:
            print("No training histories to plot.")
            return

        n_models = len(self.histories)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
        if n_models == 1:
            axes = [axes]

        for idx, (name, history) in enumerate(self.histories.items()):
            ax = axes[idx]

            if name == 'XGBoost':
                train_mae = history.get('train_mae', [])
                val_mae = history.get('val_mae', [])
                epochs = range(1, max(len(train_mae), len(val_mae)) + 1)
                if train_mae:
                    ax.plot(range(1, len(train_mae) + 1), train_mae, 'b-',
                            label='Train MAE (scaled return)', linewidth=2)
                if val_mae:
                    ax.plot(range(1, len(val_mae) + 1), val_mae, 'r-',
                            label='Val MAE (scaled return)', linewidth=2)
                ax.set_ylabel('MAE (scaled return)')
                if val_mae:
                    best_epoch = int(np.argmin(val_mae)) + 1
                    best_val = float(np.min(val_mae))
                    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
                    ax.scatter([best_epoch], [best_val], color='g', s=100, zorder=5,
                               label=f'Best: Epoch {best_epoch}')
                ax.legend(loc='best')
            else:
                train_loss = history.get('train_loss', [])
                val_loss = history.get('val_loss', [])
                val_mae = history.get('val_mae', [])
                epochs = range(1, len(train_loss) + 1)

                ax.plot(epochs, train_loss, 'b-', label='Train Loss (scaled SmoothL1)', linewidth=2)
                if val_loss:
                    ax.plot(range(1, len(val_loss) + 1), val_loss, 'b--',
                            label='Val Loss (scaled SmoothL1)', linewidth=1.5, alpha=0.7)
                ax.set_ylabel('Loss (scaled SmoothL1)', color='b')
                ax.tick_params(axis='y', labelcolor='b')

                ax2 = ax.twinx()
                ax2.plot(range(1, len(val_mae) + 1), val_mae, 'r-',
                         label='Val MAE ($)', linewidth=2)
                ax2.set_ylabel('Val MAE ($)', color='r')
                ax2.tick_params(axis='y', labelcolor='r')

                if val_mae:
                    best_epoch = int(np.argmin(val_mae)) + 1
                    best_val = float(np.min(val_mae))
                    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
                    ax2.scatter([best_epoch], [best_val], color='g', s=100, zorder=5)

                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            ax.set_title(f'{name} Training History', fontsize=14)
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = Path(save_path) if save_path is not None else Path('training_history.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nTraining history plot saved to '{out_path}'")
    
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
            
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
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
        
        # Save PyTorch models (best-val-MAE state)
        for name in ['LSTM', 'GRU', 'Transformer']:
            if self.trained[name]:
                torch.save(
                    self.models[name].state_dict(),
                    save_dir / f'{name.lower()}_best.pt'
                )

        if self.trained['XGBoost']:
            joblib.dump(self.models['XGBoost'].model, save_dir / 'xgboost_best.pkl')
        
        # Save metadata
        joblib.dump({
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'trained': self.trained,
            'metrics': self.metrics
        }, save_dir / 'multi_model_metadata.pkl')
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: Path, load_xgboost: bool = True):
        """Load trained models.

        Args:
            save_dir: Directory containing saved model artifacts.
            load_xgboost: Whether to load the pickled XGBoost artifact. Set
                this to False in UI/server contexts because incompatible
                XGBoost native libraries can segfault during unpickling.
        """
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
        
        for name in ['LSTM', 'GRU', 'Transformer']:
            if self.trained.get(name, False):
                model_path = save_dir / f'{name.lower()}_best.pt'
                if model_path.exists():
                    self.models[name].load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    self.models[name].eval()

        if not load_xgboost:
            self.trained['XGBoost'] = False

        if load_xgboost and self.trained.get('XGBoost', False):
            xgb_path = save_dir / 'xgboost_best.pkl'
            if xgb_path.exists():
                self.models['XGBoost'].model = joblib.load(xgb_path)
        
        print(f"Models loaded from {save_dir}")


if __name__ == "__main__":
    # Test the models
    batch_size = 32
    seq_len = 30
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
