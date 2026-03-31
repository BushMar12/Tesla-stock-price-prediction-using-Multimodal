"""
Multimodal fusion model combining time-series and sentiment encoders
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODEL_CONFIG
from src.models.time_series import TimeSeriesEncoder
from src.models.text_encoder import SentimentEncoder, TemporalSentimentEncoder


class CrossModalAttention(nn.Module):
    """Cross-modal attention between time-series and sentiment features"""
    
    def __init__(self, ts_dim: int, sentiment_dim: int, hidden_dim: int):
        super().__init__()
        
        self.query = nn.Linear(ts_dim, hidden_dim)
        self.key = nn.Linear(sentiment_dim, hidden_dim)
        self.value = nn.Linear(sentiment_dim, hidden_dim)
        
        self.scale = hidden_dim ** -0.5
        
        self.output_projection = nn.Linear(hidden_dim, ts_dim)
    
    def forward(self, ts_features: torch.Tensor, sentiment_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            ts_features: Time-series features (batch, ts_dim)
            sentiment_features: Sentiment features (batch, sentiment_dim)
        
        Returns:
            Attended features (batch, ts_dim)
        """
        Q = self.query(ts_features).unsqueeze(1)  # (batch, 1, hidden)
        K = self.key(sentiment_features).unsqueeze(1)  # (batch, 1, hidden)
        V = self.value(sentiment_features).unsqueeze(1)  # (batch, 1, hidden)
        
        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V).squeeze(1)
        
        return self.output_projection(out)


class MultimodalFusionModel(nn.Module):
    """
    Multimodal model combining time-series and sentiment for stock prediction.
    Supports both regression (price prediction) and classification (direction).
    """
    
    def __init__(
        self,
        ts_input_size: int,
        sentiment_input_size: int,
        ts_hidden_size: int = MODEL_CONFIG['ts_hidden_size'],
        sentiment_hidden_size: int = MODEL_CONFIG['sentiment_hidden_dim'],
        fusion_hidden_size: int = MODEL_CONFIG['fusion_hidden_dim'],
        num_classes: int = MODEL_CONFIG['num_classes'],
        dropout: float = MODEL_CONFIG['fusion_dropout'],
        use_cross_attention: bool = True
    ):
        super().__init__()
        
        self.use_cross_attention = use_cross_attention
        self.sentiment_input_size = sentiment_input_size
        
        # Time-series encoder
        self.ts_encoder = TimeSeriesEncoder(
            input_size=ts_input_size,
            hidden_size=ts_hidden_size,
            num_layers=MODEL_CONFIG['ts_num_layers'],
            dropout=MODEL_CONFIG['ts_dropout'],
            bidirectional=MODEL_CONFIG['ts_bidirectional']
        )
        
        # Sentiment encoder (if sentiment features available)
        if sentiment_input_size > 0:
            self.sentiment_encoder = TemporalSentimentEncoder(
                input_size=sentiment_input_size,
                hidden_dim=sentiment_hidden_size
            )
            
            # Cross-modal attention
            if use_cross_attention:
                self.cross_attention = CrossModalAttention(
                    ts_dim=self.ts_encoder.output_size,
                    sentiment_dim=sentiment_hidden_size,
                    hidden_dim=fusion_hidden_size
                )
            
            fusion_input_size = self.ts_encoder.output_size + sentiment_hidden_size
        else:
            self.sentiment_encoder = None
            fusion_input_size = self.ts_encoder.output_size
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden_size),
            nn.LayerNorm(fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(fusion_hidden_size, fusion_hidden_size // 2),
            nn.LayerNorm(fusion_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Regression head (price prediction)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_hidden_size // 2, fusion_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size // 4, 1)
        )
        
        # Classification head (direction prediction)
        self.classification_head = nn.Sequential(
            nn.Linear(fusion_hidden_size // 2, fusion_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size // 4, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        ts_input: torch.Tensor,
        sentiment_input: torch.Tensor = None
    ) -> dict:
        """
        Forward pass.
        
        Args:
            ts_input: Time-series input (batch, seq_len, ts_features)
            sentiment_input: Sentiment input (batch, seq_len, sentiment_features)
        
        Returns:
            Dict with regression output, classification logits, and attention weights
        """
        # Encode time-series
        ts_encoded, ts_attention = self.ts_encoder(ts_input)
        
        # Encode sentiment if available
        if self.sentiment_encoder is not None and sentiment_input is not None:
            sentiment_encoded = self.sentiment_encoder(sentiment_input)
            
            # Cross-modal attention
            if self.use_cross_attention:
                ts_attended = self.cross_attention(ts_encoded, sentiment_encoded)
                ts_encoded = ts_encoded + ts_attended  # Residual connection
            
            # Concatenate for fusion
            fused = torch.cat([ts_encoded, sentiment_encoded], dim=-1)
        else:
            fused = ts_encoded
        
        # Fusion
        fused = self.fusion(fused)
        
        # Output heads
        regression_output = self.regression_head(fused).squeeze(-1)
        classification_logits = self.classification_head(fused)
        
        return {
            'regression': regression_output,
            'classification': classification_logits,
            'ts_attention': ts_attention,
            'fused_features': fused
        }
    
    def predict(
        self,
        ts_input: torch.Tensor,
        sentiment_input: torch.Tensor = None
    ) -> dict:
        """
        Make predictions (inference mode).
        
        Returns:
            Dict with predicted price and direction
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(ts_input, sentiment_input)
            
            direction_probs = F.softmax(outputs['classification'], dim=-1)
            direction_pred = torch.argmax(direction_probs, dim=-1)
            
            return {
                'price': outputs['regression'],
                'direction': direction_pred,
                'direction_probs': direction_probs,
                'attention': outputs['ts_attention']
            }


class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for more robust predictions.
    """
    
    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, ts_input: torch.Tensor, sentiment_input: torch.Tensor = None) -> dict:
        """Average predictions from all models"""
        all_reg = []
        all_cls = []
        
        for model in self.models:
            out = model(ts_input, sentiment_input)
            all_reg.append(out['regression'])
            all_cls.append(out['classification'])
        
        return {
            'regression': torch.stack(all_reg).mean(dim=0),
            'classification': torch.stack(all_cls).mean(dim=0)
        }


def create_model(
    ts_input_size: int,
    sentiment_input_size: int = 0,
    **kwargs
) -> MultimodalFusionModel:
    """
    Factory function to create the fusion model.
    
    Args:
        ts_input_size: Number of time-series features
        sentiment_input_size: Number of sentiment features
    
    Returns:
        Initialized model
    """
    return MultimodalFusionModel(
        ts_input_size=ts_input_size,
        sentiment_input_size=sentiment_input_size,
        **kwargs
    )


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    seq_len = 60
    ts_features = 50
    sentiment_features = 8
    
    model = create_model(
        ts_input_size=ts_features,
        sentiment_input_size=sentiment_features
    )
    
    ts_input = torch.randn(batch_size, seq_len, ts_features)
    sentiment_input = torch.randn(batch_size, seq_len, sentiment_features)
    
    outputs = model(ts_input, sentiment_input)
    
    print(f"Time-series input: {ts_input.shape}")
    print(f"Sentiment input: {sentiment_input.shape}")
    print(f"Regression output: {outputs['regression'].shape}")
    print(f"Classification output: {outputs['classification'].shape}")
    print(f"Attention weights: {outputs['ts_attention'].shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
