"""
Multimodal fusion model combining time-series and sentiment encoders
"""
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODEL_CONFIG
from src.models.time_series import TimeSeriesEncoder
from src.models.text_encoder import SentimentEncoder, TemporalSentimentEncoder


class CrossModalAttention(nn.Module):
    """Cross-modal attention between time-series sequence and sentiment features.
    
    Fixed: Now operates on full temporal sequences instead of single vectors,
    avoiding the degenerate scalar-softmax problem.
    """
    
    def __init__(self, ts_dim: int, sentiment_dim: int, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        
        # Project sentiment to match ts_dim for multi-head attention
        self.sent_proj = nn.Linear(sentiment_dim, ts_dim)
        self.mha = nn.MultiheadAttention(ts_dim, n_heads, batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(ts_dim)
    
    def forward(self, ts_sequence: torch.Tensor, sentiment_features: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-modal multi-head attention.
        
        Args:
            ts_sequence: Full time-series encoder sequence (batch, seq_len, ts_dim)
            sentiment_features: Sentiment features (batch, sentiment_dim)
        
        Returns:
            Attended and pooled features (batch, ts_dim)
        """
        # Expand sentiment to a pseudo-sequence of length 1 for K/V
        sent_proj = self.sent_proj(sentiment_features).unsqueeze(1)  # (batch, 1, ts_dim)
        
        # Multi-head attention: Q = time-series sequence, K/V = sentiment
        attn_out, _ = self.mha(ts_sequence, sent_proj.expand(-1, ts_sequence.size(1), -1), 
                                sent_proj.expand(-1, ts_sequence.size(1), -1))
        
        # Residual + layer norm
        attn_out = self.layer_norm(ts_sequence + attn_out)
        
        # Pool across time
        return attn_out.mean(dim=1)


class MultimodalFusionModel(nn.Module):
    """Two-stream multimodal model predicting next-day return."""

    def __init__(
        self,
        ts_input_size: int,
        sentiment_input_size: int,
        ts_hidden_size: int = MODEL_CONFIG['ts_hidden_size'],
        sentiment_hidden_size: int = MODEL_CONFIG['sentiment_hidden_dim'],
        fusion_hidden_size: int = MODEL_CONFIG['fusion_hidden_dim'],
        dropout: float = MODEL_CONFIG['fusion_dropout'],
        use_cross_attention: bool = True,
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
            
            # Cross-modal attention (fixed: now uses multi-head attention)
            if use_cross_attention:
                self.cross_attention = CrossModalAttention(
                    ts_dim=self.ts_encoder.output_size,
                    sentiment_dim=sentiment_hidden_size,
                    hidden_dim=fusion_hidden_size,
                    n_heads=4
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
        
        # Next-day return regression head — single output
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_hidden_size // 2, fusion_hidden_size // 4),
            nn.ReLU(),
            nn.Linear(fusion_hidden_size // 4, 1)
        )

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

        Returns dict with:
            - 'regression': next-day return prediction (batch,)
            - 'ts_attention': time-series attention weights
            - 'fused_features': fused representation (for downstream analysis)
        """
        ts_encoded, ts_attention, ts_full_seq = self.ts_encoder.encode_with_attention(ts_input)

        if self.sentiment_encoder is not None and sentiment_input is not None:
            sentiment_encoded = self.sentiment_encoder(sentiment_input)

            if self.use_cross_attention:
                ts_cross_attended = self.cross_attention(ts_full_seq, sentiment_encoded)
                ts_encoded = ts_encoded + ts_cross_attended

            fused = torch.cat([ts_encoded, sentiment_encoded], dim=-1)
        else:
            fused = ts_encoded

        fused = self.fusion(fused)
        regression_output = self.regression_head(fused).squeeze(-1)

        return {
            'regression': regression_output,
            'ts_attention': ts_attention,
            'fused_features': fused
        }

    def predict(
        self,
        ts_input: torch.Tensor,
        sentiment_input: torch.Tensor = None
    ) -> dict:
        """Inference helper. Returns next-day return prediction and attention."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(ts_input, sentiment_input)
            return {
                'price': outputs['regression'],
                'attention': outputs['ts_attention']
            }


class EnsembleModel(nn.Module):
    """Ensemble that averages next-day return predictions across models."""

    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, ts_input: torch.Tensor, sentiment_input: torch.Tensor = None) -> dict:
        all_reg = [m(ts_input, sentiment_input)['regression'] for m in self.models]
        return {'regression': torch.stack(all_reg).mean(dim=0)}


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
    print(f"Attention weights: {outputs['ts_attention'].shape}")
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
