"""
Time-series encoders for stock sequence features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODEL_CONFIG


class Attention(nn.Module):
    """Attention pooling mechanism for sequence models."""
    
    def __init__(self, hidden_size: int, bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size * (2 if bidirectional else 1)
        
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1)
        )
    
    def forward(self, sequence_output: torch.Tensor) -> tuple:
        """
        Apply attention pooling to sequence output.
        
        Args:
            sequence_output: (batch, seq_len, hidden_size)
        
        Returns:
            context: (batch, hidden_size)
            attention_weights: (batch, seq_len)
        """
        # Calculate attention weights
        attention_scores = self.attention(sequence_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights
        context = torch.sum(attention_weights * sequence_output, dim=1)  # (batch, hidden_size)
        
        return context, attention_weights.squeeze(-1)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer sequence models."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeSeriesEncoder(nn.Module):
    """
    Transformer-based encoder for time-series data with attention pooling.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_CONFIG['ts_hidden_size'],
        num_layers: int = MODEL_CONFIG['ts_num_layers'],
        dropout: float = MODEL_CONFIG['ts_dropout'],
        bidirectional: bool = MODEL_CONFIG['ts_bidirectional']
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.num_directions
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, self.output_size),
            nn.LayerNorm(self.output_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.pos_encoding = PositionalEncoding(
            d_model=self.output_size,
            max_len=500,
            dropout=dropout
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.output_size,
            nhead=4,
            dim_feedforward=self.output_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Attention layer
        self.attention = Attention(hidden_size, bidirectional)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size)

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Return the full encoded sequence for cross-modal attention."""
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        return self.transformer(x)

    def encode_with_attention(self, x: torch.Tensor) -> tuple:
        """
        Return pooled output, attention weights, and the full encoded sequence.
        """
        sequence_out = self.encode_sequence(x)
        context, attention_weights = self.attention(sequence_out)
        output = self.layer_norm(context)
        return output, attention_weights, sequence_out
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
        
        Returns:
            output: Encoded representation (batch, output_size)
            attention_weights: Attention weights (batch, seq_len)
        """
        output, attention_weights, _ = self.encode_with_attention(x)
        
        return output, attention_weights


class GRUEncoder(nn.Module):
    """
    GRU-based encoder as an alternative to LSTM.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = MODEL_CONFIG['ts_hidden_size'],
        num_layers: int = MODEL_CONFIG['ts_num_layers'],
        dropout: float = MODEL_CONFIG['ts_dropout'],
        bidirectional: bool = MODEL_CONFIG['ts_bidirectional']
    ):
        super().__init__()
        
        self.input_size = input_size
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
        
        # Attention layer
        self.attention = Attention(hidden_size, bidirectional)
        
        # Output size
        self.output_size = hidden_size * self.num_directions
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_size)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass"""
        # Project input
        x = self.input_projection(x)
        
        # GRU
        gru_out, h_n = self.gru(x)
        
        # Apply attention
        context, attention_weights = self.attention(gru_out)
        
        # Layer normalization
        output = self.layer_norm(context)
        
        return output, attention_weights


if __name__ == "__main__":
    # Test the encoder
    batch_size = 32
    seq_len = 20
    input_size = 50
    
    model = TimeSeriesEncoder(input_size=input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    
    output, attn_weights = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
