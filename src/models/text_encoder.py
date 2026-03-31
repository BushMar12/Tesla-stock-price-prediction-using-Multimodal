"""
Text/Sentiment encoder using dense layers
(Optionally using FinBERT for embedding extraction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config import MODEL_CONFIG


class SentimentEncoder(nn.Module):
    """
    Encoder for sentiment features.
    Works with pre-computed sentiment scores (VADER/FinBERT).
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_dim: int = MODEL_CONFIG['sentiment_hidden_dim'],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Multi-layer encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_size = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Sentiment features (batch, seq_len, input_size) or (batch, input_size)
        
        Returns:
            Encoded sentiment representation
        """
        # If sequence input, take mean across time
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        
        return self.encoder(x)


class TemporalSentimentEncoder(nn.Module):
    """
    Encoder that preserves temporal information in sentiment data.
    Uses 1D CNN to capture local patterns.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_dim: int = MODEL_CONFIG['sentiment_hidden_dim'],
        kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        
        # Multiple CNN paths for different receptive fields
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_size, hidden_dim // len(kernel_sizes), 
                         kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_dim // len(kernel_sizes)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])
        
        # Combine CNN outputs
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.output_size = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Sentiment features (batch, seq_len, input_size)
        
        Returns:
            Encoded sentiment representation (batch, hidden_dim)
        """
        # Transpose for conv1d: (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Apply each conv and pool
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)
            # Global average pooling
            out = out.mean(dim=2)
            conv_outputs.append(out)
        
        # Concatenate
        combined = torch.cat(conv_outputs, dim=1)
        
        return self.combine(combined)


class FinBERTEncoder(nn.Module):
    """
    Text encoder using pre-trained FinBERT.
    Use this when you have raw text headlines.
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        hidden_dim: int = MODEL_CONFIG['sentiment_hidden_dim'],
        freeze_bert: bool = True
    ):
        super().__init__()
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)
            
            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
            
            self.embedding_dim = self.bert.config.hidden_size
            
        except Exception as e:
            print(f"Failed to load FinBERT: {e}")
            print("Using placeholder encoder")
            self.bert = None
            self.embedding_dim = 768
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.output_size = hidden_dim
    
    def encode_text(self, texts: list, device: torch.device) -> torch.Tensor:
        """
        Encode raw text using FinBERT.
        
        Args:
            texts: List of text strings
            device: Device to use
        
        Returns:
            Embeddings tensor
        """
        if self.bert is None:
            return torch.randn(len(texts), self.embedding_dim, device=device)
        
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
        
        # Use CLS token embedding
        return outputs.last_hidden_state[:, 0, :]
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-computed embeddings.
        
        Args:
            embeddings: (batch, embedding_dim)
        
        Returns:
            Projected embeddings (batch, hidden_dim)
        """
        return self.projection(embeddings)


if __name__ == "__main__":
    # Test the encoders
    batch_size = 32
    seq_len = 60
    input_size = 8  # Sentiment features
    
    # Test SentimentEncoder
    encoder = SentimentEncoder(input_size=input_size)
    x = torch.randn(batch_size, seq_len, input_size)
    
    output = encoder(x)
    print(f"SentimentEncoder - Input: {x.shape}, Output: {output.shape}")
    
    # Test TemporalSentimentEncoder
    temporal_encoder = TemporalSentimentEncoder(input_size=input_size)
    output = temporal_encoder(x)
    print(f"TemporalSentimentEncoder - Input: {x.shape}, Output: {output.shape}")
