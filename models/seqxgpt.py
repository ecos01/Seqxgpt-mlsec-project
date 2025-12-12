"""
SeqXGPT Model
CNN + Self-Attention architecture for AI text detection using LLM log-probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SeqXGPTModel(nn.Module):
    """
    SeqXGPT: Sequence-based detection model with CNN and self-attention.
    Input: Log-probability features from LLM [batch_size, seq_len, feature_dim]
    Output: Binary classification (0=human, 1=AI)
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # Number of features (log_prob, surprisal, entropy)
        hidden_dim: int = 128,
        num_cnn_layers: int = 3,
        kernel_size: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.3,
        max_seq_length: int = 256
    ):
        """
        Args:
            input_dim: Number of input features per position
            hidden_dim: Hidden dimension for CNN and attention
            num_cnn_layers: Number of CNN layers
            kernel_size: Kernel size for 1D convolution
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_length: Maximum sequence length
        """
        super(SeqXGPTModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        for i in range(num_cnn_layers):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Global pooling (attention-weighted)
        self.pool_attention = nn.Linear(hidden_dim, 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Logits for binary classification [batch_size, 1]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)  # [B, L, H]
        
        # CNN layers (need to transpose for Conv1d)
        x_cnn = x.transpose(1, 2)  # [B, H, L]
        for cnn_layer in self.cnn_layers:
            x_cnn = cnn_layer(x_cnn) + x_cnn  # Residual connection
        x = x_cnn.transpose(1, 2)  # [B, L, H]
        
        # Clean any NaN that might have appeared
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        
        # Self-attention
        if mask is not None:
            # Create attention mask (True for positions to mask)
            attn_mask = ~mask.bool()  # Invert: True where to mask
            # Ensure at least one position is not masked per sample
            all_masked = attn_mask.all(dim=1)
            if all_masked.any():
                # Unmask first position for samples where all are masked
                attn_mask[all_masked, 0] = False
        else:
            attn_mask = None
        
        attn_out, _ = self.attention(x, x, x, key_padding_mask=attn_mask)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
        x = self.attention_norm(x + attn_out)  # Residual + LayerNorm
        
        # Attention-weighted pooling
        attn_weights = self.pool_attention(x)  # [B, L, 1]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask.bool().unsqueeze(-1), float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Handle NaN from softmax (when all values are -inf)
        attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / attn_weights.size(1))
        
        pooled = torch.sum(x * attn_weights, dim=1)  # [B, H]
        
        # Classification
        logits = self.classifier(pooled)  # [B, 1]
        
        return logits
    
    def predict(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get predictions (probabilities).
        
        Args:
            x: Input features [batch_size, seq_len, input_dim]
            mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            Probabilities [batch_size]
        """
        logits = self.forward(x, mask)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the SeqXGPT model."""
    print("Testing SeqXGPT Model...")
    
    # Create model
    model = SeqXGPTModel(
        input_dim=3,
        hidden_dim=128,
        num_cnn_layers=3,
        kernel_size=3,
        num_attention_heads=4,
        dropout=0.3,
        max_seq_length=256
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 256
    input_dim = 3
    
    x = torch.randn(batch_size, seq_len, input_dim)
    mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    logits = model(x, mask)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    
    # Predict
    probs = model.predict(x, mask)
    print(f"Predictions shape: {probs.shape}")
    print(f"Sample predictions: {probs[:3]}")
    
    # Test with variable length (using mask)
    mask[0, 100:] = 0  # First sample has only 100 tokens
    mask[1, 150:] = 0  # Second sample has only 150 tokens
    
    logits = model(x, mask)
    probs = model.predict(x, mask)
    print(f"\nWith variable lengths:")
    print(f"Predictions: {probs[:3]}")


if __name__ == "__main__":
    test_model()
