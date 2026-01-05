"""
Lightweight Causal Transformer for Symbolic Music Generation.

A mini-GPT style architecture designed for efficient training on
consumer hardware while maintaining sufficient capacity to learn
musical structure.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class MusicTransformerConfig:
    """
    Configuration for MusicTransformer.

    Default configuration is a lightweight model suitable for
    training on a single consumer GPU.
    """
    # Vocabulary and embedding
    vocab_size: int = 92  # 88 piano keys (21-108) + 4 special tokens
    max_seq_length: int = 512
    embed_dim: int = 256

    # Transformer architecture
    n_layers: int = 6
    n_heads: int = 8
    ff_dim: int = 1024  # Feed-forward hidden dimension
    dropout: float = 0.1

    # Positional encoding
    use_learned_pos_embed: bool = True
    use_relative_attention: bool = False

    # Regularization
    embed_dropout: float = 0.1
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    # Initialization
    init_std: float = 0.02

    def __post_init__(self):
        assert self.embed_dim % self.n_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by n_heads ({self.n_heads})"


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (from original Transformer)."""

    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Ensures that positions can only attend to earlier positions (autoregressive).
    """

    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.n_heads

        # QKV projection
        self.qkv = nn.Linear(config.embed_dim, 3 * config.embed_dim)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(
                torch.ones(config.max_seq_length, config.max_seq_length),
                diagonal=1
            ).bool()
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Shape (batch, seq_len, embed_dim)
            attention_mask: Optional mask for padding (batch, seq_len)

        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn = attn.masked_fill(causal_mask, float('-inf'))

        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> expand for heads
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            attn = attn.masked_fill(~padding_mask.bool(), float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attention_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, n_heads, seq_len, head_dim)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = self.proj(out)
        out = self.residual_dropout(out)

        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.embed_dim, config.ff_dim)
        self.fc2 = nn.Linear(config.ff_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm architecture."""

    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm with residual
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ff(self.ln2(x))
        return x


class MusicTransformer(nn.Module):
    """
    Lightweight Causal Transformer for Symbolic Music Generation.

    A GPT-style architecture that predicts the next note given a sequence
    of previous notes. Designed for efficient training while maintaining
    capacity for learning musical structure.

    Args:
        config: MusicTransformerConfig with model hyperparameters
    """

    def __init__(self, config: MusicTransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional encoding
        if config.use_learned_pos_embed:
            self.pos_embed = nn.Embedding(config.max_seq_length, config.embed_dim)
        else:
            self.pos_embed = SinusoidalPositionalEncoding(
                config.embed_dim, config.max_seq_length, config.embed_dropout
            )

        self.embed_dropout = nn.Dropout(config.embed_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Output
        self.ln_final = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying (share embedding weights with output)
        self.lm_head.weight = self.token_embed.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)
            attention_mask: Optional padding mask (batch, seq_len)
            return_hidden: Also return final hidden states

        Returns:
            logits: Shape (batch, seq_len, vocab_size)
            hidden: (optional) Shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        x = self.token_embed(input_ids)

        # Position embeddings
        if isinstance(self.pos_embed, nn.Embedding):
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = x + self.pos_embed(positions)
            x = self.embed_dropout(x)
        else:
            x = self.pos_embed(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm
        x = self.ln_final(x)

        # Output logits
        logits = self.lm_head(x)

        if return_hidden:
            return logits, x
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            input_ids: Starting sequence (batch, seq_len)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None = disabled)
            top_p: Nucleus sampling (None = disabled)
            eos_token_id: Stop at this token

        Returns:
            Generated sequence including input (batch, seq_len + generated)
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncate to max length if needed
            if generated.size(1) >= self.config.max_seq_length:
                context = generated[:, -self.config.max_seq_length:]
            else:
                context = generated

            # Forward pass
            logits = self.forward(context)
            next_logits = logits[:, -1, :] / temperature

            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters (optionally excluding embeddings)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embed.weight.numel()
            if isinstance(self.pos_embed, nn.Embedding):
                n_params -= self.pos_embed.weight.numel()
        return n_params

    @classmethod
    def from_pretrained(cls, path: str) -> 'MusicTransformer':
        """Load a pretrained model."""
        checkpoint = torch.load(path, map_location='cpu')
        config = MusicTransformerConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def save_pretrained(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'config': vars(self.config),
            'model_state_dict': self.state_dict(),
        }, path)


def create_model(
    vocab_size: int = 92,
    size: str = 'small',
    **kwargs,
) -> MusicTransformer:
    """
    Create a MusicTransformer with preset size configurations.

    Args:
        vocab_size: Vocabulary size
        size: 'tiny', 'small', 'medium', or 'large'
        **kwargs: Override any config parameter

    Returns:
        MusicTransformer model
    """
    presets = {
        'tiny': {
            'embed_dim': 128,
            'n_layers': 4,
            'n_heads': 4,
            'ff_dim': 512,
        },
        'small': {
            'embed_dim': 256,
            'n_layers': 6,
            'n_heads': 8,
            'ff_dim': 1024,
        },
        'medium': {
            'embed_dim': 512,
            'n_layers': 8,
            'n_heads': 8,
            'ff_dim': 2048,
        },
        'large': {
            'embed_dim': 768,
            'n_layers': 12,
            'n_heads': 12,
            'ff_dim': 3072,
        },
    }

    if size not in presets:
        raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")

    config_dict = {'vocab_size': vocab_size, **presets[size], **kwargs}
    config = MusicTransformerConfig(**config_dict)
    return MusicTransformer(config)
