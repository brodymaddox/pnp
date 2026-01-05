"""
Tests for the MusicTransformer model.
"""

import pytest
import torch

from pnp.models.transformer import (
    MusicTransformer,
    MusicTransformerConfig,
    create_model,
)


class TestMusicTransformerConfig:
    """Tests for model configuration."""

    def test_default_config(self):
        """Default config should be valid."""
        config = MusicTransformerConfig()

        assert config.vocab_size > 0
        assert config.embed_dim % config.n_heads == 0

    def test_invalid_config(self):
        """Invalid config should raise error."""
        with pytest.raises(AssertionError):
            # embed_dim not divisible by n_heads
            MusicTransformerConfig(embed_dim=100, n_heads=8)


class TestMusicTransformer:
    """Tests for MusicTransformer model."""

    @pytest.fixture
    def config(self):
        return MusicTransformerConfig(
            vocab_size=92,
            max_seq_length=128,
            embed_dim=64,
            n_layers=2,
            n_heads=4,
            ff_dim=128,
        )

    @pytest.fixture
    def model(self, config):
        return MusicTransformer(config)

    def test_forward_shape(self, model, config):
        """Forward pass should return correct shape."""
        batch_size = 4
        seq_len = 64

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits = model(input_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_with_hidden(self, model, config):
        """Forward with return_hidden should return both outputs."""
        batch_size = 2
        seq_len = 32

        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits, hidden = model(input_ids, return_hidden=True)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)
        assert hidden.shape == (batch_size, seq_len, config.embed_dim)

    def test_causal_masking(self, model, config):
        """Model should use causal (autoregressive) masking."""
        # Create sequence where later tokens have different values
        seq_len = 32
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        logits1 = model(input_ids)

        # Change later tokens
        input_ids_modified = input_ids.clone()
        input_ids_modified[0, seq_len//2:] = 0

        logits2 = model(input_ids_modified)

        # Early predictions should be identical (causal masking)
        # They only see previous tokens
        assert torch.allclose(logits1[0, :seq_len//2], logits2[0, :seq_len//2], atol=1e-5)

    def test_generate(self, model, config):
        """Generation should produce valid output."""
        start = torch.tensor([[1]])  # BOS token
        generated = model.generate(start, max_new_tokens=50, temperature=1.0)

        assert generated.shape[0] == 1
        assert generated.shape[1] > 1  # Should have generated something

    def test_generate_with_top_k(self, model, config):
        """Generation with top-k sampling."""
        start = torch.tensor([[1]])
        generated = model.generate(start, max_new_tokens=20, top_k=10)

        assert generated.shape[1] > 1

    def test_generate_deterministic(self, model, config):
        """Generation with same seed should be deterministic."""
        torch.manual_seed(42)
        start = torch.tensor([[1]])
        gen1 = model.generate(start, max_new_tokens=30, temperature=0.5)

        torch.manual_seed(42)
        gen2 = model.generate(start, max_new_tokens=30, temperature=0.5)

        assert torch.equal(gen1, gen2)

    def test_num_params(self, model):
        """Parameter counting should work."""
        n_params = model.get_num_params()
        n_params_all = model.get_num_params(non_embedding=False)

        assert n_params > 0
        assert n_params_all > n_params  # Embedding params should be excluded

    def test_save_load(self, model, config, tmp_path):
        """Model save/load should preserve weights."""
        # Save
        save_path = tmp_path / "model.pt"
        model.save_pretrained(str(save_path))

        # Load
        loaded_model = MusicTransformer.from_pretrained(str(save_path))

        # Compare weights
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), loaded_model.named_parameters()
        ):
            assert name1 == name2
            assert torch.allclose(param1, param2)


class TestCreateModel:
    """Tests for model factory function."""

    def test_create_tiny(self):
        """Create tiny model."""
        model = create_model(vocab_size=92, size='tiny')
        assert model.config.embed_dim == 128
        assert model.config.n_layers == 4

    def test_create_small(self):
        """Create small model."""
        model = create_model(vocab_size=92, size='small')
        assert model.config.embed_dim == 256
        assert model.config.n_layers == 6

    def test_create_medium(self):
        """Create medium model."""
        model = create_model(vocab_size=92, size='medium')
        assert model.config.embed_dim == 512
        assert model.config.n_layers == 8

    def test_create_large(self):
        """Create large model."""
        model = create_model(vocab_size=92, size='large')
        assert model.config.embed_dim == 768
        assert model.config.n_layers == 12

    def test_invalid_size(self):
        """Invalid size should raise error."""
        with pytest.raises(ValueError):
            create_model(size='huge')

    def test_override_params(self):
        """Should be able to override default params."""
        model = create_model(vocab_size=92, size='small', dropout=0.5)
        assert model.config.dropout == 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
