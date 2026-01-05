"""
Tests for MIDI tokenizer.
"""

import pytest
import torch

from pnp.data.tokenizer import MidiTokenizer


class TestMidiTokenizer:
    """Tests for MidiTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        return MidiTokenizer()

    def test_vocab_size(self, tokenizer):
        """Vocab size should be correct."""
        # 4 special tokens + (108 - 21 + 1) pitch tokens = 4 + 88 = 92
        assert tokenizer.vocab_size == 92

    def test_special_tokens(self, tokenizer):
        """Special tokens should have correct IDs."""
        assert tokenizer.PAD_TOKEN == 0
        assert tokenizer.BOS_TOKEN == 1
        assert tokenizer.EOS_TOKEN == 2
        assert tokenizer.UNK_TOKEN == 3

    def test_encode_decode(self, tokenizer):
        """Encode/decode should be reversible."""
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale

        encoded = tokenizer.encode(pitches, add_special_tokens=False)
        decoded = tokenizer.decode(encoded['input_ids'])

        assert decoded == pitches

    def test_encode_with_special_tokens(self, tokenizer):
        """Encoding with special tokens."""
        pitches = [60, 62, 64]

        encoded = tokenizer.encode(pitches, add_special_tokens=True)
        tokens = encoded['input_ids']

        assert tokens[0] == tokenizer.BOS_TOKEN
        assert tokens[-1] == tokenizer.EOS_TOKEN

    def test_decode_skip_special(self, tokenizer):
        """Decoding should skip special tokens by default."""
        pitches = [60, 62, 64]

        encoded = tokenizer.encode(pitches, add_special_tokens=True)
        decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)

        assert decoded == pitches

    def test_decode_keep_special(self, tokenizer):
        """Decoding with special tokens kept."""
        pitches = [60, 62, 64]

        encoded = tokenizer.encode(pitches, add_special_tokens=True)
        decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=False)

        # Only the pitch values should be returned (special tokens have no pitch)
        assert decoded == pitches

    def test_pitch_clipping(self, tokenizer):
        """Out-of-range pitches should be clipped."""
        pitches = [10, 60, 120]  # 10 too low, 120 too high

        encoded = tokenizer.encode(pitches, add_special_tokens=False)
        decoded = tokenizer.decode(encoded['input_ids'])

        assert decoded[0] == tokenizer.min_pitch  # Clipped to 21
        assert decoded[1] == 60
        assert decoded[2] == tokenizer.max_pitch  # Clipped to 108

    def test_max_length_truncation(self, tokenizer):
        """Sequences should be truncated to max_length."""
        pitches = list(range(60, 80))  # 20 pitches

        encoded = tokenizer.encode(
            pitches,
            add_special_tokens=True,
            max_length=10,
            truncation=True,
        )

        assert len(encoded['input_ids']) == 10
        # Should end with EOS after truncation
        assert encoded['input_ids'][-1] == tokenizer.EOS_TOKEN

    def test_padding(self, tokenizer):
        """Sequences should be padded to max_length."""
        pitches = [60, 62, 64]

        encoded = tokenizer.encode(
            pitches,
            add_special_tokens=True,
            max_length=10,
            padding=True,
        )

        assert len(encoded['input_ids']) == 10
        assert tokenizer.PAD_TOKEN in encoded['input_ids']
        assert 'attention_mask' in encoded
        assert sum(encoded['attention_mask']) == 5  # BOS + 3 pitches + EOS

    def test_return_tensors(self, tokenizer):
        """Should return PyTorch tensors when requested."""
        pitches = [60, 62, 64]

        encoded = tokenizer.encode(pitches, return_tensors='pt')

        assert isinstance(encoded['input_ids'], torch.Tensor)

    def test_batch_encode(self, tokenizer):
        """Batch encoding should work."""
        sequences = [
            [60, 62, 64],
            [65, 67, 69, 71],
            [48, 50],
        ]

        encoded = tokenizer.batch_encode(sequences, padding=True)

        assert encoded['input_ids'].shape[0] == 3  # 3 sequences
        # All should have same length (padded to longest + BOS/EOS)
        assert encoded['input_ids'].shape[1] == 6  # 4 + 2

    def test_pitch_values_tensor(self, tokenizer):
        """Get pitch values tensor for soft representation."""
        values = tokenizer.get_pitch_values_tensor()

        assert values.shape == (tokenizer.vocab_size,)
        # Special tokens should have value 0
        assert values[tokenizer.PAD_TOKEN] == 0
        assert values[tokenizer.BOS_TOKEN] == 0
        # Pitch tokens should have their pitch value
        assert values[tokenizer.PITCH_OFFSET] == tokenizer.min_pitch

    def test_save_load(self, tokenizer, tmp_path):
        """Save and load should preserve configuration."""
        save_path = tmp_path / "tokenizer.json"

        tokenizer.save(save_path)
        loaded = MidiTokenizer.load(save_path)

        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.min_pitch == tokenizer.min_pitch
        assert loaded.max_pitch == tokenizer.max_pitch


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
