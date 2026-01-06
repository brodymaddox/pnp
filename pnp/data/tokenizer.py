"""
MIDI Tokenizer for Symbolic Music Generation

Converts MIDI files into sequences of tokens suitable for training
a language model. For the initial prototype, we focus purely on
pitch sequences, ignoring velocity and duration.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import json


class MidiTokenizer:
    """
    Tokenizer for MIDI pitch sequences.

    Converts MIDI note numbers (0-127) to tokens for language model training.
    Special tokens are added for sequence boundaries and padding.

    Token mapping:
        0: PAD - Padding token
        1: BOS - Beginning of sequence
        2: EOS - End of sequence
        3: UNK - Unknown token (not typically used)
        4-131: MIDI notes 0-127

    Args:
        include_special_tokens: Whether to include PAD, BOS, EOS tokens
        min_pitch: Minimum MIDI pitch to include (default: 21, A0)
        max_pitch: Maximum MIDI pitch to include (default: 108, C8)
    """

    # Special token constants
    PAD_TOKEN = 0
    BOS_TOKEN = 1
    EOS_TOKEN = 2
    UNK_TOKEN = 3
    PITCH_OFFSET = 4

    def __init__(
        self,
        include_special_tokens: bool = True,
        min_pitch: int = 21,  # A0 - lowest piano key
        max_pitch: int = 108,  # C8 - highest piano key
    ):
        self.include_special_tokens = include_special_tokens
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

        # Build vocabulary
        self._build_vocab()

    def _build_vocab(self):
        """Build the token vocabulary."""
        self.special_tokens = {
            'PAD': self.PAD_TOKEN,
            'BOS': self.BOS_TOKEN,
            'EOS': self.EOS_TOKEN,
            'UNK': self.UNK_TOKEN,
        }

        # Pitch range
        self.pitch_range = range(self.min_pitch, self.max_pitch + 1)
        self.n_pitches = len(self.pitch_range)

        # Vocab size
        self.vocab_size = self.PITCH_OFFSET + self.n_pitches

        # Mappings
        self.pitch_to_token = {
            p: p - self.min_pitch + self.PITCH_OFFSET
            for p in self.pitch_range
        }
        self.token_to_pitch = {
            v: k for k, v in self.pitch_to_token.items()
        }

    @property
    def pad_token_id(self) -> int:
        return self.PAD_TOKEN

    @property
    def bos_token_id(self) -> int:
        return self.BOS_TOKEN

    @property
    def eos_token_id(self) -> int:
        return self.EOS_TOKEN

    def encode(
        self,
        pitches: List[int],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Encode a sequence of MIDI pitches into tokens.

        Args:
            pitches: List of MIDI pitch values (0-127)
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            return_tensors: 'pt' for PyTorch tensors, None for lists

        Returns:
            Dict with 'input_ids' and optionally 'attention_mask'
        """
        # Convert pitches to tokens
        tokens = []

        if add_special_tokens:
            tokens.append(self.BOS_TOKEN)

        for pitch in pitches:
            if pitch < self.min_pitch or pitch > self.max_pitch:
                # Clip to valid range
                pitch = max(self.min_pitch, min(self.max_pitch, pitch))
            tokens.append(self.pitch_to_token[pitch])

        if add_special_tokens:
            tokens.append(self.EOS_TOKEN)

        # Handle max_length
        if max_length is not None:
            if truncation and len(tokens) > max_length:
                tokens = tokens[:max_length]
                # Ensure EOS at end if we truncated
                if add_special_tokens:
                    tokens[-1] = self.EOS_TOKEN

            if padding and len(tokens) < max_length:
                pad_length = max_length - len(tokens)
                attention_mask = [1] * len(tokens) + [0] * pad_length
                tokens = tokens + [self.PAD_TOKEN] * pad_length
            else:
                attention_mask = [1] * len(tokens)
        else:
            attention_mask = [1] * len(tokens)

        result = {'input_ids': tokens}
        if padding:
            result['attention_mask'] = attention_mask

        # Convert to tensors if requested
        if return_tensors == 'pt':
            result = {k: torch.tensor(v) for k, v in result.items()}

        return result

    def decode(
        self,
        token_ids: List[int] | torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[int]:
        """
        Decode tokens back to MIDI pitches.

        Args:
            token_ids: List or tensor of token IDs
            skip_special_tokens: Skip PAD, BOS, EOS tokens

        Returns:
            List of MIDI pitch values
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        pitches = []
        for token in token_ids:
            if skip_special_tokens and token < self.PITCH_OFFSET:
                continue
            if token in self.token_to_pitch:
                pitches.append(self.token_to_pitch[token])

        return pitches

    def batch_encode(
        self,
        sequences: List[List[int]],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = 'pt',
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple sequences with uniform padding.

        Args:
            sequences: List of pitch sequences
            max_length: Max length (None = use longest sequence)
            padding: Pad to uniform length
            truncation: Truncate long sequences
            return_tensors: Tensor format ('pt' for PyTorch)

        Returns:
            Dict with batched 'input_ids' and 'attention_mask'
        """
        # Determine max length if not specified
        if max_length is None and padding:
            max_length = max(len(seq) for seq in sequences) + 2  # +2 for BOS/EOS

        # Encode each sequence
        encoded = [
            self.encode(
                seq,
                add_special_tokens=True,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None,
            )
            for seq in sequences
        ]

        # Stack into batches
        result = {
            'input_ids': [e['input_ids'] for e in encoded],
        }
        if padding:
            result['attention_mask'] = [e['attention_mask'] for e in encoded]

        if return_tensors == 'pt':
            result = {k: torch.tensor(v) for k, v in result.items()}

        return result

    def token_to_pitch_value(self, token: int) -> float:
        """
        Convert a token to its pitch value (for soft representations).

        Args:
            token: Token ID

        Returns:
            Pitch value (or 0 for special tokens)
        """
        if token in self.token_to_pitch:
            return float(self.token_to_pitch[token])
        return 0.0

    def get_pitch_values_tensor(self, device: torch.device = None) -> torch.Tensor:
        """
        Get tensor of pitch values for each token in vocabulary.

        Useful for computing expected pitch from softmax probabilities.

        Args:
            device: Torch device

        Returns:
            Tensor of shape (vocab_size,) with pitch values
        """
        values = torch.zeros(self.vocab_size)
        for token, pitch in self.token_to_pitch.items():
            values[token] = pitch
        if device is not None:
            values = values.to(device)
        return values

    def save(self, path: str | Path):
        """Save tokenizer configuration to file."""
        config = {
            'include_special_tokens': self.include_special_tokens,
            'min_pitch': self.min_pitch,
            'max_pitch': self.max_pitch,
            'vocab_size': self.vocab_size,
        }
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> 'MidiTokenizer':
        """Load tokenizer from configuration file."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(
            include_special_tokens=config['include_special_tokens'],
            min_pitch=config['min_pitch'],
            max_pitch=config['max_pitch'],
        )

    def __repr__(self) -> str:
        return (
            f"MidiTokenizer(vocab_size={self.vocab_size}, "
            f"pitch_range=[{self.min_pitch}, {self.max_pitch}])"
        )
