"""
PyTorch Dataset for MIDI sequences.

Loads tokenized MIDI data and prepares it for training a language model
with next-token prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import numpy as np
import pickle
from pnp.data.tokenizer import MidiTokenizer


class MidiDataset(Dataset):
    """
    Dataset for tokenized MIDI pitch sequences.

    Each item is a pair of (input_ids, target_ids) where target_ids
    are input_ids shifted by one position (next-token prediction).

    Args:
        sequences: List of pitch sequences (list of MIDI pitch values)
        tokenizer: MidiTokenizer instance
        seq_length: Fixed sequence length for training
        stride: Stride for sliding window (default: seq_length // 2)
    """

    def __init__(
        self,
        sequences: List[List[int]],
        tokenizer: MidiTokenizer,
        seq_length: int = 512,
        stride: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length // 2

        # Process sequences into fixed-length chunks
        self.chunks = self._create_chunks(sequences)

    def _create_chunks(self, sequences: List[List[int]]) -> List[torch.Tensor]:
        """
        Create fixed-length chunks from variable-length sequences.

        Uses sliding window with stride to maximize data utilization.
        """
        chunks = []

        for seq in sequences:
            # Tokenize the sequence
            encoded = self.tokenizer.encode(
                seq,
                add_special_tokens=True,
                max_length=None,
                padding=False,
                return_tensors=None,
            )
            tokens = encoded['input_ids']

            # Need at least seq_length + 1 tokens (for input + target)
            if len(tokens) < self.seq_length + 1:
                # Pad short sequences
                if len(tokens) >= 32:  # Minimum viable length
                    pad_length = self.seq_length + 1 - len(tokens)
                    tokens = tokens + [self.tokenizer.PAD_TOKEN] * pad_length
                    chunks.append(torch.tensor(tokens))
            else:
                # Sliding window for long sequences
                for start in range(0, len(tokens) - self.seq_length, self.stride):
                    chunk = tokens[start:start + self.seq_length + 1]
                    chunks.append(torch.tensor(chunk))

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.

        Returns:
            input_ids: Token IDs for input, shape (seq_length,)
            target_ids: Token IDs for targets (shifted by 1), shape (seq_length,)
        """
        chunk = self.chunks[idx]

        # Input is all but last token
        input_ids = chunk[:-1]
        # Target is all but first token (shifted by 1)
        target_ids = chunk[1:]

        return input_ids, target_ids

    @classmethod
    def from_midi_files(
        cls,
        midi_paths: List[Path],
        tokenizer: MidiTokenizer,
        seq_length: int = 512,
        stride: Optional[int] = None,
    ) -> 'MidiDataset':
        """
        Create dataset from a list of MIDI files.

        Args:
            midi_paths: List of paths to MIDI files
            tokenizer: MidiTokenizer instance
            seq_length: Fixed sequence length
            stride: Sliding window stride
        """
        from pnp.data.midi_utils import extract_pitches_from_midi

        sequences = []
        for path in midi_paths:
            try:
                pitches = extract_pitches_from_midi(path)
                if len(pitches) >= 32:  # Minimum length
                    sequences.append(pitches)
            except Exception as e:
                print(f"Warning: Could not process {path}: {e}")

        return cls(sequences, tokenizer, seq_length, stride)

    def save(self, path: str | Path):
        """Save processed dataset to disk."""
        data = {
            'chunks': self.chunks,
            'seq_length': self.seq_length,
            'stride': self.stride,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path, tokenizer: MidiTokenizer) -> 'MidiDataset':
        """Load processed dataset from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        dataset = cls.__new__(cls)
        dataset.tokenizer = tokenizer
        dataset.chunks = data['chunks']
        dataset.seq_length = data['seq_length']
        dataset.stride = data['stride']
        return dataset


class MidiSequenceDataset(Dataset):
    """
    Simple dataset for pre-tokenized sequences stored as tensors.

    Useful for loading cached/preprocessed data.
    """

    def __init__(
        self,
        token_sequences: torch.Tensor,
        seq_length: int = 512,
    ):
        """
        Args:
            token_sequences: Tensor of shape (n_sequences, max_seq_len)
            seq_length: Length to use for training
        """
        self.sequences = token_sequences
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        # Ensure we have enough length
        if len(seq) < self.seq_length + 1:
            # This shouldn't happen if data is preprocessed correctly
            pad = torch.zeros(self.seq_length + 1 - len(seq), dtype=seq.dtype)
            seq = torch.cat([seq, pad])

        input_ids = seq[:self.seq_length]
        target_ids = seq[1:self.seq_length + 1]
        return input_ids, target_ids


def create_dataloaders(
    dataset: MidiDataset,
    batch_size: int = 32,
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders from a dataset.

    Args:
        dataset: MidiDataset instance
        batch_size: Batch size for training
        train_ratio: Fraction of data for training
        val_ratio: Fraction of data for validation
        test_ratio: Fraction of data for testing
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility

    Returns:
        train_loader, val_loader, test_loader
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Calculate split sizes
    n_total = len(dataset)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Stacks input/target pairs into batched tensors.
    """
    inputs, targets = zip(*batch)
    return {
        'input_ids': torch.stack(inputs),
        'targets': torch.stack(targets),
    }
