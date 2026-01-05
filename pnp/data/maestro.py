"""
MAESTRO Dataset handling for Pink Noise Prior.

The MAESTRO dataset contains high-quality classical piano MIDI recordings,
ideal for training symbolic music generation models.

Dataset: https://magenta.tensorflow.org/datasets/maestro
"""

import os
import json
import tarfile
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from urllib.request import urlretrieve
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from pnp.data.tokenizer import MidiTokenizer
from pnp.data.midi_utils import extract_pitches_from_midi


# MAESTRO dataset URLs
MAESTRO_URLS = {
    'v3.0.0': 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip',
    'v2.0.0': 'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
}

DEFAULT_DATA_DIR = Path.home() / '.cache' / 'pnp' / 'data'


class DownloadProgressBar:
    """Progress bar for downloads."""

    def __init__(self, desc: str = "Downloading"):
        self.pbar = None
        self.desc = desc

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.pbar is None:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=self.desc)
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_maestro(
    data_dir: Optional[Path] = None,
    version: str = 'v3.0.0',
    force: bool = False,
) -> Path:
    """
    Download the MAESTRO dataset.

    Args:
        data_dir: Directory to store the dataset
        version: MAESTRO version to download
        force: Force re-download even if exists

    Returns:
        Path to the extracted dataset directory
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    maestro_dir = data_dir / f'maestro-{version}'
    if maestro_dir.exists() and not force:
        print(f"MAESTRO dataset already exists at {maestro_dir}")
        return maestro_dir

    # Download
    if version not in MAESTRO_URLS:
        raise ValueError(f"Unknown MAESTRO version: {version}. Available: {list(MAESTRO_URLS.keys())}")

    url = MAESTRO_URLS[version]
    zip_path = data_dir / f'maestro-{version}-midi.zip'

    print(f"Downloading MAESTRO {version} from {url}")
    urlretrieve(url, zip_path, DownloadProgressBar(f"Downloading MAESTRO {version}"))

    # Extract
    print(f"Extracting to {data_dir}")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(data_dir)

    # Clean up zip file
    zip_path.unlink()

    print(f"MAESTRO dataset ready at {maestro_dir}")
    return maestro_dir


def load_maestro_metadata(maestro_dir: Path) -> Dict:
    """Load MAESTRO metadata JSON."""
    json_files = list(maestro_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"No metadata JSON found in {maestro_dir}")

    with open(json_files[0], 'r') as f:
        return json.load(f)


def get_maestro_splits(maestro_dir: Path) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Get train/val/test splits from MAESTRO metadata.

    Returns:
        train_paths, val_paths, test_paths
    """
    metadata = load_maestro_metadata(maestro_dir)

    train_paths = []
    val_paths = []
    test_paths = []

    for entry in metadata:
        midi_path = maestro_dir / entry['midi_filename']
        if not midi_path.exists():
            continue

        split = entry['split']
        if split == 'train':
            train_paths.append(midi_path)
        elif split == 'validation':
            val_paths.append(midi_path)
        elif split == 'test':
            test_paths.append(midi_path)

    return train_paths, val_paths, test_paths


class MAESTRODataset(Dataset):
    """
    PyTorch Dataset for the MAESTRO piano dataset.

    Provides efficient loading and caching of tokenized MIDI sequences.

    Args:
        data_dir: Path to MAESTRO dataset directory
        split: 'train', 'validation', or 'test'
        tokenizer: MidiTokenizer instance
        seq_length: Fixed sequence length for training
        stride: Stride for sliding window
        max_files: Maximum number of files to load (for debugging)
        cache_dir: Directory to cache processed data
        use_cache: Whether to use cached data
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = 'train',
        tokenizer: Optional[MidiTokenizer] = None,
        seq_length: int = 512,
        stride: Optional[int] = None,
        max_files: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer or MidiTokenizer()
        self.seq_length = seq_length
        self.stride = stride if stride is not None else seq_length // 2

        # Setup cache
        if cache_dir is None:
            cache_dir = self.data_dir / '.cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = self.cache_dir / f'{split}_seq{seq_length}_stride{self.stride}.pt'

        # Load or process data
        if use_cache and cache_file.exists():
            print(f"Loading cached data from {cache_file}")
            self.chunks = torch.load(cache_file)
        else:
            print(f"Processing MAESTRO {split} split...")
            self.chunks = self._process_midi_files(max_files)
            if use_cache:
                print(f"Caching to {cache_file}")
                torch.save(self.chunks, cache_file)

        print(f"Loaded {len(self.chunks)} sequences from {split} split")

    def _process_midi_files(self, max_files: Optional[int] = None) -> List[torch.Tensor]:
        """Process MIDI files into tokenized chunks."""
        # Get file paths for this split
        train_paths, val_paths, test_paths = get_maestro_splits(self.data_dir)

        if self.split == 'train':
            paths = train_paths
        elif self.split in ('validation', 'val'):
            paths = val_paths
        elif self.split == 'test':
            paths = test_paths
        else:
            raise ValueError(f"Unknown split: {self.split}")

        if max_files is not None:
            paths = paths[:max_files]

        chunks = []
        for path in tqdm(paths, desc=f"Processing {self.split}"):
            try:
                # Extract pitches
                pitches = extract_pitches_from_midi(path)
                if len(pitches) < 32:
                    continue

                # Tokenize
                encoded = self.tokenizer.encode(
                    pitches,
                    add_special_tokens=True,
                    return_tensors=None,
                )
                tokens = encoded['input_ids']

                # Create chunks with sliding window
                for start in range(0, len(tokens) - self.seq_length, self.stride):
                    chunk = torch.tensor(tokens[start:start + self.seq_length + 1])
                    chunks.append(chunk)

            except Exception as e:
                print(f"Warning: Error processing {path}: {e}")

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.

        Returns:
            input_ids: Shape (seq_length,)
            target_ids: Shape (seq_length,) - shifted by 1
        """
        chunk = self.chunks[idx]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids

    def get_pitch_sequence(self, idx: int) -> torch.Tensor:
        """Get the raw pitch sequence (decoded) for analysis."""
        chunk = self.chunks[idx]
        return torch.tensor(self.tokenizer.decode(chunk.tolist()))


def create_maestro_dataloaders(
    data_dir: Path,
    tokenizer: Optional[MidiTokenizer] = None,
    seq_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    max_files: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train/val/test dataloaders for MAESTRO dataset.

    Args:
        data_dir: Path to MAESTRO dataset
        tokenizer: MidiTokenizer (created if None)
        seq_length: Sequence length for training
        batch_size: Batch size
        num_workers: DataLoader workers
        max_files: Max files per split (for debugging)

    Returns:
        train_loader, val_loader, test_loader
    """
    if tokenizer is None:
        tokenizer = MidiTokenizer()

    train_dataset = MAESTRODataset(
        data_dir, 'train', tokenizer, seq_length, max_files=max_files
    )
    val_dataset = MAESTRODataset(
        data_dir, 'validation', tokenizer, seq_length, max_files=max_files
    )
    test_dataset = MAESTRODataset(
        data_dir, 'test', tokenizer, seq_length, max_files=max_files
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
