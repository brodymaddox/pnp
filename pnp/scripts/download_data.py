#!/usr/bin/env python
"""
Download and prepare the MAESTRO dataset for training.

Usage:
    python -m pnp.scripts.download_data
    python -m pnp.scripts.download_data --data-dir /path/to/data --version v3.0.0
"""

import argparse
from pathlib import Path

from pnp.data.maestro import download_maestro, get_maestro_splits, load_maestro_metadata
from pnp.data.tokenizer import MidiTokenizer
from pnp.data.midi_utils import extract_pitches_from_midi, analyze_midi_file


def parse_args():
    parser = argparse.ArgumentParser(description="Download MAESTRO dataset")

    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Directory to store dataset (default: ~/.cache/pnp/data)'
    )
    parser.add_argument(
        '--version', type=str, default='v3.0.0',
        choices=['v2.0.0', 'v3.0.0'],
        help='MAESTRO version to download'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force re-download even if exists'
    )
    parser.add_argument(
        '--analyze', action='store_true',
        help='Analyze dataset after download'
    )
    parser.add_argument(
        '--preprocess', action='store_true',
        help='Preprocess and cache tokenized sequences'
    )
    parser.add_argument(
        '--seq-length', type=int, default=512,
        help='Sequence length for preprocessing'
    )

    return parser.parse_args()


def analyze_dataset(data_dir: Path):
    """Analyze the downloaded dataset."""
    import numpy as np

    print("\nAnalyzing dataset...")

    train_paths, val_paths, test_paths = get_maestro_splits(data_dir)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_paths)} files")
    print(f"  Validation: {len(val_paths)} files")
    print(f"  Test: {len(test_paths)} files")

    # Sample analysis
    print("\nSample file analysis (first 10 train files):")

    total_notes = 0
    total_duration = 0
    all_pitches = []

    for path in train_paths[:10]:
        try:
            stats = analyze_midi_file(path)
            pitches = extract_pitches_from_midi(path)

            print(f"\n  {path.name}:")
            print(f"    Notes: {stats['n_notes']}")
            print(f"    Duration: {stats['duration_seconds']:.1f}s")
            print(f"    Pitch range: {stats['pitch_min']}-{stats['pitch_max']}")
            print(f"    Extracted sequence length: {len(pitches)}")

            total_notes += stats['n_notes']
            total_duration += stats['duration_seconds']
            all_pitches.extend(pitches)

        except Exception as e:
            print(f"    Error: {e}")

    if all_pitches:
        all_pitches = np.array(all_pitches)
        print(f"\nAggregate statistics (sample):")
        print(f"  Total notes: {total_notes:,}")
        print(f"  Total duration: {total_duration:.1f}s")
        print(f"  Pitch mean: {all_pitches.mean():.1f}")
        print(f"  Pitch std: {all_pitches.std():.1f}")
        print(f"  Pitch range: {all_pitches.min()}-{all_pitches.max()}")


def preprocess_dataset(data_dir: Path, seq_length: int):
    """Preprocess and cache tokenized sequences."""
    from pnp.data.maestro import MAESTRODataset

    print(f"\nPreprocessing dataset (seq_length={seq_length})...")
    tokenizer = MidiTokenizer()

    for split in ['train', 'validation', 'test']:
        print(f"\n  Processing {split} split...")
        dataset = MAESTRODataset(
            data_dir,
            split=split,
            tokenizer=tokenizer,
            seq_length=seq_length,
            use_cache=True,
        )
        print(f"    {len(dataset)} sequences cached")


def main():
    args = parse_args()

    print("=" * 60)
    print("MAESTRO Dataset Download and Preparation")
    print("=" * 60)

    # Download
    data_dir = download_maestro(
        data_dir=args.data_dir,
        version=args.version,
        force=args.force,
    )

    # Load metadata
    try:
        metadata = load_maestro_metadata(data_dir)
        print(f"\nMetadata loaded: {len(metadata)} performances")

        # Show composers
        composers = set(entry.get('canonical_composer', 'Unknown') for entry in metadata)
        print(f"Composers: {len(composers)}")
        for composer in sorted(composers)[:10]:
            count = sum(1 for e in metadata if e.get('canonical_composer') == composer)
            print(f"  {composer}: {count} pieces")

    except Exception as e:
        print(f"Could not load metadata: {e}")

    # Analyze if requested
    if args.analyze:
        analyze_dataset(data_dir)

    # Preprocess if requested
    if args.preprocess:
        preprocess_dataset(data_dir, args.seq_length)

    print("\n" + "=" * 60)
    print(f"Dataset ready at: {data_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
