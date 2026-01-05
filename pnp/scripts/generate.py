#!/usr/bin/env python
"""
Generation script for Pink Noise Prior music generation.

Generate MIDI files from a trained model.

Usage:
    python -m pnp.scripts.generate --checkpoint checkpoints/model.pt --output output.mid
    python -m pnp.scripts.generate --checkpoint model.pt --n-samples 10 --output-dir generated/
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from pnp.data.tokenizer import MidiTokenizer
from pnp.models.transformer import MusicTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate music from trained model")

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output MIDI file path (for single generation)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='generated',
        help='Output directory (for multiple generations)'
    )
    parser.add_argument(
        '--n-samples', type=int, default=1,
        help='Number of samples to generate'
    )
    parser.add_argument(
        '--max-length', type=int, default=512,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature (lower = more deterministic)'
    )
    parser.add_argument(
        '--top-k', type=int, default=50,
        help='Top-k sampling (0 = disabled)'
    )
    parser.add_argument(
        '--top-p', type=float, default=None,
        help='Nucleus sampling threshold'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--prompt', type=str, default=None,
        help='Starting pitches (comma-separated MIDI notes)'
    )
    parser.add_argument(
        '--bpm', type=float, default=120.0,
        help='Tempo in BPM for MIDI output'
    )
    parser.add_argument(
        '--note-duration', type=float, default=0.25,
        help='Note duration in beats'
    )

    return parser.parse_args()


def pitches_to_midi(
    pitches: List[int],
    output_path: str,
    bpm: float = 120.0,
    note_duration: float = 0.25,
    velocity: int = 80,
):
    """
    Convert a list of MIDI pitches to a MIDI file.

    Args:
        pitches: List of MIDI pitch values
        output_path: Output file path
        bpm: Tempo in beats per minute
        note_duration: Duration of each note in beats
        velocity: Note velocity (0-127)
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi is required. Install with: pip install pretty_midi")

    # Create MIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)

    # Create instrument (piano)
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    # Add notes
    current_time = 0.0
    seconds_per_beat = 60.0 / bpm

    for pitch in pitches:
        if 0 <= pitch <= 127:  # Valid MIDI range
            note = pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=current_time,
                end=current_time + note_duration * seconds_per_beat,
            )
            instrument.notes.append(note)
        current_time += note_duration * seconds_per_beat

    midi.instruments.append(instrument)
    midi.write(output_path)


def generate_and_save(
    model: MusicTransformer,
    tokenizer: MidiTokenizer,
    output_path: str,
    max_length: int = 512,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    prompt: Optional[List[int]] = None,
    bpm: float = 120.0,
    note_duration: float = 0.25,
    device: torch.device = None,
) -> List[int]:
    """
    Generate a sequence and save as MIDI.

    Returns the generated pitch sequence.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    with torch.no_grad():
        # Create starting sequence
        if prompt is not None:
            encoded = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
            start = encoded['input_ids'].to(device)
            # Remove EOS if present
            if start[0, -1] == tokenizer.eos_token_id:
                start = start[:, :-1]
        else:
            start = torch.tensor([[tokenizer.bos_token_id]], device=device)

        # Generate
        output = model.generate(
            start,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode to pitches
        pitches = tokenizer.decode(output[0].cpu().tolist())

    # Save as MIDI
    if pitches:
        pitches_to_midi(pitches, output_path, bpm=bpm, note_duration=note_duration)
        print(f"Saved {len(pitches)} notes to {output_path}")

    return pitches


def main():
    args = parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = MusicTransformer.from_pretrained(args.checkpoint)
    model.to(device)
    model.eval()
    print(f"Model loaded ({model.get_num_params():,} parameters)")

    # Create tokenizer
    tokenizer = MidiTokenizer()

    # Parse prompt if provided
    prompt = None
    if args.prompt:
        prompt = [int(p.strip()) for p in args.prompt.split(',')]
        print(f"Using prompt: {prompt}")

    # Generate
    if args.n_samples == 1 and args.output:
        # Single generation
        pitches = generate_and_save(
            model,
            tokenizer,
            args.output,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p,
            prompt=prompt,
            bpm=args.bpm,
            note_duration=args.note_duration,
            device=device,
        )
        print(f"Generated sequence with {len(pitches)} notes")
    else:
        # Multiple generations
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.n_samples):
            output_path = output_dir / f"generated_{i:04d}.mid"
            pitches = generate_and_save(
                model,
                tokenizer,
                str(output_path),
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                top_p=args.top_p,
                prompt=prompt,
                bpm=args.bpm,
                note_duration=args.note_duration,
                device=device,
            )

        print(f"\nGenerated {args.n_samples} samples in {output_dir}")


if __name__ == '__main__':
    main()
