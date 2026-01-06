"""
Utilities for processing MIDI files.

Provides functions to extract pitch sequences from MIDI files
using pretty_midi and music21.
"""

from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


def extract_pitches_from_midi(
    midi_path: str | Path,
    use_pretty_midi: bool = True,
    quantize_time: bool = True,
    time_resolution: float = 0.05,  # 50ms resolution
    merge_overlapping: bool = True,
) -> List[int]:
    """
    Extract a sequence of MIDI pitches from a MIDI file.

    For the initial prototype, we focus purely on pitch sequences,
    creating a simple time-ordered list of note-on events.

    Args:
        midi_path: Path to the MIDI file
        use_pretty_midi: Use pretty_midi (True) or music21 (False)
        quantize_time: Quantize note times to grid
        time_resolution: Time grid resolution in seconds
        merge_overlapping: Merge simultaneous notes by taking highest

    Returns:
        List of MIDI pitch values (0-127) in temporal order
    """
    if use_pretty_midi:
        return _extract_with_pretty_midi(
            midi_path, quantize_time, time_resolution, merge_overlapping
        )
    else:
        return _extract_with_music21(midi_path)


def _extract_with_pretty_midi(
    midi_path: str | Path,
    quantize_time: bool,
    time_resolution: float,
    merge_overlapping: bool,
) -> List[int]:
    """Extract pitches using pretty_midi library."""
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi is required. Install with: pip install pretty_midi")

    # Load MIDI file
    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # Collect all notes from all instruments
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue  # Skip drums
        for note in instrument.notes:
            notes.append((note.start, note.pitch, note.velocity))

    if not notes:
        return []

    # Sort by start time
    notes.sort(key=lambda x: x[0])

    if quantize_time:
        # Quantize to time grid
        quantized = {}
        for start, pitch, velocity in notes:
            time_idx = int(start / time_resolution)
            if time_idx not in quantized:
                quantized[time_idx] = []
            quantized[time_idx].append((pitch, velocity))

        # Extract pitches in order
        pitches = []
        for time_idx in sorted(quantized.keys()):
            time_notes = quantized[time_idx]
            if merge_overlapping:
                # Take highest pitch (melody typically on top)
                pitch = max(n[0] for n in time_notes)
                pitches.append(pitch)
            else:
                # Include all notes
                for pitch, _ in sorted(time_notes, key=lambda x: -x[0]):
                    pitches.append(pitch)
    else:
        # Just return pitches in temporal order
        if merge_overlapping:
            # Group by approximate time
            pitches = []
            last_time = -1
            current_pitches = []
            for start, pitch, _ in notes:
                if start - last_time > time_resolution:
                    if current_pitches:
                        pitches.append(max(current_pitches))
                    current_pitches = [pitch]
                    last_time = start
                else:
                    current_pitches.append(pitch)
            if current_pitches:
                pitches.append(max(current_pitches))
        else:
            pitches = [note[1] for note in notes]

    return pitches


def _extract_with_music21(midi_path: str | Path) -> List[int]:
    """Extract pitches using music21 library."""
    try:
        from music21 import converter, note, chord
    except ImportError:
        raise ImportError("music21 is required. Install with: pip install music21")

    # Parse MIDI file
    score = converter.parse(str(midi_path))

    # Extract notes and chords
    pitches = []
    for element in score.flatten().notes:
        if isinstance(element, note.Note):
            pitches.append(element.pitch.midi)
        elif isinstance(element, chord.Chord):
            # Take highest pitch from chord
            pitches.append(max(p.midi for p in element.pitches))

    return pitches


def midi_to_pitch_sequence(
    midi_path: str | Path,
    min_pitch: int = 21,
    max_pitch: int = 108,
    normalize: bool = False,
) -> np.ndarray:
    """
    Convert MIDI file to numpy array of pitches.

    Args:
        midi_path: Path to MIDI file
        min_pitch: Minimum pitch (notes below clipped)
        max_pitch: Maximum pitch (notes above clipped)
        normalize: Normalize to [0, 1] range

    Returns:
        Numpy array of pitch values
    """
    pitches = extract_pitches_from_midi(midi_path)

    # Convert to numpy
    pitches = np.array(pitches, dtype=np.float32)

    # Clip to range
    pitches = np.clip(pitches, min_pitch, max_pitch)

    if normalize:
        pitches = (pitches - min_pitch) / (max_pitch - min_pitch)

    return pitches


def create_synthetic_sequence(
    length: int = 1000,
    noise_type: str = 'pink',
    min_pitch: int = 48,  # C3
    max_pitch: int = 84,  # C6
    seed: Optional[int] = None,
) -> List[int]:
    """
    Create a synthetic pitch sequence with specified spectral characteristics.

    Useful for testing and validating the spectral slope loss.

    Args:
        length: Sequence length
        noise_type: 'white', 'pink', or 'brownian'
        min_pitch: Minimum MIDI pitch
        max_pitch: Maximum MIDI pitch
        seed: Random seed

    Returns:
        List of MIDI pitch values
    """
    if seed is not None:
        np.random.seed(seed)

    if noise_type == 'white':
        # White noise - uniform random
        values = np.random.randn(length)
    elif noise_type == 'pink':
        # Pink noise - 1/f spectrum
        values = _generate_pink_noise_numpy(length)
    elif noise_type == 'brownian':
        # Brownian noise - cumulative sum of white noise
        values = np.cumsum(np.random.randn(length))
        values = (values - values.mean()) / (values.std() + 1e-8)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # Scale to pitch range
    values = (values - values.min()) / (values.max() - values.min() + 1e-8)
    pitches = (values * (max_pitch - min_pitch) + min_pitch).astype(int)
    pitches = np.clip(pitches, min_pitch, max_pitch)

    return pitches.tolist()


def _generate_pink_noise_numpy(length: int) -> np.ndarray:
    """Generate pink noise using spectral method."""
    # Generate white noise
    white = np.random.randn(length)

    # FFT
    fft = np.fft.rfft(white)

    # Create 1/f filter
    freqs = np.fft.rfftfreq(length)
    freqs[0] = 1  # Avoid division by zero
    pink_filter = 1.0 / np.sqrt(freqs)
    pink_filter[0] = 0  # Zero DC

    # Apply filter
    fft_pink = fft * pink_filter

    # Inverse FFT
    pink = np.fft.irfft(fft_pink, n=length)

    # Normalize
    pink = (pink - pink.mean()) / (pink.std() + 1e-8)

    return pink


def analyze_midi_file(midi_path: str | Path) -> dict:
    """
    Analyze a MIDI file and return statistics.

    Args:
        midi_path: Path to MIDI file

    Returns:
        Dictionary with statistics about the MIDI file
    """
    try:
        import pretty_midi
    except ImportError:
        raise ImportError("pretty_midi is required")

    midi = pretty_midi.PrettyMIDI(str(midi_path))

    # Collect all notes
    all_notes = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)

    if not all_notes:
        return {'error': 'No notes found'}

    pitches = [n.pitch for n in all_notes]
    durations = [n.end - n.start for n in all_notes]
    velocities = [n.velocity for n in all_notes]

    return {
        'n_notes': len(all_notes),
        'duration_seconds': midi.get_end_time(),
        'n_instruments': len([i for i in midi.instruments if not i.is_drum]),
        'pitch_min': min(pitches),
        'pitch_max': max(pitches),
        'pitch_mean': np.mean(pitches),
        'pitch_std': np.std(pitches),
        'duration_mean': np.mean(durations),
        'velocity_mean': np.mean(velocities),
    }
