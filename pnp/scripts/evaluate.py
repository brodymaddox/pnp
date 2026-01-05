#!/usr/bin/env python
"""
Evaluation script for Pink Noise Prior music generation.

Computes objective metrics on generated sequences:
- Higuchi Fractal Dimension
- Spectral Slope
- Zipf's Law Coefficient
- Autocorrelation Analysis

Usage:
    python -m pnp.scripts.evaluate --checkpoint checkpoints/model.pt --n-samples 100
"""

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from tqdm import tqdm

from pnp.data.tokenizer import MidiTokenizer
from pnp.models.transformer import MusicTransformer
from pnp.metrics.fractal import (
    HiguchiFractalDimension,
    compute_spectral_slope,
    batch_analyze_sequences,
)
from pnp.metrics.zipf import ZipfAnalyzer, batch_analyze_zipf
from pnp.metrics.autocorrelation import AutocorrelationAnalyzer, batch_analyze_autocorrelation
from pnp.utils.visualization import plot_spectral_analysis


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate music generation model")

    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--n-samples', type=int, default=100,
        help='Number of sequences to generate'
    )
    parser.add_argument(
        '--max-length', type=int, default=512,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--temperature', type=float, default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k', type=int, default=50,
        help='Top-k sampling'
    )
    parser.add_argument(
        '--output-dir', type=str, default='evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    return parser.parse_args()


def generate_samples(
    model: MusicTransformer,
    tokenizer: MidiTokenizer,
    n_samples: int,
    max_length: int,
    temperature: float = 1.0,
    top_k: int = 50,
    device: torch.device = None,
) -> List[np.ndarray]:
    """Generate sample sequences from the model."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    samples = []

    with torch.no_grad():
        for _ in tqdm(range(n_samples), desc="Generating samples"):
            # Start with BOS token
            start = torch.tensor([[tokenizer.bos_token_id]], device=device)

            # Generate
            output = model.generate(
                start,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode to pitches
            pitches = tokenizer.decode(output[0].cpu().tolist())
            if len(pitches) > 10:  # Minimum length for analysis
                samples.append(np.array(pitches))

    return samples


def evaluate_samples(samples: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute all evaluation metrics on generated samples.

    Returns dictionary with metric values and statistics.
    """
    results = {}

    print("\nComputing metrics...")

    # Fractal analysis
    print("  Fractal Dimension & Spectral Slope...")
    fractal_results = batch_analyze_sequences(samples)
    results.update({
        'fractal_dimension_mean': fractal_results['fractal_dimension_mean'],
        'fractal_dimension_std': fractal_results['fractal_dimension_std'],
        'spectral_slope_mean': fractal_results['spectral_slope_mean'],
        'spectral_slope_std': fractal_results['spectral_slope_std'],
        'fractal_n_valid': fractal_results['n_valid'],
    })

    # Zipf analysis
    print("  Zipf's Law...")
    zipf_results = batch_analyze_zipf(samples)
    results.update({
        'zipf_coefficient_mean': zipf_results['zipf_coefficient_mean'],
        'zipf_coefficient_std': zipf_results['zipf_coefficient_std'],
        'zipf_n_valid': zipf_results['n_valid'],
    })

    # Autocorrelation analysis
    print("  Autocorrelation...")
    acf_results = batch_analyze_autocorrelation(samples)
    results.update({
        'acf_decay_rate_mean': acf_results['decay_rate_mean'],
        'acf_long_range_correlation_mean': acf_results['long_range_correlation_mean'],
        'acf_characteristic_lag_mean': acf_results['characteristic_lag_mean'],
        'acf_n_valid': acf_results['n_valid'],
    })

    # Summary statistics
    lengths = [len(s) for s in samples]
    results['mean_length'] = np.mean(lengths)
    results['std_length'] = np.std(lengths)
    results['n_samples'] = len(samples)

    return results


def print_results(results: Dict[str, float], reference_values: Dict = None):
    """Print evaluation results with optional reference comparison."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Reference values for high-quality music
    if reference_values is None:
        reference_values = {
            'fractal_dimension': (1.3, 1.5),
            'spectral_slope': (-1.2, -0.8),
            'zipf_coefficient': (0.8, 1.2),
            'acf_long_range_correlation': (0.1, 1.0),
        }

    print(f"\n{'Metric':<35} {'Value':>15} {'Reference':>15}")
    print("-" * 65)

    # Fractal metrics
    fd = results.get('fractal_dimension_mean', float('nan'))
    fd_ref = reference_values.get('fractal_dimension', (0, 0))
    in_range = fd_ref[0] <= fd <= fd_ref[1] if not np.isnan(fd) else False
    status = "✓" if in_range else "✗"
    print(f"Fractal Dimension               {fd:>15.3f} {f'{fd_ref[0]:.1f}-{fd_ref[1]:.1f}':>15} {status}")

    slope = results.get('spectral_slope_mean', float('nan'))
    slope_ref = reference_values.get('spectral_slope', (0, 0))
    in_range = slope_ref[0] <= slope <= slope_ref[1] if not np.isnan(slope) else False
    status = "✓" if in_range else "✗"
    print(f"Spectral Slope                  {slope:>15.3f} {f'{slope_ref[0]:.1f}-{slope_ref[1]:.1f}':>15} {status}")

    # Zipf
    zipf = results.get('zipf_coefficient_mean', float('nan'))
    zipf_ref = reference_values.get('zipf_coefficient', (0, 0))
    in_range = zipf_ref[0] <= zipf <= zipf_ref[1] if not np.isnan(zipf) else False
    status = "✓" if in_range else "✗"
    print(f"Zipf Coefficient                {zipf:>15.3f} {f'{zipf_ref[0]:.1f}-{zipf_ref[1]:.1f}':>15} {status}")

    # Autocorrelation
    lr_corr = results.get('acf_long_range_correlation_mean', float('nan'))
    lr_ref = reference_values.get('acf_long_range_correlation', (0, 0))
    in_range = lr_corr >= lr_ref[0] if not np.isnan(lr_corr) else False
    status = "✓" if in_range else "✗"
    print(f"Long-Range Correlation          {lr_corr:>15.3f} {f'>{lr_ref[0]:.1f}':>15} {status}")

    decay = results.get('acf_decay_rate_mean', float('nan'))
    print(f"ACF Decay Rate                  {decay:>15.4f}")

    char_lag = results.get('acf_characteristic_lag_mean', float('nan'))
    print(f"Characteristic Lag              {char_lag:>15.1f}")

    print("-" * 65)
    print(f"Samples analyzed: {results.get('n_samples', 0)}")
    print(f"Mean sequence length: {results.get('mean_length', 0):.1f}")
    print("=" * 60)


def main():
    args = parse_args()

    # Set seed
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

    # Create tokenizer
    tokenizer = MidiTokenizer()

    # Generate samples
    print(f"\nGenerating {args.n_samples} samples...")
    samples = generate_samples(
        model,
        tokenizer,
        n_samples=args.n_samples,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print(f"Successfully generated {len(samples)} valid samples")

    # Evaluate
    results = evaluate_samples(samples)

    # Print results
    print_results(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in results.items()}, f, indent=2)

    # Plot example spectral analysis
    if samples:
        plot_spectral_analysis(
            samples[0],
            save_path=output_dir / 'example_spectral.png',
            title='Example Generated Sequence'
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
