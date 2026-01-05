#!/usr/bin/env python
"""
Run complete experiment comparing baseline vs Pink Noise Prior.

This script runs both experiments, evaluates both models, and generates
a comprehensive comparison report.

Usage:
    python -m pnp.scripts.run_experiment --config configs/experiment.yaml
    python -m pnp.scripts.run_experiment --quick  # Fast debug run
"""

import argparse
from pathlib import Path
import json
import torch
import numpy as np

from pnp.data.tokenizer import MidiTokenizer
from pnp.data.maestro import create_maestro_dataloaders, download_maestro
from pnp.models.transformer import create_model, MusicTransformerConfig
from pnp.training.trainer import Trainer, TrainingConfig
from pnp.utils.config import ExperimentConfig
from pnp.utils.visualization import (
    plot_training_curves,
    plot_comparison,
    plot_evaluation_metrics,
    create_experiment_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run baseline vs Pink Noise Prior experiment"
    )

    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to experiment config YAML'
    )
    parser.add_argument(
        '--output-dir', type=str, default='experiments',
        help='Output directory for results'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick debug run with reduced settings'
    )
    parser.add_argument(
        '--skip-baseline', action='store_true',
        help='Skip baseline training (if already done)'
    )
    parser.add_argument(
        '--skip-pink', action='store_true',
        help='Skip pink noise prior training (if already done)'
    )
    parser.add_argument(
        '--data-dir', type=str, default=None,
        help='Path to MAESTRO dataset'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )

    return parser.parse_args()


def run_training(
    name: str,
    train_loader,
    val_loader,
    model_config: MusicTransformerConfig,
    training_config: TrainingConfig,
    output_dir: Path,
):
    """Run a single training experiment."""
    print(f"\n{'='*60}")
    print(f"Training: {name}")
    print(f"{'='*60}")

    model = create_model(
        vocab_size=model_config.vocab_size,
        size='small',
        max_seq_length=model_config.max_seq_length,
    )

    training_config.experiment_name = name

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )

    history = trainer.train()

    # Save
    model.save_pretrained(output_dir / name / 'model.pt')
    plot_training_curves(history, output_dir / name / 'training_curves.png', name)

    return model, history


def evaluate_model(model, tokenizer, device, n_samples=50, max_length=256):
    """Evaluate a trained model."""
    from pnp.metrics.fractal import batch_analyze_sequences
    from pnp.metrics.zipf import batch_analyze_zipf
    from pnp.metrics.autocorrelation import batch_analyze_autocorrelation

    model.eval()
    samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            start = torch.tensor([[tokenizer.bos_token_id]], device=device)
            output = model.generate(
                start,
                max_new_tokens=max_length,
                temperature=1.0,
                top_k=50,
            )
            pitches = tokenizer.decode(output[0].cpu().tolist())
            if len(pitches) > 20:
                samples.append(np.array(pitches))

    if not samples:
        return {}

    # Compute metrics
    fractal = batch_analyze_sequences(samples)
    zipf = batch_analyze_zipf(samples)
    acf = batch_analyze_autocorrelation(samples)

    return {
        'fractal_dimension': fractal['fractal_dimension_mean'],
        'spectral_slope': fractal['spectral_slope_mean'],
        'zipf_coefficient': zipf['zipf_coefficient_mean'],
        'long_range_correlation': acf['long_range_correlation_mean'],
        'characteristic_lag': acf['characteristic_lag_mean'],
        'n_samples': len(samples),
    }


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode settings
    if args.quick:
        epochs = 2
        max_files = 10
        n_eval_samples = 10
        batch_size = 8
        seq_length = 128
    else:
        epochs = 50
        max_files = None
        n_eval_samples = 100
        batch_size = 32
        seq_length = 512

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup data
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = download_maestro()
    else:
        data_dir = Path(data_dir)

    tokenizer = MidiTokenizer()

    print("Loading data...")
    train_loader, val_loader, test_loader = create_maestro_dataloaders(
        data_dir,
        tokenizer=tokenizer,
        seq_length=seq_length,
        batch_size=batch_size,
        max_files=max_files,
    )

    # Model config
    model_config = MusicTransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=seq_length,
    )

    results = {}

    # Baseline training
    if not args.skip_baseline:
        baseline_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            lambda_pink=0.0,  # No spectral loss
            checkpoint_dir=str(output_dir),
        )
        baseline_model, baseline_history = run_training(
            'baseline',
            train_loader, val_loader,
            model_config, baseline_config,
            output_dir,
        )
        results['baseline_history'] = baseline_history

        # Evaluate
        print("\nEvaluating baseline...")
        baseline_model.to(device)
        results['baseline_metrics'] = evaluate_model(
            baseline_model, tokenizer, device, n_eval_samples
        )

    # Pink Noise Prior training
    if not args.skip_pink:
        pink_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            lambda_pink=0.1,
            target_slope=-1.0,
            checkpoint_dir=str(output_dir),
        )
        pink_model, pink_history = run_training(
            'pink_prior',
            train_loader, val_loader,
            model_config, pink_config,
            output_dir,
        )
        results['pink_history'] = pink_history

        # Evaluate
        print("\nEvaluating Pink Noise Prior...")
        pink_model.to(device)
        results['pink_metrics'] = evaluate_model(
            pink_model, tokenizer, device, n_eval_samples
        )

    # Generate comparison report
    if 'baseline_history' in results and 'pink_history' in results:
        print("\nGenerating comparison report...")
        plot_comparison(
            results['baseline_history'],
            results['pink_history'],
            output_dir / 'comparison.png',
        )

    if 'baseline_metrics' in results and 'pink_metrics' in results:
        plot_evaluation_metrics(
            results['baseline_metrics'],
            results['pink_metrics'],
            output_dir / 'evaluation_comparison.png',
        )

    # Save all results
    with open(output_dir / 'results.json', 'w') as f:
        # Convert to serializable format
        save_results = {}
        for k, v in results.items():
            if 'history' in k:
                save_results[k] = {kk: [float(vv) for vv in vvv]
                                   for kk, vvv in v.items()}
            elif 'metrics' in k:
                save_results[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                                   for kk, vv in v.items()}
        json.dump(save_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    if 'baseline_metrics' in results:
        print("\nBaseline Metrics:")
        for k, v in results['baseline_metrics'].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if 'pink_metrics' in results:
        print("\nPink Noise Prior Metrics:")
        for k, v in results['pink_metrics'].items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if 'baseline_metrics' in results and 'pink_metrics' in results:
        print("\nImprovement (Pink - Baseline):")
        for k in results['baseline_metrics']:
            if k in results['pink_metrics'] and isinstance(results['baseline_metrics'][k], float):
                diff = results['pink_metrics'][k] - results['baseline_metrics'][k]
                print(f"  {k}: {diff:+.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
