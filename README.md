# Pink Noise Prior

**Enforcing Fractal Dynamics in Symbolic Music Generation via Differentiable Spectral Regularization**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This research project implements a novel approach to symbolic music generation that addresses the "structural amnesia" problem in Transformer-based models. Standard models trained with Cross-Entropy loss optimize for next-token prediction but have no mathematical incentive to maintain long-range structural coherence.

**Key Innovation:** We introduce a differentiable **Spectral Slope Loss** that enforces 1/f (pink noise) spectral characteristics during training, effectively constraining the model to produce music with fractal dynamics similar to human-composed music.

## The Problem

Standard Transformer models (like Music Transformer or GPT-based MusicGen) suffer from:
- **Structural amnesia**: Generated music "wanders" and loses global coherence
- **Local optimization**: Cross-entropy loss only optimizes for the next immediate note
- **No fractal structure**: Generated sequences lack the 1/f characteristics found in high-quality music

## Our Solution

We add a differentiable "Pink Noise Prior" to the training loss:

```
L_total = L_CE + λ × L_Pink
```

Where `L_Pink` penalizes deviation from the target spectral slope (-1.0 for pink noise).

### How It Works

1. **Soft Representation**: Convert model logits to differentiable "expected pitch" values
2. **Differentiable FFT**: Apply `torch.fft.rfft` to get the frequency spectrum
3. **Power Spectral Density**: Compute magnitude squared
4. **Slope Estimation**: Calculate log-log slope via differentiable linear regression
5. **Penalty**: `L_Pink = ||slope - (-1.0)||²`

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/pink-noise-prior.git
cd pink-noise-prior

# Install dependencies
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Quick Start

### 1. Download Data

```bash
# Download MAESTRO dataset
python -m pnp.scripts.download_data --analyze
```

### 2. Train Models

```bash
# Train baseline (CE loss only)
python -m pnp.scripts.train --lambda-pink 0.0 --experiment-name baseline

# Train with Pink Noise Prior
python -m pnp.scripts.train --lambda-pink 0.1 --experiment-name pink_prior
```

### 3. Evaluate

```bash
# Evaluate trained model
python -m pnp.scripts.evaluate --checkpoint outputs/pink_prior/best.pt --n-samples 100
```

### 4. Generate Music

```bash
# Generate MIDI files
python -m pnp.scripts.generate --checkpoint outputs/pink_prior/best.pt --n-samples 10
```

## Running the Full Experiment

Run the complete baseline vs. pink noise prior comparison:

```bash
# Full experiment (takes several hours)
python -m pnp.scripts.run_experiment --output-dir experiments/full

# Quick debug run
python -m pnp.scripts.run_experiment --quick
```

## Project Structure

```
pnp/
├── data/           # Data loading and preprocessing
│   ├── tokenizer.py    # MIDI pitch tokenization
│   ├── dataset.py      # PyTorch datasets
│   └── maestro.py      # MAESTRO dataset handling
├── models/         # Neural network architectures
│   └── transformer.py  # Causal Transformer (mini-GPT)
├── losses/         # Loss functions
│   └── spectral.py     # Spectral Slope Loss (core innovation)
├── metrics/        # Evaluation metrics
│   ├── fractal.py      # Higuchi Fractal Dimension
│   ├── zipf.py         # Zipf's Law analysis
│   └── autocorrelation.py
├── training/       # Training utilities
│   ├── trainer.py      # Training loop
│   └── scheduler.py    # Learning rate schedulers
├── utils/          # Utilities
│   ├── config.py       # Configuration management
│   └── visualization.py
└── scripts/        # Command-line scripts
    ├── train.py
    ├── evaluate.py
    ├── generate.py
    └── run_experiment.py
```

## Configuration

Create a config file or use command-line arguments:

```yaml
# configs/my_experiment.yaml
name: my_experiment
model_size: small
epochs: 50
lambda_pink: 0.1
target_slope: -1.0
```

```bash
python -m pnp.scripts.train --config configs/my_experiment.yaml
```

## Evaluation Metrics

We use objective metrics to evaluate generated music:

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Fractal Dimension** | Higuchi FD - measures complexity | 1.3 - 1.5 |
| **Spectral Slope** | 1/f characteristics | -1.2 to -0.8 |
| **Zipf Coefficient** | Rank-frequency distribution | 0.8 - 1.2 |
| **Long-Range Correlation** | Autocorrelation at lag 100 | > 0.1 |

## Steerability

Once trained, you can adjust the target slope to control musical complexity:

| Target Slope | Effect |
|--------------|--------|
| -2.0 | Brownian noise (smoother, more predictable) |
| -1.0 | Pink noise (natural, musical structure) |
| -0.5 | Chaotic/modern (more unpredictable) |

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_spectral_loss.py -v

# Run with coverage
pytest tests/ --cov=pnp --cov-report=html
```

## Key Files

- **`pnp/losses/spectral.py`**: Core innovation - differentiable spectral slope loss
- **`pnp/models/transformer.py`**: Lightweight causal Transformer
- **`pnp/metrics/fractal.py`**: Higuchi Fractal Dimension implementation
- **`pnp/training/trainer.py`**: Training loop with combined loss

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pinknoisepior2024,
  title={The Pink Noise Prior: Enforcing Fractal Dynamics in Symbolic Music Generation via Differentiable Spectral Regularization},
  year={2024}
}
```

## References

- **Auraloss**: Steinmetz et al. - Audio-focused loss functions (adapted for symbolic)
- **Higuchi's Algorithm**: Higuchi, T. (1988) - Fractal dimension estimation
- **Music Transformer**: Huang et al. - Base architecture inspiration

## License

MIT License - see [LICENSE](LICENSE) for details.
