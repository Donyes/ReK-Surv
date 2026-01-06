# ReK-Surv

Residual-enhanced Kolmogorov-Arnold Network for Survival Analysis

## Overview

This repository implements ReK-Surv, a survival analysis model based on Kolmogorov-Arnold Networks (KAN) with B-spline activation functions. The model combines the interpretability of KAN with the Cox proportional hazards framework for survival prediction.

## Key Features

- **B-spline Activation Functions**: Learnable univariate functions using B-spline basis
- **Residual Connections**: Identity shortcuts for same-dimension layers
- **Elastic Net Regularization**: L1 via proximal soft-thresholding + L2 via weight decay
- **Cox Partial Likelihood Loss**: Standard survival analysis objective

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
ReK-Surv/
├── models/
│   ├── __init__.py
│   └── kan.py              # KAN model implementation
├── utils/
│   ├── __init__.py
│   ├── loss.py             # Loss functions
│   └── data.py             # Data loading utilities
├── experiments/
│   └── hyperparameter_sensitivity.py
├── data/
│   └── metabric.h5         # METABRIC dataset (not included)
├── train.py                # Training script
├── requirements.txt
└── README.md
```

## Usage

### Training

```bash
# Train on METABRIC dataset with default parameters
python train.py --dataset metabric --data_path data/metabric.h5

# Custom hyperparameters
python train.py \
    --grid_size 1 \
    --spline_order 3 \
    --tau 0.25 \
    --wd 0.5 \
    --depth 2 \
    --epochs 500
```

### Hyperparameter Sensitivity Analysis

```bash
cd experiments
python hyperparameter_sensitivity.py
```

This generates:
- `stage_a_capacity.csv`: Spline order × Grid size results
- `stage_b_regularization.csv`: L1 × L2 regularization results
- `stage_c_width.csv`: Hidden layer width results
- `stage_d_depth.csv`: Network depth results
- `sensitivity_results.png`: Visualization

## Model Architecture

The KAN model uses the following architecture:

```
Input (n_features) → KANLinear → ... → KANLinear → Output (1)
```

Each KANLinear layer computes:
```
output = base_activation(x) @ base_weight + B_splines(x) @ spline_weight + residual
```

## Hyperparameters

| Parameter | Description | Default (METABRIC) |
|-----------|-------------|-------------------|
| grid_size | B-spline grid intervals | 1 |
| spline_order | B-spline order | 3 |
| tau | L1 regularization (proximal soft-thresholding) | 0.25 |
| wd | L2 regularization (AdamW weight decay) | 0.5 |
| depth | Number of hidden layers | 2 |
| width | Hidden layer width | 1 |

### Elastic Net Regularization

The model uses Elastic Net regularization combining L1 and L2 penalties:

- **L1 (τ)**: Implemented via proximal gradient method (soft-thresholding) applied after each optimizer step
- **L2 (wd)**: Implemented via AdamW optimizer's built-in weight decay

## Citation

If you use this code, please cite:

```bibtex
@article{rek-surv,
  title={ReK-Surv: Residual-enhanced Kolmogorov-Arnold Networks for Survival Analysis},
  author={...},
  journal={...},
  year={2025}
}
```

## License

MIT License
