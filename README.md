# ReK-Surv

Residual-enhanced Kolmogorov-Arnold Network for Survival Analysis

## Overview

This repository implements ReK-Surv, a survival analysis model based on Kolmogorov-Arnold Networks (KAN) with B-spline activation functions. The model combines the interpretability of KAN with the Cox proportional hazards framework for survival prediction.

## Key Features

- **B-spline Activation Functions**: Learnable univariate functions using B-spline basis
- **Residual Connections**: Identity shortcuts for same-dimension layers
- **Proximal L1 Regularization**: Soft thresholding for sparsity
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
| tau | L1 regularization (proximal) | 0.25 |
| wd | L2 regularization (weight decay) | 0.5 |
| depth | Number of hidden layers | 2 |
| width | Hidden layer width | 1 |

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
