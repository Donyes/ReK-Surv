"""
Kolmogorov-Arnold Network (KAN) for Survival Analysis

This module implements the KAN architecture with B-spline basis functions
for survival prediction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    KAN Linear layer with learnable B-spline activation functions.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        grid_size: Number of grid intervals for B-splines
        spline_order: Order of B-spline basis functions
        scale_noise: Scale of initialization noise
        scale_base: Scale for base weight initialization
        scale_spline: Scale for spline weight initialization
        base_activation: Base activation function class
        grid_range: Range of the grid for B-splines
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 2,
        spline_order: int = 2,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type = nn.ReLU,
        grid_eps: float = 0.01,
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Initialize grid for B-splines
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(self.grid.T[self.spline_order:-self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis functions."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Convert spline curves to coefficients."""
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """Get scaled spline weights."""
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # Base linear transformation
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        # Spline transformation
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output

        # Residual connection for same-dimension layers
        if self.in_features == self.out_features:
            output = output.view(*original_shape[:-1], self.out_features)
            residual = x.view(*original_shape[:-1], self.out_features)
            output = output + residual
        else:
            output = output.view(*original_shape[:-1], self.out_features)

        return output


class KAN(nn.Module):
    """
    Kolmogorov-Arnold Network for survival analysis.
    
    Args:
        layers_hidden: List of layer dimensions, e.g., [input_dim, hidden1, ..., 1]
        grid_size: Number of grid intervals for B-splines
        spline_order: Order of B-spline basis functions
    """
    
    def __init__(
        self,
        layers_hidden: list,
        grid_size: int = 2,
        spline_order: int = 2,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.5,
        base_activation: type = nn.ReLU,
        grid_eps: float = 0.01,
        grid_range: list = [-1, 1],
    ):
        super().__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
