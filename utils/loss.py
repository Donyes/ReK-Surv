"""
Loss functions for survival analysis.

This module implements the Cox partial likelihood loss and
regularization methods for ReK-Surv models.
"""

import torch


def negative_log_likelihood(y_pred: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """
    Compute the negative log partial likelihood for Cox proportional hazards model.
    
    The data must be sorted by survival time in descending order.
    
    Args:
        y_pred: Predicted risk scores (log hazard ratios), shape (n_samples,) or (n_samples, 1)
        event: Event indicators (1=event, 0=censored), shape (n_samples,)
    
    Returns:
        Normalized negative log partial likelihood
    
    Note:
        This implementation assumes samples are sorted by time in descending order,
        which is required for the cumulative sum to correctly compute the risk set.
    """
    y_pred = y_pred.squeeze()
    
    hazard_ratio = torch.exp(y_pred)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0) + 1e-10)
    uncensored_likelihood = y_pred - log_risk
    censored_likelihood = uncensored_likelihood * event
    
    num_events = torch.sum(event) + 1e-10
    neg_likelihood = -torch.sum(censored_likelihood) / num_events
    
    return neg_likelihood


def proximal_l1(model: torch.nn.Module, tau: float, lr: float) -> None:
    """
    Apply proximal L1 regularization (soft thresholding) to model parameters.
    
    This implements the proximal gradient method for L1 regularization,
    which is more accurate than adding L1 penalty to the loss function.
    
    Args:
        model: PyTorch model
        tau: L1 regularization strength
        lr: Current learning rate
    
    Mathematical formulation:
        Soft thresholding: S_λ(w) = sign(w) * max(|w| - λ, 0)
        where λ = tau * lr
    """
    threshold = tau * lr
    
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                sign = torch.sign(param)
                shrinkage = torch.clamp(torch.abs(param) - threshold, min=0.0)
                param.data = sign * shrinkage
