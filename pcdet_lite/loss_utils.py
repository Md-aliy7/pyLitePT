"""
Loss Utilities
==============
Detection loss functions.
Adapted from OpenPCDet: pcdet/utils/loss_utils.py

Device-agnostic (works on CPU and GPU).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    
    Good for class imbalance - reduces loss for well-classified examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Focusing parameter for hard examples
            alpha: Balancing parameter for positive/negative examples
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """
        Numerically stable sigmoid cross entropy.
        
        max(x, 0) - x * z + log(1 + exp(-abs(x)))
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, N, C) or (N, C) predicted logits
            target: (B, N, C) or (N, C) one-hot targets
            weights: (B, N) or (N,) sample weights

        Returns:
            weighted_loss: same shape as input
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.dim() == 2 or (weights.dim() == 1 and target.dim() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.dim() == loss.dim()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss.
    
    smoothl1(x) = 0.5 * x^2 / beta    if |x| < beta
                  |x| - 0.5 * beta    otherwise
    """
    
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: L1 to L2 transition point
            code_weights: (#codes,) weights for each code element
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        self.code_weights = None
        self.code_weights_np = None
        
        if code_weights is not None:
            self.code_weights_np = np.array(code_weights, dtype=np.float32)

    def _get_code_weights(self, device):
        """Get code_weights tensor on the specified device."""
        if self.code_weights_np is None:
            return None
        if self.code_weights is None or self.code_weights.device != device:
            self.code_weights = torch.from_numpy(self.code_weights_np).to(device)
        return self.code_weights

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, N, C) or (N, C) predictions
            target: (B, N, C) or (N, C) targets
            weights: (B, N) or (N,) sample weights

        Returns:
            loss: weighted smooth L1 loss
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        
        # Code-wise weighting
        code_weights = self._get_code_weights(input.device)
        if code_weights is not None:
            diff = diff * code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # Sample-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Anchor-wise weighted cross entropy loss."""
    
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, N, C) logits
            target: (B, N, C) one-hot targets
            weights: (B, N) sample weights

        Returns:
            loss: (B, N) weighted cross entropy
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


class L1Loss(nn.Module):
    """Simple L1 loss."""
    
    def __init__(self):
        super(L1Loss, self).__init__()
       
    def forward(self, pred, target):
        if target.numel() == 0:
            return pred.sum() * 0
        assert pred.size() == target.size()
        loss = torch.abs(pred - target)
        return loss


class GaussianFocalLoss(nn.Module):
    """
    Gaussian focal loss for heatmap-based detection.
    
    Used in CenterPoint-style detectors.
    """

    def __init__(self, alpha=2.0, gamma=4.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        eps = 1e-12
        pos_weights = target.eq(1)
        neg_weights = (1 - target).pow(self.gamma)
        pos_loss = -(pred + eps).log() * (1 - pred).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pred + eps).log() * pred.pow(self.alpha) * neg_weights

        return pos_loss + neg_loss
