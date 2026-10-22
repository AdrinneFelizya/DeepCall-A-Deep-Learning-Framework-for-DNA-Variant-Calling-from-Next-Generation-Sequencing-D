"""
Focal Loss for highly imbalanced variant calling.

Variant sites are extremely rare compared to reference sites. Focal loss
down-weights easy (well-classified) examples so the model focuses on hard,
rare variant sites during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


__all__ = ["FocalLoss"]