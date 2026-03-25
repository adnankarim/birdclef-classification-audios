from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probas = torch.sigmoid(logits)
        xs_pos = probas
        xs_neg = 1.0 - probas
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp_min(self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp_min(self.eps))
        loss = los_pos + los_neg
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            loss = loss * torch.pow(1.0 - pt, gamma)
        return -loss.mean()


def linear_softmax_pooling(frame_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(frame_logits)
    pooled = probs.square().sum(dim=1) / probs.sum(dim=1).clamp_min(1e-6)
    return torch.logit(pooled.clamp(1e-4, 1.0 - 1e-4))


class DistillationLoss(nn.Module):
    def __init__(self, hard_weight: float = 1.0, soft_weight: float = 1.0, temperature: float = 1.0):
        super().__init__()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.temperature = temperature
        self.hard_loss = AsymmetricLossMultiLabel()

    def forward(
        self,
        student_logits: torch.Tensor,
        hard_targets: torch.Tensor,
        teacher_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.hard_weight * self.hard_loss(student_logits, hard_targets)
        if teacher_probs is not None:
            teacher_probs = torch.nan_to_num(teacher_probs.float(), nan=0.0, posinf=1.0, neginf=0.0)
            teacher_probs = teacher_probs.clamp(1e-4, 1.0 - 1e-4)
            student_probs = torch.sigmoid(student_logits / self.temperature)
            soft = F.binary_cross_entropy(student_probs, teacher_probs, reduction="mean")
            loss = loss + self.soft_weight * soft
        return loss
