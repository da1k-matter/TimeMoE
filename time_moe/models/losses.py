import torch
from torch import nn

class CryptoLoss(nn.Module):
    """Huber loss with an auxiliary directional F1 term.

    The F1 component encourages the sign of the predictions to match
the sign of the targets which can be beneficial for financial
forecasting tasks.
    """

    def __init__(self, delta: float = 0.01, f1_weight: float = 0.1, reduction: str = "none"):
        super().__init__()
        self.delta = delta
        self.f1_weight = f1_weight
        self.reduction = reduction
        self.huber = nn.HuberLoss(delta=delta, reduction="none")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        base_loss = self.huber(predictions, targets)

        pred_sign = (predictions > 0).float()
        target_sign = (targets > 0).float()
        tp = (pred_sign * target_sign).sum()
        fp = (pred_sign * (1 - target_sign)).sum()
        fn = ((1 - pred_sign) * target_sign).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        directional_penalty = 1 - f1

        if self.reduction == "none":
            dir_term = torch.full_like(base_loss, directional_penalty)
            return base_loss + self.f1_weight * dir_term
        elif self.reduction == "mean":
            return base_loss.mean() + self.f1_weight * directional_penalty
        elif self.reduction == "sum":
            return base_loss.sum() + self.f1_weight * directional_penalty
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
