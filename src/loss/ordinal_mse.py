import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalMSELoss(nn.Module):
    def __init__(self, num_classes: int = 5, reduction: str = "mean"):
        super().__init__()

        self.reduction = reduction

        self.register_buffer(
            "class_indices",
            torch.arange(num_classes, dtype=torch.float32)
        )

    def forward(self, logits, targets):
        probabilities = F.softmax(logits, dim=1)

        expected_score = torch.sum(
            probabilities * self.class_indices,
            dim=1
        )

        loss = F.mse_loss(
            expected_score,
            targets.float(),
            reduction=self.reduction
        )

        return loss