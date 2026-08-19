import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalMSELoss(nn.Module):
    """
    MSE loss for ordinal multi-class classification.

    Converts class probabilities into an expected ordinal score
    and computes MSE against the ground-truth class index.
    """

    def __init__(self, num_classes: int = 5, reduction: str = "mean"):
        super().__init__()

        self.num_classes = num_classes
        self.reduction = reduction

        self.register_buffer(
            "class_indices",
            torch.arange(num_classes, dtype=torch.float32),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            inputs:
                Raw model logits.
                Shape: (batch_size, num_classes)

            targets:
                Ground-truth ordinal labels.
                Shape: (batch_size,)

        Returns:
            Ordinal MSE loss.
        """

        probabilities = F.softmax(inputs, dim=1)

        expected_severity = (
            probabilities * self.class_indices
        ).sum(dim=1)

        targets = targets.float()

        return F.mse_loss(
            expected_severity,
            targets,
            reduction=self.reduction,
        )