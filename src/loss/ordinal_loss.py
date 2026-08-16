import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalBCELoss(nn.Module):
    """
    Ordinal BCE loss for K ordered classes.

    Generates K-1 targets:
        y > 0
        y > 1
        ...
        y > K-2
    """

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        num_boundaries = logits.shape[1]

        boundary_targets = torch.stack(
            [
                (targets > boundary).float()
                for boundary in range(num_boundaries)
            ],
            dim=1,
        )

        return F.binary_cross_entropy_with_logits(
            logits,
            boundary_targets,
        )