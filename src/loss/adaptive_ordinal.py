import torch.nn as nn
import torch.nn.functional as F

from src.loss.ordinal_loss import OrdinalBCELoss
from src.loss.boundary_consistency import (
    AdaptiveBoundaryConsistencyLoss,
)


class AdaptiveOrdinalLoss(nn.Module):

    def __init__(
        self,
        lambda_ordinal: float = 1.0,
        lambda_boundary: float = 0.5,
        boundary_uncertainty_alpha: float = 2.0,
        boundary_disagreement_beta: float = 2.0,
    ):
        super().__init__()

        self.lambda_ordinal = lambda_ordinal
        self.lambda_boundary = lambda_boundary

        self.ordinal_loss = OrdinalBCELoss()

        self.boundary_consistency = (
            AdaptiveBoundaryConsistencyLoss(
                uncertainty_alpha=boundary_uncertainty_alpha,
                disagreement_beta=boundary_disagreement_beta,
            )
        )

    def forward(
        self,
        outputs,
        targets,
    ):

        class_logits = outputs["class_logits"]

        ordinal_logits = outputs["ordinal_logits"]

        global_ordinal_logits = (
            outputs["global_ordinal_logits"]
        )

        local_ordinal_logits = (
            outputs["local_ordinal_logits"]
        )

        classification_loss = F.cross_entropy(
            class_logits,
            targets,
        )

        ordinal_loss = self.ordinal_loss(
            ordinal_logits,
            targets,
        )
        boundary_loss = (
            self.boundary_consistency(
                class_logits=class_logits,
                ordinal_logits=ordinal_logits,
                global_ordinal_logits=global_ordinal_logits,
                local_ordinal_logits=local_ordinal_logits,
            )
        )

        total_loss = (
            classification_loss
            + self.lambda_ordinal * ordinal_loss
            + self.lambda_boundary * boundary_loss
        )

        return total_loss