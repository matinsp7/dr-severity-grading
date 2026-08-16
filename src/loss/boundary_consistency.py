import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBoundaryConsistencyLoss(nn.Module):
    """
    Enforces consistency among:

    1. Categorical class probabilities converted into
       implicit ordinal boundary probabilities.

    2. Fused ordinal predictions.

    3. Global ordinal predictions.

    4. Local ordinal predictions.

    Boundary weights are sample-adaptive and depend on
    probability mass around neighboring classes.
    """

    def __init__(
        self,
        alpha: float = 2.0,
    ):
        super().__init__()

        self.alpha = alpha

    def forward(
        self,
        class_logits: torch.Tensor,
        ordinal_logits: torch.Tensor,
        global_ordinal_logits: torch.Tensor,
        local_ordinal_logits: torch.Tensor,
    ) -> torch.Tensor:

        class_probs = F.softmax(
            class_logits,
            dim=1,
        )

        # ---------------------------------------------------------
        # Implicit ordinal probabilities from class distribution
        #
        # C_k = P(y > k)
        # ---------------------------------------------------------
        cumulative_from_classes = torch.flip(
            torch.cumsum(
                torch.flip(
                    class_probs,
                    dims=[1],
                ),
                dim=1,
            ),
            dims=[1],
        )

        # Drop final boundary because P(y > 4) = 0
        cumulative_from_classes = (
            cumulative_from_classes[:, 1:]
        )

        # ---------------------------------------------------------
        # Boundary difficulty:
        # probability mass around adjacent classes.
        # Detach so this weighting mechanism does not introduce
        # an unwanted shortcut through the weighting branch.
        # ---------------------------------------------------------
        boundary_uncertainty = (
            class_probs[:, :-1]
            + class_probs[:, 1:]
        ).detach()

        boundary_weights = (
            1.0
            + self.alpha * boundary_uncertainty
        )

        fused_boundary_probs = torch.sigmoid(
            ordinal_logits
        )

        global_boundary_probs = torch.sigmoid(
            global_ordinal_logits
        )

        local_boundary_probs = torch.sigmoid(
            local_ordinal_logits
        )

        # ---------------------------------------------------------
        # 1. Classification ↔ fused ordinal consistency
        # ---------------------------------------------------------
        fused_error = (
            fused_boundary_probs
            - cumulative_from_classes
        ).pow(2)

        # ---------------------------------------------------------
        # 2. Global ↔ local ordinal consistency
        # ---------------------------------------------------------
        branch_error = (
            global_boundary_probs
            - local_boundary_probs
        ).pow(2)

        # ---------------------------------------------------------
        # 3. Fused ↔ branch agreement
        # ---------------------------------------------------------
        fusion_error = 0.5 * (
            (
                fused_boundary_probs
                - global_boundary_probs
            ).pow(2)
            +
            (
                fused_boundary_probs
                - local_boundary_probs
            ).pow(2)
        )

        total_error = (
            fused_error
            + branch_error
            + fusion_error
        )

        weighted_error = (
            boundary_weights
            * total_error
        )

        return weighted_error.mean()