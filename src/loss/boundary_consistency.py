import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveBoundaryConsistencyLoss(nn.Module):
    """
    Adaptive boundary consistency loss.

    Boundary importance depends on two signals:

    1. Local class uncertainty:
       p_k + p_{k+1}

    2. Disagreement between:
       - implicit ordinal probabilities from the classification head
       - explicit ordinal probabilities from the fused ordinal head

    The weighting signals are detached from the graph.
    """

    def __init__(
        self,
        uncertainty_alpha: float = 2.0,
        disagreement_beta: float = 2.0,
    ):
        super().__init__()

        self.uncertainty_alpha = uncertainty_alpha
        self.disagreement_beta = disagreement_beta

    @staticmethod
    def _implicit_boundary_probabilities(
        class_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert 5-class probabilities into 4 cumulative
        ordinal probabilities:

            P(y > 0)
            P(y > 1)
            P(y > 2)
            P(y > 3)
        """

        return torch.flip(
            torch.cumsum(
                torch.flip(
                    class_probs,
                    dims=[1],
                ),
                dim=1,
            ),
            dims=[1],
        )[:, 1:]

    def forward(
        self,
        class_logits: torch.Tensor,
        ordinal_logits: torch.Tensor,
        global_ordinal_logits: torch.Tensor,
        local_ordinal_logits: torch.Tensor,
    ) -> torch.Tensor:

        # ---------------------------------------------------------
        # Classification probabilities
        # ---------------------------------------------------------
        class_probs = F.softmax(
            class_logits,
            dim=1,
        )

        # ---------------------------------------------------------
        # Convert categorical probabilities into ordinal
        # boundary probabilities.
        # ---------------------------------------------------------
        implicit_boundary_probs = (
            self._implicit_boundary_probabilities(
                class_probs
            )
        )

        # ---------------------------------------------------------
        # Explicit ordinal predictions
        # ---------------------------------------------------------
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
        # Signal 1:
        # difficulty of neighboring class boundary.
        # ---------------------------------------------------------
        boundary_uncertainty = (
            class_probs[:, :-1]
            + class_probs[:, 1:]
        )
        # ---------------------------------------------------------
        # Signal 2:
        # classification-vs-ordinal disagreement.
        # ---------------------------------------------------------
        boundary_disagreement = (
            implicit_boundary_probs
            - fused_boundary_probs
        ).abs()

        # ---------------------------------------------------------
        # Adaptive weights.
        #
        # Detach the signals so the model cannot optimize the
        # weighting mechanism itself as a shortcut.
        # ---------------------------------------------------------
        boundary_weights = (
            1.0
            + self.uncertainty_alpha
            * boundary_uncertainty.detach()
            + self.disagreement_beta
            * boundary_disagreement.detach()
        )

        # ---------------------------------------------------------
        # 1. Classification ↔ fused ordinal consistency
        # ---------------------------------------------------------
        fused_error = (
            fused_boundary_probs
            - implicit_boundary_probs.detach()
        ).pow(2)

        # ---------------------------------------------------------
        # 2. Global ↔ local ordinal consistency
        # ---------------------------------------------------------
        branch_error = (
            global_boundary_probs
            - local_boundary_probs
        ).pow(2)

        # ---------------------------------------------------------
        # 3. Fused ↔ global/local agreement
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