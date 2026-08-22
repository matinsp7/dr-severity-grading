import torch.nn as nn

from src.loss.adaptive_ordinal import (
    AdaptiveOrdinalLoss,
)


def build_loss(cfg):

    if cfg.loss.name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if cfg.loss.name == "focal":
        from src.loss.focal_loss import FocalLoss

        return FocalLoss(
            gamma=cfg.loss.gamma,
        )

    if cfg.loss.name == "adaptive_ordinal":
        return AdaptiveOrdinalLoss(
            lambda_ordinal=cfg.loss.lambda_ordinal,
            lambda_boundary=cfg.loss.lambda_boundary,
            boundary_uncertainty_alpha=(
                cfg.loss.boundary_uncertainty_alpha
            ),
            boundary_disagreement_beta=(
                cfg.loss.boundary_disagreement_beta
            ),
        )

    raise ValueError(
        f"Unknown loss: {cfg.loss.name}"
    )