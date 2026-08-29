import torch.nn as nn

from src.loss.focal_loss import MultiClassFocalLoss
from src.loss.ordinal_mse import OrdinalMSELoss

from src.imbalance.class_weights import (
    compute_class_weights,
)

def build_loss(cfg, train_labels=None):

    loss_name = cfg.loss.name.lower()

    if loss_name == "cross_entropy":

        class_weights = None

        if cfg.imbalance.use_class_weights:

            if train_labels is None:
                raise ValueError(
                    "train_labels are required when "
                    "use_class_weights=True."
                )

            class_weights = compute_class_weights(
                labels=train_labels,
                num_classes=cfg.model.num_classes,
            )

        return nn.CrossEntropyLoss(
            weight=class_weights
        )

    if loss_name == "focal_loss":
        return MultiClassFocalLoss(None, gamma=cfg.loss.gamma)

    if loss_name == "mse":
        return OrdinalMSELoss(
            num_classes=cfg.model.num_classes,
        )

    raise ValueError(
        f"Unsupported loss: {cfg.loss.name}"
    )