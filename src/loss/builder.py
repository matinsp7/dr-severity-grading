import torch.nn as nn

from src.loss.focal_loss import MultiClassFocalLoss
from src.loss.ordinal_mse import OrdinalMSELoss


def build_loss(cfg):

    loss_name = cfg.loss.name.lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if loss_name == "focal_loss":
        return MultiClassFocalLoss(None, gamma=cfg.loss.gamma)

    if loss_name == "mse":
        return OrdinalMSELoss(
            num_classes=cfg.model.num_classes,
        )

    raise ValueError(
        f"Unsupported loss: {cfg.loss.name}"
    )