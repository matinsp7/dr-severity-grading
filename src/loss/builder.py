import torch.nn as nn

from src.loss.focal_loss import MultiClassFocalLoss


def build_loss(cfg):

    loss_name = cfg.loss.name.lower()

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if loss_name == "focal_loss":
        return MultiClassFocalLoss(None, gamma=cfg.loss.gamma)

    raise ValueError(
        f"Unsupported loss: {cfg.loss.name}"
    )