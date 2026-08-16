from src.models.dual_branch import DualBranchDR
from src.models.efficientnet import EfficientNetDR


def build_model(cfg):

    if cfg.model.name == "efficientnet_b0":
        return EfficientNetDR(
            num_classes=cfg.model.num_classes
        )

    if cfg.model.name == "dual_branch":
        return DualBranchDR(
            global_backbone=cfg.model.global_backbone,
            local_backbone=cfg.model.local_backbone,
            num_classes=cfg.model.num_classes,
            image_size=cfg.model.image_size,
            fusion_dim=cfg.model.fusion_dim,
            dropout=cfg.model.dropout,
            pretrained=cfg.model.pretrained,
        )

    raise ValueError(
        f"Unknown model: {cfg.model.name}"
    )