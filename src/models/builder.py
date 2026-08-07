import timm

from src.models.efficientnet import EfficientNetDR


def build_model(cfg):

    if cfg.model.name == "efficientnet_b0":
        return EfficientNetDR(
            num_classes=cfg.model.num_classes
        )

    raise ValueError(f"Unknown model: {cfg.model.name}")