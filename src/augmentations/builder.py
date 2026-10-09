from src.augmentations.transforms import (
    build_strong_train_transform,
    build_valid_transform,
)


def build_train_augmentation(cfg):
    """
    Builds the training augmentation pipeline
    based on the configured augmentation preset.
    """

    name = cfg.augmentation.name.lower()

    if name == "strong":
        return build_strong_train_transform(cfg)

    if name == "none":
        return build_valid_transform(cfg)

    raise ValueError(
        f"Unknown augmentation preset: '{name}'. "
        "Supported presets: ['none', 'strong']"
    )


def build_valid_augmentation(cfg):
    """
    Builds the validation preprocessing pipeline.

    Validation must never use stochastic training
    augmentation.
    """

    return build_valid_transform(cfg)