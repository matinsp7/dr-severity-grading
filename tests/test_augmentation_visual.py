from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from src.augmentations.builder import build_train_augmentation
from src.utils.config import load_config


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def denormalize(image):

    mean = image.new_tensor(
        IMAGENET_MEAN
    ).view(3, 1, 1)

    std = image.new_tensor(
        IMAGENET_STD
    ).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)


def main():

    cfg = load_config(
        "../configs/exp003_augmentation.yaml"
    )

    transform = build_train_augmentation(
        cfg
    )

    image_path = Path(
        "../data/train_images/1b8ad0afe9fb.png"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    augmented = transform(
        image
    )

    augmented = denormalize(
        augmented
    )

    original = image.resize(
        (
            cfg.dataset.image_size,
            cfg.dataset.image_size,
        )
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(
        augmented.permute(1, 2, 0)
    )
    axes[1].set_title(
        f"Augmented ({cfg.augmentation.name})"
    )
    axes[1].axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()