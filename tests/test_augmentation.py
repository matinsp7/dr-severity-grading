from src.utils.config import load_config
from src.augmentations.builder import (
    build_train_augmentation,
    build_valid_augmentation,
)

from PIL import Image
import torch


cfg = load_config(
    "../configs/exp003_augmentation.yaml"
)


train_transform = build_train_augmentation(
    cfg
)

valid_transform = build_valid_augmentation(
    cfg
)


image = Image.open(
    "../data/train_images/1b8ad0afe9fb.png"
).convert("RGB")


train_image = train_transform(
    image
)

valid_image = valid_transform(
    image
)


print(
    "Train shape:",
    train_image.shape
)

print(
    "Valid shape:",
    valid_image.shape
)

print(
    "Train dtype:",
    train_image.dtype
)

print(
    "Valid dtype:",
    valid_image.dtype
)

print(
    "Train is tensor:",
    isinstance(train_image, torch.Tensor)
)

print(
    "Valid is tensor:",
    isinstance(valid_image, torch.Tensor)
)

valid_transform = build_valid_augmentation(
    cfg
)

image_1 = valid_transform(image)
image_2 = valid_transform(image)

print(
    "Validation deterministic:",
    torch.equal(image_1, image_2)
)