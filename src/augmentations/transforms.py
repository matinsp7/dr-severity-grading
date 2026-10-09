from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]


def build_strong_train_transform(cfg):
    """
    Builds the strong training augmentation pipeline.
    """

    aug = cfg.augmentation
    image_size = cfg.dataset.image_size

    return transforms.Compose(
        [

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.RandomApply(
                [
                    transforms.RandomResizedCrop(
                        size=(image_size, image_size),
                        scale=(
                            aug.resized_crop.scale.min,
                            aug.resized_crop.scale.max,
                        ),
                        ratio=(
                            aug.resized_crop.ratio.min,
                            aug.resized_crop.ratio.max,
                        ),
                    )
                ],
                p=aug.resized_crop.p,
            ),

            transforms.RandomHorizontalFlip(
                p=aug.horizontal_flip.p
            ),

            transforms.RandomApply(
                [
                    transforms.RandomRotation(
                        degrees=aug.rotation.degrees
                    )
                ],
                p=aug.rotation.p
            ),

            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=aug.affine.degrees,
                        translate=(
                            aug.affine.translate,
                            aug.affine.translate,
                        ),
                        scale=(
                            aug.affine.scale.min,
                            aug.affine.scale.max,
                        ),
                        shear=aug.affine.shear
                    )
                ],
                p=aug.affine.p
            ),

            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=aug.color_jitter.brightness,
                        contrast=aug.color_jitter.contrast,
                        saturation=aug.color_jitter.saturation,
                        hue=aug.color_jitter.hue,
                    )
                ],
                p=aug.color_jitter.p,
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def build_valid_transform(cfg):
    """
    Builds the validation preprocessing pipeline.

    No stochastic augmentation is applied.
    """

    image_size = cfg.dataset.image_size

    return transforms.Compose(
        [

            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )