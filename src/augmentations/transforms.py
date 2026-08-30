
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def build_strong_train_transform(cfg):
    image_size = cfg.dataset.image_size
    aug = cfg.augmentation

    return A.Compose([
        A.Resize(
            image_size,
            image_size
        ),
        A.HorizontalFlip(
            p=aug.horizontal_flip.p
        ),
        A.Rotate(
            limit=aug.rotation.degrees,
            p=aug.rotation.p
        ),
        A.Affine(
            scale=(
                aug.affine.scale.min,
                aug.affine.scale.max
            ),
            translate_percent=(
                0.0,
                aug.affine.translate
            ),
            rotate=(
                -aug.affine.rotate,
                aug.affine.rotate
            ),
            p=aug.affine.p
        ),
        A.RandomBrightnessContrast(
            brightness_limit=aug.color_jitter.brightness,
            contrast_limit=aug.color_jitter.contrast,
            p=aug.color_jitter.p
        ),
        A.HueSaturationValue(
            hue_shift_limit=aug.color_jitter.hue,
            sat_shift_limit=aug.color_jitter.saturation,
            val_shift_limit=aug.color_jitter.value,
            p=aug.color_jitter.hue_p
        ),
        A.GaussNoise(
            std_range=(
                aug.gauss_noise.min,
                aug.gauss_noise.max
            ),
            p=aug.gauss_noise.p
        ),
        A.CLAHE(
            clip_limit=aug.clahe.clip_limit,
            p=aug.clahe.p
        ),
        A.Normalize(
            mean=tuple(IMAGENET_MEAN),
            std=tuple(IMAGENET_STD)
        ),
        ToTensorV2()

    ], seed=cfg.seed)

def build_valid_transform(cfg):
    """
    Transform pipeline for VALIDATION and TEST data.
    """
    image_size = cfg.dataset.image_size
    return A.Compose([
        A.Resize(image_size, image_size),
        # If you use CLAHE in preprocessing, ensure valid/test images also have it applied.
        A.Normalize(mean=tuple(IMAGENET_MEAN), std=tuple(IMAGENET_STD)),
        ToTensorV2()
    ], seed=42)


