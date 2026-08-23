from dataclasses import dataclass, field, asdict
from typing import Any
import yaml


# ============================================================
# Dataset
# ============================================================

@dataclass
class DatasetConfig:
    train_csv: str
    valid_csv: str
    test_csv: str

    train_dir: str
    valid_dir: str
    test_dir: str

    image_size: int

    class_names: list[str]


# ============================================================
# DataLoader
# ============================================================

@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool


# ============================================================
# Model
# ============================================================

@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    num_classes: int

    global_backbone: str = "swin_tiny_patch4_window7_224"
    local_backbone: str = "efficientnet_b0"

    image_size: int = 384
    fusion_dim: int = 512
    dropout: float = 0.2


# ============================================================
# Optimizer
# ============================================================

@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


# ============================================================
# Scheduler
# ============================================================

@dataclass
class SchedulerConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

# ============================================================
# Loss
# ============================================================

@dataclass
class LossConfig:
    name: str
    gamma: float = 2.0

    lambda_ordinal: float = 1.0
    lambda_boundary: float = 0.5
    
    boundary_uncertainty_alpha: float = 2.0
    boundary_disagreement_beta: float = 2.0


# ============================================================
# Trainer
# ============================================================

@dataclass
class TrainerConfig:
    epochs: int
    patience: int
    gradient_accumulation_steps: int = 1


# ============================================================
# Output
# ============================================================

@dataclass
class OutputConfig:
    save_dir: str


# ============================================================
# Logging
# ============================================================

@dataclass
class LoggingConfig:
    enabled: bool = True
    backend: str = "wandb"
    project: str = "dr-severity-grading"


# ============================================================
# Augmentation configs
# ============================================================

@dataclass
class RangeConfig:
    min: float
    max: float


@dataclass
class ProbabilityConfig:
    p: float


@dataclass
class RotationConfig:
    degrees: float
    p: float


@dataclass
class AffineConfig:
    degrees: float
    translate: float
    scale: RangeConfig
    shear: float
    p: float


@dataclass
class ColorJitterConfig:
    brightness: float
    contrast: float
    saturation: float
    hue: float
    p: float


@dataclass
class ResizedCropConfig:
    scale: RangeConfig
    ratio: RangeConfig
    p: float


@dataclass
class AugmentationConfig:
    name: str

    horizontal_flip: Optional[ProbabilityConfig] = None
    rotation: Optional[RotationConfig] = None
    affine: Optional[AffineConfig] = None
    color_jitter: Optional[ColorJitterConfig] = None
    resized_crop: Optional[ResizedCropConfig] = None


# ============================================================
# Main Config
# ============================================================

@dataclass
class Config:
    experiment_name: str
    seed: int
    device: str

    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    trainer: TrainerConfig
    output: OutputConfig
    logging: LoggingConfig
    augmentation: AugmentationConfig


# ============================================================
# Utilities
# ============================================================

def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


# ============================================================
# Config Loader
# ============================================================

def load_config(path: str) -> Config:

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    raw_aug = raw.get("augmentation", {})

    augmentation = AugmentationConfig(
        name=raw_aug.get("name", "none"),

        horizontal_flip=(
            ProbabilityConfig(**raw_aug["horizontal_flip"])
            if raw_aug.get("horizontal_flip") is not None
            else None
        ),

        rotation=(
            RotationConfig(**raw_aug["rotation"])
            if raw_aug.get("rotation") is not None
            else None
        ),

        affine=(
            AffineConfig(
                degrees=raw_aug["affine"]["degrees"],
                translate=raw_aug["affine"]["translate"],

                scale=RangeConfig(
                    **raw_aug["affine"]["scale"]
                ),

                shear=raw_aug["affine"]["shear"],
                p=raw_aug["affine"]["p"],
            )
            if raw_aug.get("affine") is not None
            else None
        ),

        color_jitter=(
            ColorJitterConfig(**raw_aug["color_jitter"])
            if raw_aug.get("color_jitter") is not None
            else None
        ),

        resized_crop=(
            ResizedCropConfig(
                scale=RangeConfig(
                    **raw_aug["resized_crop"]["scale"]
                ),

                ratio=RangeConfig(
                    **raw_aug["resized_crop"]["ratio"]
                ),

                p=raw_aug["resized_crop"]["p"],
            )
            if raw_aug.get("resized_crop") is not None
            else None
        ),
    )

    # --------------------------------------------------------
    # Main config
    # --------------------------------------------------------

    return Config(
        experiment_name=raw["experiment_name"],
        seed=raw["seed"],
        device=raw["device"],

        dataset=DatasetConfig(
            **raw["dataset"]
        ),

        dataloader=DataLoaderConfig(
            **raw["dataloader"]
        ),

        model=ModelConfig(
            **raw["model"]
        ),

        optimizer=OptimizerConfig(
            **raw["optimizer"]
        ),

        scheduler=SchedulerConfig(
            **raw["scheduler"]
        ),

        loss=LossConfig(
            **raw["loss"]
        ),

        trainer=TrainerConfig(
            **raw["trainer"]
        ),

        output=OutputConfig(
            **raw["output"]
        ),

        logging=LoggingConfig(
            **raw.get("logging", {})
        ),

        augmentation=augmentation,
    )