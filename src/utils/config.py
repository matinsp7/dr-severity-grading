from dataclasses import dataclass, field
from dataclasses import asdict
from typing import Any, Optional
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
    gamma: float

#=============================================================
# Imbalance handling
#=============================================================

@dataclass
class ImbalanceConfig:
    use_class_weights: bool = False
    use_weighted_sampler: bool = False

# ============================================================
# Trainer
# ============================================================

@dataclass
class TrainerConfig:
    epochs: int
    patience: int


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
class HorizontalFlipConfig:
    p: float


@dataclass
class RotationConfig:
    degrees: float
    p: float


@dataclass
class AffineConfig:
    translate: float
    scale: RangeConfig
    rotate: float
    p: float


@dataclass
class ColorJitterConfig:
    brightness: float
    contrast: float
    saturation: float
    hue: float
    value: float
    p: float
    hue_p: float

@dataclass
class  GaussNoiseConfig:
    min: float
    max: float
    p: float


@dataclass
class ClaheConfig:
    clip_limit: float
    p: float


@dataclass
class AugmentationConfig:
    name: str

    horizontal_flip: Optional[HorizontalFlipConfig] = None
    rotation: Optional[RotationConfig] = None
    affine: Optional[AffineConfig] = None
    color_jitter: Optional[ColorJitterConfig] = None
    gauss_noise: Optional[GaussNoiseConfig] = None
    clahe: Optional[ClaheConfig] = None


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
    imbalance: ImbalanceConfig


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
            HorizontalFlipConfig(
                **raw_aug["horizontal_flip"]
            )
            if raw_aug.get("horizontal_flip") is not None
            else None
        ),

        rotation=(
            RotationConfig(
                **raw_aug["rotation"]
            )
            if raw_aug.get("rotation") is not None
            else None
        ),

        affine=(
            AffineConfig(
                translate=raw_aug["affine"]["translate"],
                scale=RangeConfig(
                    **raw_aug["affine"]["scale"]
                ),
                rotate=raw_aug["affine"]["rotate"],
                p=raw_aug["affine"]["p"],
            )
            if raw_aug.get("affine") is not None
            else None
        ),

        color_jitter=(
            ColorJitterConfig(
                brightness=raw_aug["color_jitter"]["brightness"],
                contrast=raw_aug["color_jitter"]["contrast"],
                saturation=raw_aug["color_jitter"]["saturation"],
                hue=raw_aug["color_jitter"]["hue"],
                value=raw_aug["color_jitter"]["value"],
                p=raw_aug["color_jitter"]["p"],
                hue_p=raw_aug["color_jitter"]["hue_p"],
            )
            if raw_aug.get("color_jitter") is not None
            else None
        ),

        gauss_noise=(
            GaussNoiseConfig(
                **raw_aug["gauss_noise"]
            )
            if raw_aug.get("gauss_noise") is not None
            else None
        ),

        clahe=(
            ClaheConfig(
                **raw_aug["clahe"]
            )
            if raw_aug.get("clahe") is not None
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

        imbalance=ImbalanceConfig(
            **raw.get("imbalance", {})
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


