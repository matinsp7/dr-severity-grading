from dataclasses import asdict, dataclass

import yaml


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


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class InferenceConfig:
    mode: str = "argmax"

    fusion_lambda: float = 0.7

    thresholds: list[float] | None = None


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


@dataclass
class OptimizerConfig:
    name: str
    lr: float
    weight_decay: float


@dataclass
class SchedulerConfig:
    name: str
    t_max: int


@dataclass
class LossConfig:
    name: str
    gamma: float = 2.0

    lambda_ordinal: float = 1.0
    lambda_boundary: float = 0.5

    boundary_uncertainty_alpha: float = 2.0
    boundary_disagreement_beta: float = 2.0


@dataclass
class TrainerConfig:
    epochs: int
    patience: int
    gradient_accumulation_steps: int = 1


@dataclass
class OutputConfig:
    save_dir: str


@dataclass
class LoggingConfig:
    enabled: bool = True
    backend: str = "wandb"
    project: str = "dr-severity-grading"


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

    horizontal_flip: ProbabilityConfig

    rotation: RotationConfig

    affine: AffineConfig

    color_jitter: ColorJitterConfig

    resized_crop: ResizedCropConfig


@dataclass
class Config:
    experiment_name: str
    seed: int
    device: str

    dataset: DatasetConfig
    dataloader: DataLoaderConfig
    inference: InferenceConfig

    model: ModelConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    trainer: TrainerConfig

    output: OutputConfig
    logging: LoggingConfig
    augmentation: AugmentationConfig


def config_to_dict(cfg):
    return asdict(cfg)


def load_config(path: str) -> Config:

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

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

        inference=InferenceConfig(
            **raw.get(
                "inference",
                {
                    "mode": "argmax",
                    "fusion_lambda": 0.7,
                    "thresholds": [
                        0.5,
                        1.5,
                        2.5,
                        3.5,
                    ],
                },
            )
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
            **raw["logging"]
        ),

        augmentation=AugmentationConfig(
            name=raw["augmentation"]["name"],

            horizontal_flip=ProbabilityConfig(
                **raw["augmentation"]["horizontal_flip"]
            ),

            rotation=RotationConfig(
                **raw["augmentation"]["rotation"]
            ),

            affine=AffineConfig(
                degrees=raw["augmentation"]["affine"]["degrees"],

                translate=raw["augmentation"]["affine"]["translate"],

                scale=RangeConfig(
                    **raw["augmentation"]["affine"]["scale"]
                ),

                shear=raw["augmentation"]["affine"]["shear"],

                p=raw["augmentation"]["affine"]["p"],
            ),

            color_jitter=ColorJitterConfig(
                **raw["augmentation"]["color_jitter"]
            ),

            resized_crop=ResizedCropConfig(
                scale=RangeConfig(
                    **raw["augmentation"]["resized_crop"]["scale"]
                ),

                ratio=RangeConfig(
                    **raw["augmentation"]["resized_crop"]["ratio"]
                ),

                p=raw["augmentation"]["resized_crop"]["p"],
            ),
        ),
    )