from dataclasses import dataclass
from dataclasses import asdict
import yaml


@dataclass
class DatasetConfig:
    train_csv: str
    valid_csv: str

    train_dir: str
    valid_dir: str

    image_size: int

    class_names: list[str]


@dataclass
class DataLoaderConfig:
    batch_size: int
    num_workers: int
    pin_memory: bool


@dataclass
class ModelConfig:
    name: str
    pretrained: bool
    num_classes: int


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


@dataclass
class TrainerConfig:
    epochs: int


@dataclass
class OutputConfig:
    save_dir: str


@dataclass
class LoggingConfig:
    enabled: bool = True
    backend: str = "wandb"
    project: str = "dr-severity-grading"


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

def config_to_dict(cfg):
    return asdict(cfg)

def load_config(path: str) -> Config:

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        experiment_name=raw["experiment_name"],
        seed=raw["seed"],
        device=raw["device"],

        dataset=DatasetConfig(**raw["dataset"]),
        dataloader=DataLoaderConfig(**raw["dataloader"]),
        model=ModelConfig(**raw["model"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
        scheduler=SchedulerConfig(**raw["scheduler"]),
        loss=LossConfig(**raw["loss"]),
        trainer=TrainerConfig(**raw["trainer"]),
        output=OutputConfig(**raw["output"]),
        logging=LoggingConfig( **raw["logging"]),
    )