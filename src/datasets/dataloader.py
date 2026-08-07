from torch.utils.data import DataLoader

from src.datasets.aptos_dataset import APTOSDataset
from src.augmentations.transforms import (
    get_train_transform,
    get_valid_transform,
)


def get_train_dataloader(cfg):
    dataset = APTOSDataset(
        csv_file=cfg.dataset.train_csv,
        image_dir=cfg.dataset.train_dir,
        transform=get_train_transform(),
    )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )


def get_valid_dataloader(cfg):
    dataset = APTOSDataset(
        csv_file=cfg.dataset.valid_csv,
        image_dir=cfg.dataset.valid_dir,
        transform=get_valid_transform(),
    )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )