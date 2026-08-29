from torch.utils.data import DataLoader

from src.datasets.aptos_dataset import APTOSDataset
from src.augmentations.builder import (
    build_train_augmentation,
    build_valid_augmentation,
)

from src.imbalance.sampler import (
    build_weighted_sampler,
)

def get_train_dataloader(cfg):

    train_dataset = APTOSDataset(
        csv_file=cfg.dataset.train_csv,
        image_dir=cfg.dataset.train_dir,
        transform=build_train_augmentation(cfg),
    )

    sampler = None

    if cfg.imbalance.use_weighted_sampler:
        sampler = build_weighted_sampler(
            train_dataset.labels
        )

    return DataLoader(
        dataset=train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )


def get_valid_dataloader(cfg):

    valid_transform = build_valid_augmentation(
        cfg
    )

    dataset = APTOSDataset(
        csv_file=cfg.dataset.valid_csv,
        image_dir=cfg.dataset.valid_dir,
        transform=valid_transform,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )

def get_test_dataloader(cfg):

    valid_transform = build_valid_augmentation(
        cfg
    )

    dataset = APTOSDataset(
        csv_file=cfg.dataset.test_csv,
        image_dir=cfg.dataset.test_dir,
        transform=valid_transform,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
    )