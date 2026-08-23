from torch.utils.data import DataLoader

from src.datasets.aptos_dataset import APTOSDataset

from src.augmentations.builder import (
    build_train_augmentation,
    build_valid_augmentation,
)

from src.datasets.sampler import (
    build_weighted_sampler,
    get_class_counts,
)


def get_train_dataloader(cfg):

    train_transform = build_train_augmentation(
        cfg
    )

    dataset = APTOSDataset(
        csv_file=cfg.dataset.train_csv,
        image_dir=cfg.dataset.train_dir,
        transform=train_transform,
    )

    sampler = None

    if (
        cfg.sampling.enabled
        and cfg.sampling.strategy
        == "weighted"
    ):
        sampler = build_weighted_sampler(
            dataset=dataset,
            alpha=cfg.sampling.alpha,
            seed=cfg.seed,
        )

        class_counts = get_class_counts(
            dataset
        )

        print(
            "Train class counts:",
            class_counts,
        )

        print(
            "Sampling strategy:",
            "soft weighted sampling",
        )

        print(
            "Sampling alpha:",
            cfg.sampling.alpha,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=(sampler is None),
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