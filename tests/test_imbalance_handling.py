import torch

from src.utils.config import load_config
from src.datasets.dataloader import (
    get_train_dataloader,
    get_valid_dataloader,
)
from src.imbalance.class_weights import (
    compute_class_weights,
)
from src.imbalance.sampler import (
    build_weighted_sampler,
)


def print_original_distribution(labels, num_classes):

    print("\nOriginal training distribution")
    print("-" * 40)

    total = len(labels)

    for class_id in range(num_classes):

        count = (labels == class_id).sum()

        percentage = (
            count / total
        ) * 100

        print(
            f"Class {class_id}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )


def test_class_weights(
    labels,
    num_classes,
):

    print("\nClass weights")
    print("-" * 40)

    weights = compute_class_weights(
        labels=labels,
        num_classes=num_classes,
    )

    for class_id, weight in enumerate(weights):

        print(
            f"Class {class_id}: "
            f"{weight.item():.6f}"
        )

    return weights


def test_weighted_sampler(
    labels,
    num_classes,
    num_samples=None,
):

    print("\nWeightedRandomSampler distribution")
    print("-" * 40)

    sampler = build_weighted_sampler(
        labels=labels,
        num_samples=num_samples,
    )

    sampled_indices = list(iter(sampler))

    sampled_labels = labels[
        sampled_indices
    ]

    total = len(sampled_labels)

    for class_id in range(num_classes):

        count = (
            sampled_labels == class_id
        ).sum()

        percentage = (
            count / total
        ) * 100

        print(
            f"Class {class_id}: "
            f"{count:6d} "
            f"({percentage:6.2f}%)"
        )


def test_train_loader(
    cfg,
):

    print("\nTrain DataLoader")
    print("-" * 40)

    loader = get_train_dataloader(cfg)

    print(
        f"Dataset size : "
        f"{len(loader.dataset)}"
    )

    print(
        f"Batch size   : "
        f"{loader.batch_size}"
    )

    print(
        f"Num batches  : "
        f"{len(loader)}"
    )

    images, labels = next(
        iter(loader)
    )

    print(
        f"Images shape : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Labels shape : "
        f"{tuple(labels.shape)}"
    )

    print(
        f"Labels dtype : "
        f"{labels.dtype}"
    )

    print("✓ Train DataLoader works.")


def test_valid_loader(
    cfg,
):

    print("\nValidation DataLoader")
    print("-" * 40)

    loader = get_valid_dataloader(cfg)

    images, labels = next(
        iter(loader)
    )

    print(
        f"Images shape : "
        f"{tuple(images.shape)}"
    )

    print(
        f"Labels shape : "
        f"{tuple(labels.shape)}"
    )

    print("✓ Validation DataLoader works.")


def main():

    config_path =  "../configs/exp007_imbalance_handling.yaml"

    cfg = load_config(
        config_path
    )

    print("=" * 50)
    print("IMBALANCE PIPELINE TEST")
    print("=" * 50)

    train_loader = get_train_dataloader(
        cfg
    )

    labels = (
        train_loader
        .dataset
        .labels
    )

    labels = torch.as_tensor(
        labels
    )

    num_classes = (
        cfg.model.num_classes
    )

    print_original_distribution(
        labels,
        num_classes,
    )

    test_class_weights(
        labels,
        num_classes,
    )

    test_weighted_sampler(
        labels,
        num_classes,
    )

    test_train_loader(
        cfg
    )

    test_valid_loader(
        cfg
    )

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()