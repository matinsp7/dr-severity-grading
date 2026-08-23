import torch

from src.datasets.aptos_dataset import APTOSDataset
from src.datasets.sampler import (
    build_weighted_sampler,
    get_class_counts,
)


def main():

    dataset = APTOSDataset(
        csv_file="data/train_1.csv",
        image_dir="data/train_images",
        transform=None,
    )

    counts = get_class_counts(
        dataset
    )

    print("Original class counts:")

    for class_id, count in counts.items():
        print(
            f"Class {class_id}: {count}"
        )

    sampler = build_weighted_sampler(
        dataset=dataset,
        alpha=0.5,
        seed=42,
    )

    sampled_indices = list(
        iter(sampler)
    )

    sampled_labels = [
        int(
            dataset.data.iloc[index][
                "diagnosis"
            ]
        )
        for index in sampled_indices
    ]

    sampled_counts = torch.bincount(
        torch.tensor(sampled_labels),
        minlength=5,
    )

    print()
    print("Sampled class counts:")

    for class_id, count in enumerate(
        sampled_counts.tolist()
    ):
        print(
            f"Class {class_id}: {count}"
        )

    assert len(sampled_indices) == len(
        dataset
    )

    assert sum(
        sampled_counts
    ).item() == len(dataset)

    print()
    print("Weighted sampler test: OK")


if __name__ == "__main__":
    main()