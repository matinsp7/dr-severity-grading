from collections import Counter

import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(
    dataset,
    alpha: float = 0.5,
    seed: int = 42,
):
    """
    Build a soft inverse-frequency sampler.

    Class weight:
        w_c = (1 / n_c) ** alpha

    With alpha=0.5, the imbalance correction is softer than
    standard inverse-frequency weighting.
    """

    labels = dataset.data["diagnosis"].astype(int).tolist()

    class_counts = Counter(labels)

    class_weights = {
        class_id: (1.0 / count) ** alpha
        for class_id, count in class_counts.items()
    }

    sample_weights = [
        class_weights[label]
        for label in labels
    ]

    weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double,
    )

    generator = torch.Generator()

    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )

    return sampler


def get_class_counts(dataset):
    labels = dataset.data["diagnosis"].astype(int).tolist()

    return dict(
        sorted(
            Counter(labels).items()
        )
    )