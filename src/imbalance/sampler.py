import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_weighted_sampler(
    labels,
    num_samples=None,
):
    labels = np.asarray(labels)

    num_classes = int(labels.max()) + 1

    class_counts = np.bincount(
        labels,
        minlength=num_classes,
    )

    if np.any(class_counts == 0):
        missing_classes = np.where(
            class_counts == 0
        )[0]

        raise ValueError(
            "Cannot build weighted sampler because "
            f"these classes have no samples: "
            f"{missing_classes.tolist()}"
        )

    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]

    sample_weights = torch.as_tensor(
        sample_weights,
        dtype=torch.double,
    )

    if num_samples is None:
        num_samples = len(sample_weights)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )