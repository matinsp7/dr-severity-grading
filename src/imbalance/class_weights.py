import numpy as np
import torch


def compute_class_weights(
    labels,
    num_classes,
):
    labels = np.asarray(labels)

    counts = np.bincount(
        labels,
        minlength=num_classes,
    )

    if np.any(counts == 0):
        missing_classes = np.where(
            counts == 0
        )[0]

        raise ValueError(
            "Cannot compute class weights because "
            f"these classes have no samples: "
            f"{missing_classes.tolist()}"
        )

    total_samples = len(labels)

    weights = total_samples / (
        num_classes * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )