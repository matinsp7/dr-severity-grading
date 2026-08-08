import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
)


def compute_metrics(labels, predictions):

    return {

        "accuracy": accuracy_score(
            labels,
            predictions,
        ),

        "precision": precision_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "recall": recall_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "f1": f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),

        "qwk": cohen_kappa_score(
            labels,
            predictions,
            weights="quadratic",
        ),

    }

def compute_auc_metrics(
    labels,
    probabilities,
):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    num_classes = probabilities.shape[1]

    one_hot_labels = np.eye(
        num_classes
    )[labels]

    roc_auc = roc_auc_score(
        one_hot_labels,
        probabilities,
        average="macro",
    )

    pr_auc = average_precision_score(
        one_hot_labels,
        probabilities,
        average="macro",
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }