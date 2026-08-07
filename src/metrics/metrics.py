from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
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