from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
)


def compute_metrics(labels, predictions):

    return {

        "accuracy": accuracy_score(labels, predictions),

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


def save_confusion_matrix(
    labels,
    predictions,
    save_path,
):

    cm = confusion_matrix(labels, predictions)

    disp = ConfusionMatrixDisplay(cm)

    disp.plot(values_format="d")

    Path(save_path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(save_path)

    plt.close()