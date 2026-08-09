from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use('Agg')

def _save_figure(
    fig,
    output_path,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_confusion_matrix(
    labels,
    predictions,
    class_names,
    output_path,
):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        values_format="d",
        cmap="Blues",
    )

    ax.set_title(
        "Confusion Matrix"
    )

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
    )


def plot_roc_curve(
    labels,
    probabilities,
    class_names,
    output_path,
):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    num_classes = len(class_names)

    binary_labels = label_binarize(
        labels,
        classes=np.arange(num_classes),
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for class_index, class_name in enumerate(
        class_names
    ):
        fpr, tpr, _ = roc_curve(
            binary_labels[:, class_index],
            probabilities[:, class_index],
        )

        ax.plot(
            fpr,
            tpr,
            label=class_name,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curves"
    )

    ax.legend()

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
    )

def plot_pr_curve(
    labels,
    probabilities,
    class_names,
    output_path,
):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    num_classes = len(class_names)

    binary_labels = label_binarize(
        labels,
        classes=np.arange(num_classes),
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for class_index, class_name in enumerate(
        class_names
    ):
        precision, recall, _ = (
            precision_recall_curve(
                binary_labels[:, class_index],
                probabilities[:, class_index],
            )
        )

        ax.plot(
            recall,
            precision,
            label=class_name,
        )

    ax.set_xlabel(
        "Recall"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        "Precision-Recall Curves"
    )

    ax.legend()

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
    )

def plot_normalized_confusion_matrix(
    labels,
    predictions,
    class_names,
    output_path,
):
    matrix = confusion_matrix(
        labels,
        predictions,
        labels=np.arange(len(class_names)),
        normalize="true",
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        values_format=".2f",
        cmap="Greens"
    )

    ax.set_title(
        "Normalized Confusion Matrix"
    )

    fig.tight_layout()

    _save_figure(
        fig,
        output_path,
    )