import argparse

import torch
import torch.nn as nn
from src.callbacks.logger import build_logger

from src.callbacks.checkpoint import CheckpointManager
from src.datasets.dataloader import get_test_dataloader
from src.evaluation.evaluator import Evaluator
from src.models.builder import build_model
from src.utils.config import load_config
from src.utils.paths import ExperimentPaths
from sklearn.metrics import classification_report
from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(
        args.config
    )

    device = torch.device(
        "cuda"
        if cfg.device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    paths = ExperimentPaths(
        cfg.experiment_name
    )

    test_loader = get_test_dataloader(
        cfg
    )

    model = build_model(
        cfg
    ).to(device)

    checkpoint = CheckpointManager(
        paths
    )

    checkpoint.load_best(
        model=model,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    evaluator = Evaluator(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    results = evaluator.evaluate()

    plot_confusion_matrix(
        labels=results["labels"],
        predictions=results["predictions"],
        class_names=cfg.dataset.class_names,
        output_path=paths.confusion_matrix,
    )

    plot_normalized_confusion_matrix(
        labels=results["labels"],
        predictions=results["predictions"],
        class_names=cfg.dataset.class_names,
        output_path=paths.normalized_confusion_matrix,
    )

    plot_roc_curve(
        labels=results["labels"],
        probabilities=results["probabilities"],
        class_names=cfg.dataset.class_names,
        output_path=paths.roc_curve,
    )

    plot_pr_curve(
        labels=results["labels"],
        probabilities=results["probabilities"],
        class_names=cfg.dataset.class_names,
        output_path=paths.pr_curve,
    )

    logger = build_logger(cfg=cfg, paths=paths)

    logger.log_image("evaluation/confusion_matrix", paths.confusion_matrix)                         #w&b
    logger.log_image("evaluation/normalized_confusion_matrix", paths.normalized_confusion_matrix)   #w&b
    logger.log_image("evaluation/roc_curve", paths.roc_curve)                                       #w&b
    logger.log_image("evaluation/pr_curve", paths.pr_curve)                                         #w&b

    logger.log_summary(results["metrics"])

    logger.finish()


    print()
    print("=" * 45)
    print("Evaluation")
    print("=" * 45)

    print(
        f"Checkpoint  : {paths.best_model}"
    )

    print(
        f"Samples     : {len(results['labels'])}"
    )

    print(f"\n{30*'-'}\nClassification Report\n{classification_report(results["labels"], results["predictions"])}" )

    print()
    print("Metrics")
    print("-" * 45)

    metrics = results["metrics"]


    print(
        f"Val Loss    : {metrics['val_loss']:.4f}"
    )

    print(
        f"Accuracy    : {metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : {metrics['precision']:.4f}"
    )

    print(
        f"Recall      : {metrics['recall']:.4f}"
    )

    print(
        f"F1          : {metrics['f1']:.4f}"
    )

    print(
        f"QWK         : {metrics['qwk']:.4f}"
    )

    print(
        f"ROC-AUC     : {metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC      : {metrics['pr_auc']:.4f}"
    )

    print()
    print("=" * 45)
    print("Evaluation Finished!")
    print("=" * 45)


if __name__ == "__main__":
    main()