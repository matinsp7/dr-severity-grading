import argparse

import torch

from sklearn.metrics import classification_report

from src.callbacks.checkpoint import (
    CheckpointManager,
)

from src.callbacks.logger import (
    build_logger,
)

from src.datasets.dataloader import (
    get_test_dataloader,
)

from src.evaluation.evaluator import (
    Evaluator,
)

from src.evaluation.plots import (
    plot_confusion_matrix,
    plot_normalized_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)

from src.loss.builder import build_loss

from src.models.builder import build_model

from src.utils.config import load_config

from src.utils.paths import ExperimentPaths


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

    test_loader = (
        get_test_dataloader(cfg)
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

    criterion = build_loss(
        cfg
    )

    evaluator = Evaluator(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        inference_mode=(
            cfg.inference.mode
        ),
        fusion_lambda=(
            cfg.inference.fusion_lambda
        ),
        thresholds=(
            cfg.inference.thresholds
        ),
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
        output_path=(
            paths.normalized_confusion_matrix
        ),
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

    logger = build_logger(
        cfg=cfg,
        paths=paths,
    )

    logger.log_image(
        "test/confusion_matrix",
        paths.confusion_matrix,
    )

    logger.log_image(
        "test/normalized_confusion_matrix",
        paths.normalized_confusion_matrix,
    )

    logger.log_image(
        "test/roc_curve",
        paths.roc_curve,
    )

    logger.log_image(
        "test/pr_curve",
        paths.pr_curve,
    )

    logger.log_summary(
        results["metrics"]
    )

    logger.finish()

    print()

    print("=" * 45)
    print("Evaluation")
    print("=" * 45)

    print(
        f"Checkpoint  : "
        f"{paths.best_model}"
    )

    print(
        f"Samples     : "
        f"{len(results['labels'])}"
    )

    print(
        f"Inference   : "
        f"{cfg.inference.mode}"
    )

    if cfg.inference.mode == "fused":

        print(
            f"Fusion λ    : "
            f"{cfg.inference.fusion_lambda}"
        )

        print(
            "Thresholds  : "
            f"{cfg.inference.thresholds}"
        )

    print()

    print(
        "-" * 30
    )

    print(
        "Classification Report"
    )

    print(
        classification_report(
            results["labels"],
            results["predictions"],
            zero_division=0,
        )
    )

    print()

    print("Metrics")
    print("-" * 45)

    metrics = results["metrics"]

    print(
        f"Test Loss    : "
        f"{metrics['val_loss']:.4f}"
    )

    print(
        f"Accuracy    : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision   : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall      : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1          : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"QWK         : "
        f"{metrics['qwk']:.4f}"
    )

    print(
        f"ROC-AUC     : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC      : "
        f"{metrics['pr_auc']:.4f}"
    )

    print()

    print("=" * 45)
    print("Evaluation Finished!")
    print("=" * 45)


if __name__ == "__main__":
    main()