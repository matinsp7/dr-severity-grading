import argparse
import shutil

import torch
import torch.nn as nn

from src.callbacks.checkpoint import CheckpointManager
from src.callbacks.config_backup import backup_config
from src.callbacks.git import save_git_hash
from src.callbacks.logger import MetricLogger
from src.datasets.dataloader import (
    get_train_dataloader,
    get_valid_dataloader,
)
from src.models.builder import build_model
from src.trainer.trainer import Trainer
from src.utils.config import load_config
from src.utils.paths import ExperimentPaths
from src.callbacks.early_stopping import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        "cuda"
        if cfg.device == "cuda"
        and torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    paths = ExperimentPaths(
        cfg.experiment_name
    )

    backup_config(
        args.config,
        paths.config_copy,
    )

    save_git_hash(
        paths.git_commit,
    )

    train_loader = get_train_dataloader(
        cfg
    )

    valid_loader = get_valid_dataloader(
        cfg
    )

    model = build_model(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    criterion = nn.CrossEntropyLoss()

    logger = MetricLogger(
        paths.metrics_csv
    )

    checkpoint = CheckpointManager(
        paths
    )

    early_stopping = EarlyStopping(
        patience=cfg.trainer.patience,
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=criterion,
        checkpoint=checkpoint,
        logger=logger,
        paths=paths,
        device=device,
        early_stopping=early_stopping,
        resume=args.resume,
    )

    trainer.fit()

    print()

    print("Training Finished!")

    print(paths.root)


if __name__ == "__main__":
    main()