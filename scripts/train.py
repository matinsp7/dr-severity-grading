import argparse
import shutil

import torch
import torch.nn as nn
import random
import numpy as np

from src.callbacks.checkpoint import CheckpointManager
from src.callbacks.config_backup import backup_config
from src.callbacks.git import save_git_hash
from src.callbacks.logger import build_logger
from src.datasets.dataloader import (
    get_train_dataloader,
    get_valid_dataloader,
)
from src.models.builder import build_model
from src.trainer.trainer import Trainer
from src.utils.config import load_config
from src.utils.paths import ExperimentPaths


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    return parser.parse_args()

def set_seed(seed):
    """
    Sets random seed for Python, NumPy, and PyTorch.
    CRITICAL for reproducibility - without this, you'll get
    different results every time you run the same code,
    making debugging and comparison impossible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():

    args = parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

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

    logger = build_logger(
        cfg=cfg,
        paths=paths,
    )

    checkpoint = CheckpointManager(
        paths
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
    )

    trainer.fit()

    logger.log_artifact(
        name=f"{cfg.experiment_name}-checkpoints",
        artifact_type="model",
        files=[
            paths.best_model,
            paths.last_model,
        ],
        metadata={
            "experiment_name": cfg.experiment_name,
            "model": cfg.model.name,
            "num_classes": cfg.model.num_classes,
            "best_qwk": checkpoint.best_qwk,
        },
    )

    logger.finish()

    print()
    print("Training Finished!")

    print(paths.root)

if __name__ == "__main__":
    main()