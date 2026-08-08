import wandb

class WandBLogger:

    def __init__(
        self,
        project,
        experiment_name,
        config,
    ):
        self.run = wandb.init(
            project=project,
            name=experiment_name,
            config=config,
        )

    def log(self, metrics):
        epoch = metrics["epoch"]

        self.run.log(
            {
                "train/loss": metrics["train_loss"],
                "val/loss": metrics["val_loss"],
                "val/accuracy": metrics["accuracy"],
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "val/f1": metrics["f1"],
                "val/qwk": metrics["qwk"],
            },
            step=epoch,
        )

    def finish(self):
        self.run.finish()