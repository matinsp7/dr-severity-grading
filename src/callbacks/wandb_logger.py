import wandb

class WandBLogger:

    def __init__(
        self,
        project,
        experiment_name,
        config,
        run_id=None,
        resume=None,
    ):
        self.run = wandb.init(
            project=project,
            name=experiment_name,
            config=config,
            id=run_id,
            resume=resume,
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

    def log_artifact(self, name, artifact_type, files, metadata=None):

        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata or {},
        )

        for file_path in files:
            artifact.add_file(
                str(file_path)
            )

        self.run.log_artifact(
            artifact
        )

    def log_image(self, name, image_path):
        self.run.log(
            {
                name: wandb.Image(str(image_path))
            }
        )

    def log_summary(self, metrics):
        for key, value in metrics.items():
            if key == "val_loss":
                self.run.summary["test/loss"] = value
            else:
                self.run.summary[f"test/{key}"] = value

    def finish(self):
        self.run.finish()