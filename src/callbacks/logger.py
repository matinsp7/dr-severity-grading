import csv
from pathlib import Path
from src.utils.config import config_to_dict

class ExperimentLogger:

    def log(self, metrics):
        raise NotImplementedError

    def log_artifact(self, name, artifact_type, files, metadata=None):
        raise NotImplementedError

    def log_image(self, name, image_path):
        raise NotImplementedError

    def log_summary(self, metrics):
        raise NotImplementedError

    def finish(self):
        raise NotImplementedError


class MetricLogger:

    HEADER = [
        "epoch",
        "train_loss",
        "val_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "qwk",
    ]

    def __init__(self, csv_path):

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():

            with open(
                self.csv_path,
                "w",
                newline="",
            ) as f:

                writer = csv.writer(f)

                writer.writerow(self.HEADER)

    def log(self, metrics):

        with open(
            self.csv_path,
            "a",
            newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(

                [
                    metrics["epoch"],
                    metrics["train_loss"],
                    metrics["val_loss"],
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["qwk"],
                ]

            )

    def log_artifact(self, name, artifact_type, files, metadata=None):
        pass

    def log_image(self, name, image_path):
        pass

    def log_summary(self, metrics):

        summary_path = (
                self.csv_path.parent
                / "evaluation_summary.csv"
        )

        with open(
                summary_path,
                "w",
                newline="",
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["metric", "value"]
            )

            for name, value in metrics.items():

                if isinstance(value, (int, float)):
                    writer.writerow(
                        [name, value]
                    )

    def finish(self):
        pass

class CombinedLogger(ExperimentLogger):

    def __init__(self, loggers):
        self.loggers = loggers

    def log(self, metrics):
        for logger in self.loggers:
            logger.log(metrics)

    def finish(self):
        for logger in self.loggers:
            logger.finish()

    def log_artifact(self, name, artifact_type, files, metadata=None):
        for logger in self.loggers:
            logger.log_artifact(
                name=name,
                artifact_type=artifact_type,
                files=files,
                metadata=metadata,
            )

    def log_image(self, name, image_path):
        for logger in self.loggers:
            logger.log_image(name, image_path)

    def log_summary(self, metrics):
        for logger in self.loggers:
            logger.log_summary(metrics)


def build_logger(cfg, paths):
    loggers = [MetricLogger(paths.metrics_csv)]

    if cfg.logging.enabled:

        if cfg.logging.backend == "wandb":

            from src.callbacks.wandb_logger import WandBLogger
            import wandb

            run_id_path = paths.root / "wandb_run_id.txt"

            if run_id_path.exists():
                with open(run_id_path, "r") as f:
                    run_id = f.read().strip()
            else:
                run_id = wandb.util.generate_id()
                with open(run_id_path, "w") as f:
                    f.write(run_id)
            # ---------------------------------------------

            loggers.append(
                WandBLogger(
                    project=cfg.logging.project,
                    experiment_name=cfg.experiment_name,
                    config=config_to_dict(cfg),
                    run_id=run_id,
                    resume="allow",
                )
            )

        else:
            raise ValueError(
                f"Unknown logging backend: "
                f"{cfg.logging.backend}"
            )

    return CombinedLogger(loggers)