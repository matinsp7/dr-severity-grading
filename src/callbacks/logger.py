import csv
from pathlib import Path


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