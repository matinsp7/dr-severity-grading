from pathlib import Path


class ExperimentPaths:

    def __init__(self, experiment_name: str):

        self.root = Path("outputs") / experiment_name

        self.checkpoints = self.root / "checkpoints"

        self.logs = self.root / "logs"

        self.figures = self.root / "figures"

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoints.mkdir(exist_ok=True)

        self.logs.mkdir(exist_ok=True)

        self.figures.mkdir(exist_ok=True)

    @property
    def best_model(self):

        return self.checkpoints / "best_model.pth"

    @property
    def last_model(self):

        return self.checkpoints / "last_model.pth"

    @property
    def metrics_csv(self):

        return self.root / "metrics.csv"

    @property
    def config_copy(self):

        return self.root / "config.yaml"

    @property
    def git_commit(self):

        return self.root / "git_commit.txt"

    @property
    def confusion_matrix(self):

        return self.figures / "confusion_matrix.png"

    @property
    def loss_curve(self):

        return self.figures / "loss_curve.png"

    @property
    def accuracy_curve(self):

        return self.figures / "accuracy_curve.png"

    @property
    def qwk_curve(self):

        return self.figures / "qwk_curve.png"