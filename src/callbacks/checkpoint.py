import torch


class CheckpointManager:

    def __init__(self, paths):

        self.paths = paths

        self.best_qwk = -1

    def save_last(self, model):

        torch.save(
            model.state_dict(),
            self.paths.last_model,
        )

    def update_best(
        self,
        model,
        metrics,
    ):

        if metrics["qwk"] <= self.best_qwk:
            return False

        self.best_qwk = metrics["qwk"]

        torch.save(
            model.state_dict(),
            self.paths.best_model,
        )

        return True