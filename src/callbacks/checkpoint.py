import torch


class CheckpointManager:

    def __init__(self, paths):

        self.paths = paths

        self.best_qwk = -1

    def save_last(
        self,
        model,
        optimizer,
        epoch,
        patience_counter,
    ):
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_qwk": self.best_qwk,
            "patience_counter": patience_counter,
        }

        torch.save(
            checkpoint,
            self.paths.last_model,
        )

    def load_best(self, model, device):
        checkpoint_path = self.paths.best_model

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Best checkpoint not found: {checkpoint_path}"
            )

        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
        )

        model.load_state_dict(state_dict)

        return model

    def update_best(self, model, metrics):

        if metrics["qwk"] <= self.best_qwk:
            return False

        self.best_qwk = metrics["qwk"]

        torch.save(
            model.state_dict(),
            self.paths.best_model,
        )

        return True