import torch

from src.metrics.metrics import compute_metrics


class Trainer:

    def __init__(
        self,
        cfg,
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        checkpoint,
        logger,
        paths,
        device,
        early_stopping,
        resume=False,
    ):

        self.cfg = cfg

        self.model = model

        self.train_loader = train_loader

        self.valid_loader = valid_loader

        self.optimizer = optimizer

        self.criterion = criterion

        self.checkpoint = checkpoint

        self.logger = logger

        self.paths = paths

        self.device = device

        self.resume_training = resume

        self.early_stopping = early_stopping

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0

        for images, labels in self.train_loader:

            images = images.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(
            self.train_loader
        )

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0

        predictions = []

        labels_list = []

        for images, labels in self.valid_loader:

            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            outputs = self.model(
                images
            )

            loss = self.criterion(
                outputs,
                labels,
            )

            running_loss += loss.item()

            preds = outputs.argmax(
                dim=1
            )

            predictions.extend(
                preds.cpu().tolist()
            )

            labels_list.extend(
                labels.cpu().tolist()
            )

        metrics = compute_metrics(
            labels_list,
            predictions,
        )

        metrics["val_loss"] = (

            running_loss
            / len(self.valid_loader)

        )

        metrics["labels"] = labels_list

        metrics["predictions"] = predictions

        return metrics

    def fit(self):

        start_epoch = 0

        if self.resume_training:
            start_epoch = self.resume()

        for epoch in range(
            start_epoch,
            self.cfg.trainer.epochs,
        ):

            train_loss = (
                self.train_one_epoch()
            )

            val_metrics = (
                self.validate()
            )

            metrics = {

                "epoch": epoch + 1,

                "train_loss": train_loss,

                **val_metrics,

            }

            self.logger.log(metrics)

            is_best = (

                self.checkpoint.update_best(

                    self.model,

                    metrics,

                )

            )

            should_stop = self.early_stopping.step(
                is_best
            )

            self.checkpoint.save_last(
                self.model,
                self.optimizer,
                epoch ,
                self.early_stopping.counter,
            )

            self.print_metrics(

                metrics,

                is_best,

            )

            if should_stop:
                print(
                    "EarlyStopping: "
                    "QWK did not improve for "
                    f"{self.early_stopping.patience} epochs."
                )
                break

    def print_metrics(

        self,

        metrics,

        is_best,

    ):

        print()

        print("=" * 45)

        print(
            f"Epoch {metrics['epoch']}/"
            f"{self.cfg.trainer.epochs}"
        )

        print("=" * 45)

        print(
            f"Train Loss : "
            f"{metrics['train_loss']:.4f}"
        )

        print(
            f"Val Loss   : "
            f"{metrics['val_loss']:.4f}"
        )

        print(
            f"Accuracy   : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Precision  : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall     : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1         : "
            f"{metrics['f1']:.4f}"
        )

        print(
            f"QWK        : "
            f"{metrics['qwk']:.4f}"
        )

        if is_best:

            print()

            print(
                "✓ Best model updated."
            )

        print()

    def resume(self):
        if not self.paths.last_model.exists():
            print(
                "No checkpoint found! "
                "Starting training from scratch."
            )
            return 0

        checkpoint = torch.load(
            self.paths.last_model,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.checkpoint.best_qwk = (
            checkpoint["best_qwk"]
        )

        self.early_stopping.counter = (
            checkpoint["patience_counter"]
        )

        start_epoch = checkpoint["epoch"]+1

        print("Resume Training")
        print(
            f"Resuming from epoch {start_epoch}"
        )

        return start_epoch