import torch
from tqdm import tqdm

from src.metrics.metrics import compute_metrics


class Trainer:

    def __init__(
        self,
        cfg,
        model,
        train_loader,
        valid_loader,
        optimizer,
        scheduler,
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

        self.scheduler = scheduler

        self.criterion = criterion

        self.checkpoint = checkpoint

        self.logger = logger

        self.paths = paths

        self.device = device

        self.resume_training = resume

        self.early_stopping = early_stopping

        self.use_amp = (self.cfg.device == "cuda" and torch.cuda.is_available())

        self.amp_dtype = torch.float16

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        accumulation_steps = (
            self.cfg.trainer.gradient_accumulation_steps
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        for step, (images, labels) in enumerate(
            tqdm(
                self.train_loader,
                "Training",
                leave=False,
            )
        ):

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):

                outputs = self.model(images)

                loss = self.criterion(
                    outputs,
                    labels,
                )

                loss = (
                    loss
                    / accumulation_steps
                )

            self.scaler.scale(
                loss
            ).backward()

            should_step = (
                (step + 1) % accumulation_steps == 0
                or (step + 1) == len(self.train_loader)
            )

            if should_step:

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += (
                loss.item()
                * accumulation_steps
            )

        return (
            running_loss
            / len(self.train_loader)
        )

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        running_loss = 0.0

        predictions = []
        labels_list = []

        for images, labels in self.valid_loader:

            images = images.to(
                self.device,
                non_blocking=True,
            )

            labels = labels.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                labels,
            )

            running_loss += loss.item()

            class_logits = self._get_class_logits(
                outputs
            )

            preds = class_logits.argmax(
                dim=1
            )

            predictions.extend(
                preds.detach()
                .cpu()
                .tolist()
            )

            labels_list.extend(
                labels.detach()
                .cpu()
                .tolist()
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

            if self.cfg.scheduler.name == "plateau":
                self.scheduler.step(
                    val_metrics["val_loss"]
                )
            else:
                self.scheduler.step()

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
    
    def _get_class_logits(self, outputs):

        if isinstance(outputs, dict):
            return outputs["class_logits"]

        return outputs