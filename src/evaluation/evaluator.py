import torch
from tqdm import tqdm

from src.metrics.metrics import (
    compute_auc_metrics,
    compute_metrics,
)


class Evaluator:

    def __init__(
        self,
        model,
        dataloader,
        criterion,
        device,
    ):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

    @torch.inference_mode()
    def predict(self):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        labels = []
        predictions = []
        probabilities = []

        for images, batch_labels in tqdm(self.dataloader, "Evaluating", leave=False):
            images = images.to(
                self.device,
                non_blocking=True,
            )

            batch_labels = batch_labels.to(
                self.device,
                non_blocking=True,
            )

            logits = self.model(images)

            loss = self.criterion(
                logits,
                batch_labels,
            )

            total_loss += loss.item() * batch_labels.size(0)
            total_samples += batch_labels.size(0)

            batch_probabilities = torch.softmax(
                logits,
                dim=1,
            )

            batch_probabilities = torch.softmax(
                logits,
                dim=1,
            )

            batch_predictions = torch.sum(
                batch_probabilities * self.criterion.class_indices,
                dim=1,
            )

            batch_predictions = torch.round(
                batch_predictions
            ).long()

            labels.extend(
                batch_labels.cpu().tolist()
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

            probabilities.extend(
                batch_probabilities.cpu().tolist()
            )

        return {
            "labels": labels,
            "predictions": predictions,
            "probabilities": probabilities,
            "loss": total_loss / total_samples
        }

    def evaluate(self):
        results = self.predict()

        metrics = compute_metrics(
            results["labels"],
            results["predictions"],
        )

        auc_metrics = compute_auc_metrics(
            results["labels"],
            results["probabilities"],
        )

        metrics.update(
            auc_metrics
        )

        metrics["val_loss"] = results["loss"]

        return {
            "metrics": metrics,
            "labels": results["labels"],
            "predictions": results["predictions"],
            "probabilities": results["probabilities"],
        }