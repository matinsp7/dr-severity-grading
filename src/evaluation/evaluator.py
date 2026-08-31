import numpy as np
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
        inference_mode="argmax",
        fusion_lambda=0.7,
        thresholds=None,
    ):

        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.device = device

        self.inference_mode = inference_mode
        self.fusion_lambda = fusion_lambda

        if thresholds is None:
            thresholds = [
                0.5,
                1.5,
                2.5,
                3.5,
            ]

        self.thresholds = np.asarray(
            thresholds,
            dtype=np.float32,
        )

    def _get_class_logits(self, outputs):

        if isinstance(outputs, dict):
            return outputs["class_logits"]

        return outputs

    def _get_predictions(
        self,
        outputs,
    ):

        class_logits = self._get_class_logits(
            outputs
        )

        class_probabilities = torch.softmax(
            class_logits,
            dim=1,
        )

        if self.inference_mode == "argmax":

            return (
                class_probabilities.argmax(
                    dim=1
                ),
                class_probabilities,
            )

        if self.inference_mode == "fused":

            if not isinstance(outputs, dict):
                raise TypeError(
                    "Fused inference requires "
                    "dictionary model outputs."
                )

            ordinal_logits = outputs[
                "ordinal_logits"
            ]

            ordinal_probabilities = (
                torch.sigmoid(
                    ordinal_logits
                )
            )

            class_ids = torch.arange(
                class_probabilities.shape[1],
                device=class_probabilities.device,
                dtype=class_probabilities.dtype,
            )

            class_expected = (
                class_probabilities
                * class_ids
            ).sum(
                dim=1
            )

            ordinal_expected = (
                ordinal_probabilities.sum(
                    dim=1
                )
            )

            fused_score = (
                self.fusion_lambda
                * class_expected
                + (
                    1.0
                    - self.fusion_lambda
                )
                * ordinal_expected
            )

            threshold_tensor = torch.tensor(
                self.thresholds,
                device=fused_score.device,
                dtype=fused_score.dtype,
            )

            predictions = torch.bucketize(
                fused_score,
                threshold_tensor,
                right=False,
            )

            return (
                predictions,
                class_probabilities,
            )

        raise ValueError(
            f"Unknown inference mode: "
            f"{self.inference_mode}"
        )

    @torch.inference_mode()
    def predict(self):

        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        labels = []
        predictions = []
        probabilities = []

        for images, batch_labels in tqdm(
            self.dataloader,
            "Evaluating",
            leave=False,
        ):

            images = images.to(
                self.device,
                non_blocking=True,
            )

            batch_labels = batch_labels.to(
                self.device,
                non_blocking=True,
            )

            outputs = self.model(
                images
            )

            loss = self.criterion(
                outputs,
                batch_labels,
            )

            batch_size = (
                batch_labels.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            (
                batch_predictions,
                batch_probabilities,
            ) = self._get_predictions(
                outputs
            )

            labels.extend(
                batch_labels.cpu().tolist()
            )

            predictions.extend(
                batch_predictions.cpu().tolist()
            )

            probabilities.extend(
                batch_probabilities.cpu().tolist()
            )

        if total_samples == 0:
            raise RuntimeError(
                "Evaluation dataloader "
                "produced zero samples."
            )

        return {
            "labels": labels,
            "predictions": predictions,
            "probabilities": probabilities,
            "loss": (
                total_loss
                / total_samples
            ),
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

        metrics["val_loss"] = (
            results["loss"]
        )

        return {
            "metrics": metrics,
            "labels": results["labels"],
            "predictions": results["predictions"],
            "probabilities": results["probabilities"],
        }