import torch

from src.metrics.metrics import (
    compute_metrics,
)


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        criterion,
        device,
    ):

        self.model = model

        self.train_loader = train_loader

        self.valid_loader = valid_loader

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0

        for images, labels in self.train_loader:

            images = images.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        predictions = []

        labels_list = []

        running_loss = 0

        for images, labels in self.valid_loader:

            images = images.to(self.device)

            labels = labels.to(self.device)

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())

            labels_list.extend(labels.cpu().numpy())

        metrics = compute_metrics(
            labels_list,
            predictions,
        )

        metrics["loss"] = (
            running_loss
            / len(self.valid_loader)
        )

        return metrics
    
    def fit(self, epochs):

        best_qwk = -1

        for epoch in range(epochs):

            train_loss = self.train_one_epoch()

            metrics = self.validate()

            print()

            print(f"Epoch {epoch+1}/{epochs}")

            print(f"Train Loss : {train_loss:.4f}")

            print(f"Val Loss   : {metrics['loss']:.4f}")

            print(f"Accuracy   : {metrics['accuracy']:.4f}")

            print(f"Precision  : {metrics['precision']:.4f}")

            print(f"Recall     : {metrics['recall']:.4f}")

            print(f"F1         : {metrics['f1']:.4f}")

            print(f"QWK        : {metrics['qwk']:.4f}")

            if metrics["qwk"] > best_qwk:

                best_qwk = metrics["qwk"]

                print("New Best Model!")

                torch.save(
                    self.model.state_dict(),
                    "outputs/best_model.pth",
                )