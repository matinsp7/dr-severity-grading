import torch
import torch.nn as nn


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        criterion,
        device,
    ):

        self.model = model

        self.train_loader = train_loader

        self.optimizer = optimizer

        self.criterion = criterion

        self.device = device

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device)

            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {running_loss/20:.4f}"
                )

                running_loss = 0