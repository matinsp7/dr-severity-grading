import torch
import torch.nn as nn

from src.datasets.dataloader import get_train_dataloader
from src.models.efficientnet import EfficientNetDR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    print(f"Using device: {DEVICE}")

    train_loader = get_train_dataloader(
        csv_file="data/train_1.csv",
        image_dir="data/train_images",
        batch_size=16,
        num_workers=0,
    )

    model = EfficientNetDR(num_classes=5).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Batch {batch_idx+1}/{len(train_loader)} | "
                f"Loss: {running_loss/20:.4f}"
            )
            running_loss = 0.0

    print("Training Finished!")


if __name__ == "__main__":
    main()