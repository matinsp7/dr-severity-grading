from src.datasets.dataloader import get_train_dataloader

train_loader = get_train_dataloader(
    csv_file="data/train_1.csv",
    image_dir="data/train_images",
)

images, labels = next(iter(train_loader))

print(images.shape)
print(labels.shape)
print(labels)