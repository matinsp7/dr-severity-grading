from src.datasets.aptos_dataset import APTOSDataset
from src.augmentations.transforms import get_train_transform

dataset = APTOSDataset(
    csv_file="data/train_1.csv",
    image_dir="data/train_images",
    transform=get_train_transform(),
)

print(f"Dataset size: {len(dataset)}")

image, label = dataset[0]

print(image.shape)
print(image.dtype)
print(label)