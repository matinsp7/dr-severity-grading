from src.datasets.aptos_dataset import APTOSDataset

dataset = APTOSDataset(
    csv_file="data/train_1.csv",
    image_dir="data/train_images",
)

print(f"Dataset size: {len(dataset)}")

image, label = dataset[0]

print(type(image))
print(image.size)
print(label)