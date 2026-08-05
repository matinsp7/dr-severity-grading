from torch.utils.data import DataLoader

from src.augmentations.transforms import get_train_transform
from src.datasets.aptos_dataset import APTOSDataset


def get_train_dataloader(
    csv_file: str,
    image_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
):
    dataset = APTOSDataset(
        csv_file=csv_file,
        image_dir=image_dir,
        transform=get_train_transform(),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )