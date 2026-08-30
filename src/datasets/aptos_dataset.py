from pathlib import Path
import numpy as np

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class APTOSDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labels = self.data["diagnosis"].astype(int).to_numpy()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = self.image_dir / f"{row['id_code']}.png"

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        label = int(row["diagnosis"])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label
