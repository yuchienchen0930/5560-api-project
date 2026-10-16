import io
import os
import urllib.request

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Official CIFAR-10 dataset (uoft-cs) hosted on Hugging Face as Parquet files.
# Same 50,000 train / 10,000 test images and labels as torchvision.datasets.CIFAR10.
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet"

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


class CIFAR10Parquet(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        split = "train" if train else "test"
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, f"{split}.parquet")
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError(path)
            print(f"Downloading CIFAR-10 {split} split...")
            urllib.request.urlretrieve(HF_URL.format(split=split), path)

        df = pd.read_parquet(path)
        self.images = [Image.open(io.BytesIO(img["bytes"])).convert("RGB") for img in df["img"]]
        self.labels = df["label"].tolist()
        self.transform = transform
        self.classes = CLASSES

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
