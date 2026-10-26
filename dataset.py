from pathlib import Path
from torchvision import transforms
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


CLASS_NAMES = [
    "cbb",
    "cbs",
    "cmd",
    "healthy",
]

LABEL_TO_IDX = {
    name: idx
    for idx, name in enumerate(CLASS_NAMES)
}

IDX_TO_LABEL = {
    idx: name
    for name, idx in LABEL_TO_IDX.items()
}


class CassavaDataset(Dataset):

    def __init__(
        self,
        dataframe,
        root,
        transform=None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = self.root / row["filepath"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        label_name = row["label"]
        label_idx = LABEL_TO_IDX[label_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx
    
    
IMAGE_SIZE = 224


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.80, 1.0),
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomVerticalFlip(
        p=0.2
    ),

    transforms.RandomRotation(
        degrees=15
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parent

    SPLIT_MANIFEST = (
        ROOT
        / "data"
        / "processed"
        / "split_manifest.csv"
    )

    df = pd.read_csv(SPLIT_MANIFEST)

    train_df = df[
        df["split"] == "train"
    ].copy()

    dataset = CassavaDataset(
        dataframe=train_df,
        root=ROOT,
        transform=train_transform,
    )

    print("Dataset size:", len(dataset))

    image, label_idx = dataset[0]

    print("Image shape :", image.shape)
    print("Image dtype :", image.dtype)

    print("Label index :", label_idx)
    print(
        "Label name  :",
        IDX_TO_LABEL[label_idx]
    )

    print()
    print("Label map:")
    print(LABEL_TO_IDX)
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    images, labels = next(iter(loader))

    print()
    print("First batch")
    print("Images shape :", images.shape)
    print("Labels shape :", labels.shape)
    print("Images dtype :", images.dtype)
    print("Labels dtype :", labels.dtype)

    print()
    print("First 10 labels:")
    print(labels[:10])

    print()
    print("First 10 label names:")
    print([
        IDX_TO_LABEL[label.item()]
        for label in labels[:10]
    ])
    
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    images, labels = next(iter(loader))

    print()
    print("First batch")
    print("Images shape :", images.shape)
    print("Labels shape :", labels.shape)
    print("Images dtype :", images.dtype)
    print("Labels dtype :", labels.dtype)

    print()
    print("First 10 labels:")
    print(labels[:10])

    print()
    print("First 10 label names:")
    print([
        IDX_TO_LABEL[label.item()]
        for label in labels[:10]
    ])
    
    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()
    
    train_dataset = CassavaDataset(
        dataframe=train_df,
        root=ROOT,
        transform=train_transform,
    )

    val_dataset = CassavaDataset(
        dataframe=val_df,
        root=ROOT,
        transform=eval_transform,
    )

    test_dataset = CassavaDataset(
        dataframe=test_df,
        root=ROOT,
        transform=eval_transform,
    )
    
    BATCH_SIZE = 64

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    print()
    print("=" * 64)
    print("DATALOADER SUMMARY")
    print("=" * 64)

    print("Train samples :", len(train_dataset))
    print("Val samples   :", len(val_dataset))
    print("Test samples  :", len(test_dataset))

    print()
    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))
    print("Test batches  :", len(test_loader))
    
    
    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    print()
    print("Train batch:")
    print("  images:", train_images.shape)
    print("  labels:", train_labels.shape)

    print()
    print("Val batch:")
    print("  images:", val_images.shape)
    print("  labels:", val_labels.shape)