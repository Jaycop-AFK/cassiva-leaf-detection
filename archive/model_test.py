from pathlib import Path
import sys

import pandas as pd

import torch
from torch.utils.data import DataLoader

import timm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset import (
    CassavaDataset,
    train_transform,
    CLASS_NAMES,
)



SPLIT_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "split_manifest.csv"
)

BATCH_SIZE = 64
NUM_CLASSES = len(CLASS_NAMES)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 64)
print("MODEL FORWARD TEST")
print("=" * 64)

print("Device      :", device)

if device.type == "cuda":
    print("GPU         :", torch.cuda.get_device_name(0))
    print(
        "VRAM        :",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2
        ),
        "GB"
    )

df = pd.read_csv(SPLIT_MANIFEST)

train_df = df[
    df["split"] == "train"
].copy()

train_dataset = CassavaDataset(
    dataframe=train_df,
    root=ROOT,
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

images, labels = next(iter(train_loader))

print()
print("CPU batch")
print("Images :", images.shape)
print("Labels :", labels.shape)


print()
print("Creating EfficientNet-B0...")

model = timm.create_model(
    "efficientnet_b0",
    pretrained=True,
    num_classes=NUM_CLASSES,
)

model = model.to(device)

print("Model created successfully")


print()
print("Classifier:")
print(model.get_classifier())


images = images.to(
    device,
    non_blocking=True
)

labels = labels.to(
    device,
    non_blocking=True
)

print()
print("GPU batch")
print("Images device:", images.device)
print("Labels device:", labels.device)


model.eval()

with torch.inference_mode():
    logits = model(images)
    
    
print()
print("Forward pass")
print("Logits shape :", logits.shape)
print("Logits dtype :", logits.dtype)
print("Logits device:", logits.device)


predictions = logits.argmax(dim=1)

print()
print("First 10 predictions:")
print(predictions[:10])

print("First 10 true labels:")
print(labels[:10])


if device.type == "cuda":

    with torch.inference_mode():
        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            amp_logits = model(images)

    print()
    print("AMP forward")
    print("Shape :", amp_logits.shape)
    print("Dtype :", amp_logits.dtype)
    
    
if device.type == "cuda":

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print()
    print("GPU memory")
    print(
        f"Allocated: {allocated:.2f} GB"
    )
    print(
        f"Reserved : {reserved:.2f} GB"
    )
