from pathlib import Path
import os
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from dataset import (
    CassavaDataset,
    train_transform,
    eval_transform,
    CLASS_NAMES,
    LABEL_TO_IDX,
)


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parent

SPLIT_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "split_manifest.csv"
)

OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

MODEL_NAME = os.environ.get(
    "CASSAVA_MODEL_NAME",
    "efficientnet_b0",
)

RUN_TAG = os.environ.get(
    "CASSAVA_RUN_TAG",
    "",
)

OUTPUT_TAG = RUN_TAG or "efficientnet_b0"

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / f"{OUTPUT_TAG}_best.pt"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR
    / f"{OUTPUT_TAG}_last.pt"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / (
        f"training_history_{RUN_TAG}.csv"
        if RUN_TAG
        else "training_history.csv"
    )
)


SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = 0

EPOCHS = 10

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

AMP_INITIAL_SCALE = 8192.0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ================================================================
# CREATE OUTPUT DIRECTORIES
# ================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ================================================================
# DEVICE INFO
# ================================================================

print("=" * 70)
print("CASSAVA LEAF DISEASE TRAINING")
print("=" * 70)

print("Device :", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU    :",
        torch.cuda.get_device_name(0)
    )

    gpu_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
        / 1024**3
    )

    print(
        f"VRAM   : {gpu_memory:.2f} GB"
    )


print(
    "AMP    :",
    USE_AMP
)

print(
    "Model  :",
    MODEL_NAME
)

print(
    "Classes:",
    CLASS_NAMES
)


# ================================================================
# LOAD SPLIT MANIFEST
# ================================================================

df = pd.read_csv(
    SPLIT_MANIFEST
)


train_df = df[
    df["split"] == "train"
].copy()


val_df = df[
    df["split"] == "val"
].copy()


test_df = df[
    df["split"] == "test"
].copy()


print()
print("=" * 70)
print("DATASET")
print("=" * 70)

print(
    "Train:",
    len(train_df)
)

print(
    "Val  :",
    len(val_df)
)

print(
    "Test :",
    len(test_df)
)


# ================================================================
# CLASS WEIGHTS
# ================================================================

class_counts = (
    train_df["label"]
    .value_counts()
)


num_train_samples = len(
    train_df
)


num_classes = len(
    CLASS_NAMES
)


class_weights_list = []


for class_name in CLASS_NAMES:

    count = class_counts[
        class_name
    ]

    weight = (
        num_train_samples
        /
        (
            num_classes
            * count
        )
    )

    class_weights_list.append(
        weight
    )


class_weights = torch.tensor(
    class_weights_list,
    dtype=torch.float32,
    device=DEVICE,
)


print()
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)


for class_name in CLASS_NAMES:

    idx = LABEL_TO_IDX[
        class_name
    ]

    print(
        f"{idx} | "
        f"{class_name:<8} | "
        f"count={class_counts[class_name]:>5} | "
        f"weight={class_weights[idx].item():.4f}"
    )


# ================================================================
# DATASETS
# ================================================================

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


# ================================================================
# DATALOADERS
# ================================================================

train_generator = (
    torch.Generator()
    .manual_seed(SEED)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
    generator=train_generator,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
)


print()
print("=" * 70)
print("DATALOADERS")
print("=" * 70)

print(
    "Train batches:",
    len(train_loader)
)

print(
    "Val batches  :",
    len(val_loader)
)

print(
    "Test batches :",
    len(test_loader)
)


# ================================================================
# MODEL
# ================================================================

print()
print("=" * 70)
print("MODEL")
print("=" * 70)

print(
    "Creating",
    MODEL_NAME,
    "..."
)


model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=num_classes,
)


model = model.to(
    DEVICE
)


print(
    "Classifier:",
    model.get_classifier()
)


# ================================================================
# LOSS FUNCTION
# ================================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ================================================================
# OPTIMIZER
# ================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ================================================================
# LEARNING RATE SCHEDULER
# ================================================================

scheduler = (
    torch.optim.lr_scheduler
    .ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
)


# ================================================================
# AMP
# ================================================================

scaler = torch.amp.GradScaler(
    "cuda",
    init_scale=AMP_INITIAL_SCALE,
    enabled=USE_AMP,
)


# ================================================================
# TRAIN ONE EPOCH
# ================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0


    progress = tqdm(
        loader,
        desc="Train",
        leave=False,
    )


    for images, labels in progress:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ):

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )


        scaler.scale(
            loss
        ).backward()


        scaler.step(
            optimizer
        )


        scaler.update()


        batch_size = (
            labels.size(0)
        )


        total_loss += (
            loss.item()
            * batch_size
        )


        predictions = (
            logits.argmax(
                dim=1
            )
        )


        total_correct += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )


        total_samples += (
            batch_size
        )


        current_loss = (
            total_loss
            / total_samples
        )


        current_accuracy = (
            total_correct
            / total_samples
        )


        progress.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_accuracy * 100:.2f}%",
            amp=f"{scaler.get_scale():.0f}",
        )


    epoch_loss = (
        total_loss
        / total_samples
    )


    epoch_accuracy = (
        total_correct
        / total_samples
    )


    return (
        epoch_loss,
        epoch_accuracy,
    )


# ================================================================
# VALIDATION
# ================================================================

def evaluate(
    model,
    loader,
    criterion,
):

    model.eval()


    total_loss = 0.0

    total_samples = 0


    all_targets = []

    all_predictions = []


    with torch.inference_mode():

        progress = tqdm(
            loader,
            desc="Val",
            leave=False,
        )


        for images, labels in progress:

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )


            with torch.amp.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=USE_AMP,
            ):

                logits = model(
                    images
                )

                loss = criterion(
                    logits,
                    labels
                )


            batch_size = (
                labels.size(0)
            )


            total_loss += (
                loss.item()
                * batch_size
            )


            total_samples += (
                batch_size
            )


            predictions = (
                logits.argmax(
                    dim=1
                )
            )


            all_targets.extend(
                labels
                .detach()
                .cpu()
                .tolist()
            )


            all_predictions.extend(
                predictions
                .detach()
                .cpu()
                .tolist()
            )


    val_loss = (
        total_loss
        / total_samples
    )


    val_accuracy = (
        accuracy_score(
            all_targets,
            all_predictions,
        )
    )


    val_macro_f1 = (
        f1_score(
            all_targets,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            all_targets,
            all_predictions,
            labels=list(
                range(
                    len(CLASS_NAMES)
                )
            ),
            zero_division=0,
        )
    )


    per_class_metrics = {}


    for idx, class_name in enumerate(
        CLASS_NAMES
    ):

        per_class_metrics[
            class_name
        ] = {

            "precision":
                float(
                    precision[idx]
                ),

            "recall":
                float(
                    recall[idx]
                ),

            "f1":
                float(
                    f1[idx]
                ),

            "support":
                int(
                    support[idx]
                ),
        }


    return {
        "loss":
            val_loss,

        "accuracy":
            val_accuracy,

        "macro_f1":
            val_macro_f1,

        "per_class":
            per_class_metrics,
    }


# ================================================================
# SAVE CHECKPOINT
# ================================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    val_metrics,
):

    checkpoint = {

        "epoch":
            epoch,

        "model_name":
            MODEL_NAME,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "class_names":
            CLASS_NAMES,

        "label_to_idx":
            LABEL_TO_IDX,

        "val_macro_f1":
            val_metrics[
                "macro_f1"
            ],

        "val_accuracy":
            val_metrics[
                "accuracy"
            ],

        "val_loss":
            val_metrics[
                "loss"
            ],

        "config": {

            "batch_size":
                BATCH_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "seed":
                SEED,
        },
    }


    torch.save(
        checkpoint,
        path,
    )


# ================================================================
# TRAINING LOOP
# ================================================================

print()
print("=" * 70)
print("START TRAINING")
print("=" * 70)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Initial LR:",
    LEARNING_RATE
)

print(
    "Initial AMP scale:",
    scaler.get_scale()
)


best_macro_f1 = -1.0

best_epoch = 0

history = []


training_start = (
    time.time()
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    epoch_start = (
        time.time()
    )


    print()
    print(
        "=" * 70
    )

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print(
        "=" * 70
    )


    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )
    )


    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    val_metrics = evaluate(
        model,
        val_loader,
        criterion,
    )


    val_loss = (
        val_metrics[
            "loss"
        ]
    )


    val_accuracy = (
        val_metrics[
            "accuracy"
        ]
    )


    val_macro_f1 = (
        val_metrics[
            "macro_f1"
        ]
    )


    # ------------------------------------------------------------
    # SCHEDULER
    # ------------------------------------------------------------

    scheduler.step(
        val_macro_f1
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # ------------------------------------------------------------
    # TIME
    # ------------------------------------------------------------

    epoch_time = (
        time.time()
        - epoch_start
    )


    # ------------------------------------------------------------
    # PRINT SUMMARY
    # ------------------------------------------------------------

    print()
    print("Epoch summary")

    print(
        f"Train loss     : {train_loss:.4f}"
    )

    print(
        f"Train accuracy : {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val loss       : {val_loss:.4f}"
    )

    print(
        f"Val accuracy   : {val_accuracy * 100:.2f}%"
    )

    print(
        f"Val macro-F1   : {val_macro_f1:.4f}"
    )

    print(
        f"Learning rate  : {current_lr:.8f}"
    )

    print(
        f"AMP scale      : {scaler.get_scale():.0f}"
    )

    print(
        f"Epoch time     : {epoch_time:.1f} sec"
    )


    # ------------------------------------------------------------
    # PER-CLASS VALIDATION METRICS
    # ------------------------------------------------------------

    print()
    print(
        "Validation per-class metrics"
    )


    for class_name in CLASS_NAMES:

        metrics = (
            val_metrics[
                "per_class"
            ][
                class_name
            ]
        )


        print(
            f"{class_name:<8} | "
            f"P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"N={metrics['support']}"
        )


    # ------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------

    history_row = {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "val_macro_f1":
            val_macro_f1,

        "learning_rate":
            current_lr,

        "amp_scale":
            scaler.get_scale(),

        "epoch_seconds":
            epoch_time,
    }


    for class_name in CLASS_NAMES:

        metrics = (
            val_metrics[
                "per_class"
            ][
                class_name
            ]
        )


        history_row[
            f"{class_name}_precision"
        ] = metrics[
            "precision"
        ]


        history_row[
            f"{class_name}_recall"
        ] = metrics[
            "recall"
        ]


        history_row[
            f"{class_name}_f1"
        ] = metrics[
            "f1"
        ]


    history.append(
        history_row
    )


    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_PATH,
        index=False,
    )


    # ------------------------------------------------------------
    # BEST CHECKPOINT
    # ------------------------------------------------------------

    if (
        val_macro_f1
        > best_macro_f1
    ):

        best_macro_f1 = (
            val_macro_f1
        )

        best_epoch = (
            epoch
        )


        save_checkpoint(
            BEST_MODEL_PATH,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            val_metrics,
        )


        print()
        print(
            "✅ New best model saved"
        )

        print(
            "Path:",
            BEST_MODEL_PATH
        )

        print(
            f"Best macro-F1: {best_macro_f1:.4f}"
        )


    # ------------------------------------------------------------
    # SAVE LAST MODEL
    # ------------------------------------------------------------

    save_checkpoint(
        LAST_MODEL_PATH,
        epoch,
        model,
        optimizer,
        scheduler,
        scaler,
        val_metrics,
    )


# ================================================================
# TRAINING FINISHED
# ================================================================

total_training_time = (
    time.time()
    - training_start
)


print()
print("=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print(
    "Best epoch:",
    best_epoch
)

print(
    f"Best validation macro-F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"Total training time: "
    f"{total_training_time / 60:.2f} min"
)

print()
print(
    "Best model:",
    BEST_MODEL_PATH
)

print(
    "Last model:",
    LAST_MODEL_PATH
)

print(
    "History:",
    HISTORY_PATH
)


# ================================================================
# GPU MEMORY
# ================================================================

if DEVICE.type == "cuda":

    max_memory = (
        torch.cuda
        .max_memory_allocated()
        / 1024**3
    )


    print()
    print(
        f"Maximum GPU memory allocated: "
        f"{max_memory:.2f} GB"
    )
