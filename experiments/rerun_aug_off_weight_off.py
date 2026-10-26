from pathlib import Path
import sys
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from dataset import (
    CassavaDataset,
    eval_transform,
    CLASS_NAMES,
    LABEL_TO_IDX,
)


# ================================================================
# CONFIG
# ================================================================


SPLIT_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "split_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "ablation"
    / "aug_off_weight_off_rerun"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "history.csv"
)

RESULT_PATH = (
    OUTPUT_DIR
    / "verified_result.csv"
)

BEST_MODEL_PATH = (
    OUTPUT_DIR
    / "best.pt"
)

LAST_MODEL_PATH = (
    OUTPUT_DIR
    / "last.pt"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 42

BATCH_SIZE = 64
NUM_WORKERS = 0

EPOCHS = 10

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

AMP_INITIAL_SCALE = 8192.0

MODEL_NAME = "efficientnet_b0"


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ================================================================
# SEED
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

print("=" * 72)
print("RERUN - AUG OFF + CLASS WEIGHT OFF")
print("=" * 72)

print(
    "Device:",
    DEVICE
)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "AMP:",
    USE_AMP
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Augmentation:",
    False
)

print(
    "Class weighting:",
    False
)


# ================================================================
# LOAD DATA
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


print()
print("=" * 72)
print("DATA")
print("=" * 72)

print(
    "Train:",
    len(train_df)
)

print(
    "Val:",
    len(val_df)
)


# ================================================================
# DATASETS
#
# Augmentation OFF
# ดังนั้น train และ validation ใช้ eval_transform เหมือนกัน
# ================================================================

train_dataset = CassavaDataset(
    dataframe=train_df,
    root=ROOT,
    transform=eval_transform,
)


val_dataset = CassavaDataset(
    dataframe=val_df,
    root=ROOT,
    transform=eval_transform,
)


# ================================================================
# DATALOADERS
# ================================================================

generator = torch.Generator()

generator.manual_seed(
    SEED
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
    generator=generator,
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
)


print()
print(
    "Train batches:",
    len(train_loader)
)

print(
    "Val batches:",
    len(val_loader)
)


# ================================================================
# MODEL
# ================================================================

print()
print("=" * 72)
print("MODEL")
print("=" * 72)


model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=len(CLASS_NAMES),
)


model = model.to(
    DEVICE
)


print(
    "Classifier:",
    model.get_classifier()
)


# ================================================================
# LOSS
#
# Class weighting OFF
# ================================================================

criterion = nn.CrossEntropyLoss()


# ================================================================
# OPTIMIZER
# ================================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ================================================================
# SCHEDULER
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

def train_one_epoch():

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0


    progress = tqdm(
        train_loader,
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


        predictions = logits.argmax(
            dim=1
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


        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{total_correct / total_samples * 100:.2f}%",
            amp=f"{scaler.get_scale():.0f}",
        )


    train_loss = (
        total_loss
        / total_samples
    )


    train_accuracy = (
        total_correct
        / total_samples
    )


    return (
        train_loss,
        train_accuracy,
    )


# ================================================================
# VALIDATION
# ================================================================

def validate():

    model.eval()

    total_loss = 0.0
    total_samples = 0

    targets = []
    predictions_all = []


    with torch.inference_mode():

        progress = tqdm(
            val_loader,
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


            predictions = logits.argmax(
                dim=1
            )


            targets.extend(
                labels
                .cpu()
                .tolist()
            )


            predictions_all.extend(
                predictions
                .cpu()
                .tolist()
            )


    val_loss = (
        total_loss
        / total_samples
    )


    val_accuracy = accuracy_score(
        targets,
        predictions_all,
    )


    val_macro_f1 = f1_score(
        targets,
        predictions_all,
        average="macro",
        zero_division=0,
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            targets,
            predictions_all,
            labels=list(
                range(
                    len(CLASS_NAMES)
                )
            ),
            zero_division=0,
        )
    )


    per_class = {}


    for idx, class_name in enumerate(
        CLASS_NAMES
    ):

        per_class[
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
            per_class,
    }


# ================================================================
# SAVE CHECKPOINT
# ================================================================

def save_checkpoint(
    path,
    epoch,
    val_metrics,
):

    checkpoint = {

        "experiment":
            "aug_off_weight_off",

        "augmentation":
            False,

        "class_weighting":
            False,

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
    }


    torch.save(
        checkpoint,
        path,
    )


# ================================================================
# TRAIN
# ================================================================

print()
print("=" * 72)
print("START RERUN")
print("=" * 72)


best_macro_f1 = -1.0
best_epoch = 0
best_accuracy = 0.0
best_loss = 0.0

history = []

training_start = time.time()


for epoch in range(
    1,
    EPOCHS + 1,
):

    epoch_start = time.time()


    print()
    print("=" * 72)

    print(
        f"EPOCH {epoch}/{EPOCHS}"
    )

    print("=" * 72)


    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    train_loss, train_accuracy = (
        train_one_epoch()
    )


    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    val_metrics = validate()


    scheduler.step(
        val_metrics[
            "macro_f1"
        ]
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    epoch_seconds = (
        time.time()
        - epoch_start
    )


    # ------------------------------------------------------------
    # PRINT
    # ------------------------------------------------------------

    print(
        f"Train loss     : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train accuracy : "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val loss       : "
        f"{val_metrics['loss']:.4f}"
    )

    print(
        f"Val accuracy   : "
        f"{val_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Val macro-F1   : "
        f"{val_metrics['macro_f1']:.6f}"
    )

    print(
        f"Learning rate  : "
        f"{current_lr:.8f}"
    )

    print(
        f"AMP scale      : "
        f"{scaler.get_scale():.0f}"
    )

    print(
        f"Epoch time     : "
        f"{epoch_seconds:.1f} sec"
    )


    print()
    print(
        "Per-class F1:"
    )


    for class_name in CLASS_NAMES:

        print(
            f"  "
            f"{class_name:<8}"
            f"{val_metrics['per_class'][class_name]['f1']:.4f}"
        )


    # ------------------------------------------------------------
    # HISTORY
    # ------------------------------------------------------------

    row = {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_metrics[
                "loss"
            ],

        "val_accuracy":
            val_metrics[
                "accuracy"
            ],

        "val_macro_f1":
            val_metrics[
                "macro_f1"
            ],

        "learning_rate":
            current_lr,

        "amp_scale":
            scaler.get_scale(),

        "epoch_seconds":
            epoch_seconds,
    }


    for class_name in CLASS_NAMES:

        metrics = (
            val_metrics[
                "per_class"
            ][
                class_name
            ]
        )


        row[
            f"{class_name}_precision"
        ] = metrics[
            "precision"
        ]

        row[
            f"{class_name}_recall"
        ] = metrics[
            "recall"
        ]

        row[
            f"{class_name}_f1"
        ] = metrics[
            "f1"
        ]


    history.append(
        row
    )


    pd.DataFrame(
        history
    ).to_csv(
        HISTORY_PATH,
        index=False,
    )


    # ------------------------------------------------------------
    # SAVE EVERY EPOCH
    # ------------------------------------------------------------

    epoch_checkpoint_path = (
        CHECKPOINT_DIR
        / f"epoch_{epoch:02d}.pt"
    )


    save_checkpoint(
        epoch_checkpoint_path,
        epoch,
        val_metrics,
    )


    # ------------------------------------------------------------
    # BEST
    # ------------------------------------------------------------

    if (
        val_metrics[
            "macro_f1"
        ]
        > best_macro_f1
    ):

        best_macro_f1 = (
            val_metrics[
                "macro_f1"
            ]
        )

        best_epoch = epoch

        best_accuracy = (
            val_metrics[
                "accuracy"
            ]
        )

        best_loss = (
            val_metrics[
                "loss"
            ]
        )


        save_checkpoint(
            BEST_MODEL_PATH,
            epoch,
            val_metrics,
        )


        print()
        print(
            "✅ NEW BEST MODEL"
        )

        print(
            f"Epoch: {best_epoch}"
        )

        print(
            f"Macro-F1: "
            f"{best_macro_f1:.6f}"
        )


    # ------------------------------------------------------------
    # LAST
    # ------------------------------------------------------------

    save_checkpoint(
        LAST_MODEL_PATH,
        epoch,
        val_metrics,
    )


# ================================================================
# TRAINING FINISHED
# ================================================================

training_minutes = (
    time.time()
    - training_start
) / 60


print()
print("=" * 72)
print("TRAINING FINISHED")
print("=" * 72)

print(
    "Best epoch:",
    best_epoch
)

print(
    f"Best Val Macro-F1: "
    f"{best_macro_f1:.6f}"
)

print(
    f"Best Val Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Training time: "
    f"{training_minutes:.2f} min"
)


# ================================================================
# VERIFY BEST.PT AGAINST HISTORY
# ================================================================

history_df = pd.read_csv(
    HISTORY_PATH
)


history_best = (
    history_df.loc[
        history_df[
            "val_macro_f1"
        ].idxmax()
    ]
)


checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location="cpu",
    weights_only=False,
)


history_best_epoch = int(
    history_best[
        "epoch"
    ]
)

history_best_f1 = float(
    history_best[
        "val_macro_f1"
    ]
)


checkpoint_epoch = int(
    checkpoint[
        "epoch"
    ]
)

checkpoint_f1 = float(
    checkpoint[
        "val_macro_f1"
    ]
)


epoch_matches = (
    history_best_epoch
    == checkpoint_epoch
)


f1_matches = (
    abs(
        history_best_f1
        - checkpoint_f1
    )
    < 1e-8
)


print()
print("=" * 72)
print("FINAL CHECKPOINT VERIFICATION")
print("=" * 72)

print(
    "History best epoch :",
    history_best_epoch
)

print(
    "Checkpoint epoch   :",
    checkpoint_epoch
)

print(
    "Epoch matches      :",
    epoch_matches
)

print()

print(
    "History best F1    :",
    history_best_f1
)

print(
    "Checkpoint F1      :",
    checkpoint_f1
)

print(
    "F1 matches         :",
    f1_matches
)


if not epoch_matches:

    raise RuntimeError(
        "Checkpoint epoch does not match history."
    )


if not f1_matches:

    raise RuntimeError(
        "Checkpoint F1 does not match history."
    )


# ================================================================
# SAVE VERIFIED RESULT
# ================================================================

result = pd.DataFrame(
    [
        {

            "experiment":
                "aug_off_weight_off",

            "augmentation":
                False,

            "class_weighting":
                False,

            "best_epoch":
                checkpoint_epoch,

            "best_val_macro_f1":
                checkpoint_f1,

            "best_val_accuracy":
                float(
                    checkpoint[
                        "val_accuracy"
                    ]
                ),

            "best_val_loss":
                float(
                    checkpoint[
                        "val_loss"
                    ]
                ),

            "training_minutes":
                training_minutes,

            "checkpoint_verified":
                True,
        }
    ]
)


result.to_csv(
    RESULT_PATH,
    index=False,
)


print()
print("=" * 72)
print("RERUN VERIFIED ✅")
print("=" * 72)

print(
    result.to_string(
        index=False
    )
)

print()
print(
    "Best checkpoint:",
    BEST_MODEL_PATH
)

print(
    "History:",
    HISTORY_PATH
)

print(
    "Verified result:",
    RESULT_PATH
)
