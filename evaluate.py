from pathlib import Path
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
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

ROOT = Path(__file__).resolve().parent

SPLIT_MANIFEST = (
    ROOT
    / "data"
    / "processed"
    / "split_manifest.csv"
)

MODEL_NAME = os.environ.get("CASSAVA_MODEL_NAME", "efficientnet_b0")
EVAL_TAG = os.environ.get("CASSAVA_EVAL_TAG", "")

CHECKPOINT_PATH = Path(
    os.environ.get(
        "CASSAVA_CHECKPOINT",
        str(ROOT / "outputs" / "checkpoints" / "efficientnet_b0_best.pt"),
    )
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / (f"evaluation_{EVAL_TAG}" if EVAL_TAG else "evaluation")
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "test_metrics.csv"
)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "confusion_matrix.png"
)

NORMALIZED_CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR
    / "confusion_matrix_normalized.png"
)


BATCH_SIZE = 64
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

USE_AMP = (
    DEVICE.type == "cuda"
)


# ================================================================
# DEVICE INFO
# ================================================================

print("=" * 70)
print("STEP 7 - FINAL TEST EVALUATION")
print("=" * 70)

print("Device :", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU    :",
        torch.cuda.get_device_name(0)
    )


print(
    "AMP    :",
    USE_AMP
)


# ================================================================
# LOAD DATA
# ================================================================

df = pd.read_csv(
    SPLIT_MANIFEST
)


test_df = df[
    df["split"] == "test"
].copy()


print()
print("=" * 70)
print("TEST DATA")
print("=" * 70)

print(
    "Test samples:",
    len(test_df)
)

print()

print(
    test_df["label"]
    .value_counts()
)


# ================================================================
# DATASET
# ================================================================

test_dataset = CassavaDataset(
    dataframe=test_df,
    root=ROOT,
    transform=eval_transform,
)


# ================================================================
# DATALOADER
# ================================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=USE_AMP,
    drop_last=False,
)


print()
print(
    "Test batches:",
    len(test_loader)
)


# ================================================================
# LOAD CHECKPOINT
# ================================================================

print()
print("=" * 70)
print("LOAD BEST MODEL")
print("=" * 70)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False,
)


print(
    "Checkpoint epoch:",
    checkpoint["epoch"]
)

print(
    "Validation macro-F1:",
    checkpoint["val_macro_f1"]
)


# ================================================================
# CREATE MODEL
# ================================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=len(CLASS_NAMES),
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "Model loaded successfully"
)


# ================================================================
# LOSS
# ================================================================

train_df = df[
    df["split"] == "train"
].copy()


class_counts = (
    train_df["label"]
    .value_counts()
)


num_train = len(
    train_df
)


num_classes = len(
    CLASS_NAMES
)


weights = []


for class_name in CLASS_NAMES:

    count = class_counts[
        class_name
    ]

    weight = (
        num_train
        /
        (
            num_classes
            * count
        )
    )

    weights.append(
        weight
    )


class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=DEVICE,
)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ================================================================
# TEST LOOP
# ================================================================

all_targets = []
all_predictions = []
all_probabilities = []

total_loss = 0.0
total_samples = 0


print()
print("=" * 70)
print("RUN TEST SET")
print("=" * 70)


with torch.inference_mode():

    progress = tqdm(
        test_loader,
        desc="Testing",
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


        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )


        predictions = probabilities.argmax(
            dim=1
        )


        batch_size = labels.size(0)


        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size


        all_targets.extend(
            labels
            .cpu()
            .tolist()
        )


        all_predictions.extend(
            predictions
            .cpu()
            .tolist()
        )


        all_probabilities.extend(
            probabilities
            .cpu()
            .numpy()
            .tolist()
        )


# ================================================================
# METRICS
# ================================================================

test_loss = (
    total_loss
    / total_samples
)


test_accuracy = accuracy_score(
    all_targets,
    all_predictions,
)


test_macro_f1 = f1_score(
    all_targets,
    all_predictions,
    average="macro",
    zero_division=0,
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


# ================================================================
# PRINT FINAL RESULTS
# ================================================================

print()
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test loss     : {test_loss:.4f}"
)

print(
    f"Test accuracy : {test_accuracy * 100:.2f}%"
)

print(
    f"Test macro-F1 : {test_macro_f1:.4f}"
)


print()
print(
    "Per-class metrics"
)

print(
    "-" * 70
)


for idx, class_name in enumerate(
    CLASS_NAMES
):

    print(
        f"{class_name:<8} | "
        f"P={precision[idx]:.4f} | "
        f"R={recall[idx]:.4f} | "
        f"F1={f1[idx]:.4f} | "
        f"N={support[idx]}"
    )


# ================================================================
# SAVE METRICS
# ================================================================

metrics_rows = []


for idx, class_name in enumerate(
    CLASS_NAMES
):

    metrics_rows.append({

        "class":
            class_name,

        "precision":
            precision[idx],

        "recall":
            recall[idx],

        "f1":
            f1[idx],

        "support":
            support[idx],
    })


metrics_df = pd.DataFrame(
    metrics_rows
)


metrics_df.to_csv(
    METRICS_PATH,
    index=False,
)


# ================================================================
# CONFUSION MATRIX
# ================================================================

cm = confusion_matrix(
    all_targets,
    all_predictions,
    labels=list(
        range(
            len(CLASS_NAMES)
        )
    ),
)


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)


# ================================================================
# PLOT CONFUSION MATRIX
# ================================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)


image = ax.imshow(
    cm
)


ax.set_xticks(
    np.arange(
        len(CLASS_NAMES)
    )
)

ax.set_yticks(
    np.arange(
        len(CLASS_NAMES)
    )
)


ax.set_xticklabels(
    CLASS_NAMES
)

ax.set_yticklabels(
    CLASS_NAMES
)


ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "True"
)

ax.set_title(
    "Confusion Matrix - Test Set"
)


for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        len(CLASS_NAMES)
    ):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
        )


fig.colorbar(
    image,
    ax=ax,
)


plt.tight_layout()


plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ================================================================
# NORMALIZED CONFUSION MATRIX
# ================================================================

cm_normalized = (
    cm.astype(
        np.float64
    )
    /
    cm.sum(
        axis=1,
        keepdims=True,
    )
)


fig, ax = plt.subplots(
    figsize=(8, 7)
)


image = ax.imshow(
    cm_normalized
)


ax.set_xticks(
    np.arange(
        len(CLASS_NAMES)
    )
)

ax.set_yticks(
    np.arange(
        len(CLASS_NAMES)
    )
)


ax.set_xticklabels(
    CLASS_NAMES
)

ax.set_yticklabels(
    CLASS_NAMES
)


ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "True"
)

ax.set_title(
    "Normalized Confusion Matrix - Test Set"
)


for i in range(
    len(CLASS_NAMES)
):

    for j in range(
        len(CLASS_NAMES)
    ):

        ax.text(
            j,
            i,
            f"{cm_normalized[i, j]:.2f}",
            ha="center",
            va="center",
        )


fig.colorbar(
    image,
    ax=ax,
)


plt.tight_layout()


plt.savefig(
    NORMALIZED_CONFUSION_MATRIX_PATH,
    dpi=200,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ================================================================
# SAVE INDIVIDUAL PREDICTIONS
# ================================================================

prediction_df = test_df.reset_index(
    drop=True
).copy()


prediction_df[
    "true_idx"
] = all_targets


prediction_df[
    "pred_idx"
] = all_predictions


prediction_df[
    "pred_label"
] = [
    CLASS_NAMES[idx]
    for idx in all_predictions
]


prediction_df[
    "correct"
] = (
    prediction_df[
        "true_idx"
    ]
    ==
    prediction_df[
        "pred_idx"
    ]
)


probabilities_array = np.array(
    all_probabilities
)


for idx, class_name in enumerate(
    CLASS_NAMES
):

    prediction_df[
        f"prob_{class_name}"
    ] = probabilities_array[
        :,
        idx
    ]


prediction_df[
    "confidence"
] = probabilities_array.max(
    axis=1
)


prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ================================================================
# CLASSIFICATION REPORT
# ================================================================

report = classification_report(
    all_targets,
    all_predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0,
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ================================================================
# ERROR ANALYSIS SUMMARY
# ================================================================

wrong_df = prediction_df[
    prediction_df["correct"] == False
].copy()


print()
print("=" * 70)
print("ERROR SUMMARY")
print("=" * 70)

print(
    "Correct:",
    prediction_df["correct"].sum()
)

print(
    "Wrong  :",
    len(wrong_df)
)


if len(wrong_df) > 0:

    print()
    print(
        "Most common mistakes:"
    )

    mistakes = (
        wrong_df
        .groupby(
            [
                "label",
                "pred_label",
            ]
        )
        .size()
        .sort_values(
            ascending=False
        )
    )

    print(
        mistakes.head(10)
    )


# ================================================================
# OUTPUT FILES
# ================================================================

print()
print("=" * 70)
print("EVALUATION FINISHED")
print("=" * 70)

print(
    "Metrics:",
    METRICS_PATH
)

print(
    "Predictions:",
    PREDICTIONS_PATH
)

print(
    "Confusion matrix:",
    CONFUSION_MATRIX_PATH
)

print(
    "Normalized confusion matrix:",
    NORMALIZED_CONFUSION_MATRIX_PATH
)
