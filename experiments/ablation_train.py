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
    train_transform,
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

OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "ablation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


BASELINE_CHECKPOINT = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "efficientnet_b0_best.pt"
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
# ABLATION CONFIGURATIONS
#
# baseline เดิม:
# augmentation=True
# class_weighting=True
#
# เราจะ train อีก 3 configuration
# ================================================================

EXPERIMENTS = [

    {
        "name": "aug_off_weight_off",
        "augmentation": False,
        "class_weighting": False,
    },

    {
        "name": "aug_off_weight_on",
        "augmentation": False,
        "class_weighting": True,
    },

    {
        "name": "aug_on_weight_off",
        "augmentation": True,
        "class_weighting": False,
    },

]


# ================================================================
# RANDOM SEED
# ================================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


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


print("=" * 72)
print("STEP 8 - ABLATION STUDY")
print("=" * 72)

print("Device:", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print(
    "Train samples:",
    len(train_df)
)

print(
    "Val samples:",
    len(val_df)
)

print(
    "Experiments:",
    len(EXPERIMENTS)
)


# ================================================================
# CLASS WEIGHTS
# ================================================================

def calculate_class_weights():

    class_counts = (
        train_df["label"]
        .value_counts()
    )

    num_samples = len(
        train_df
    )

    num_classes = len(
        CLASS_NAMES
    )

    weights = []


    for class_name in CLASS_NAMES:

        count = (
            class_counts[
                class_name
            ]
        )

        weight = (
            num_samples
            /
            (
                num_classes
                * count
            )
        )

        weights.append(
            weight
        )


    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE,
    )


CLASS_WEIGHTS = (
    calculate_class_weights()
)


# ================================================================
# CREATE DATALOADERS
# ================================================================

def create_dataloaders(
    use_augmentation,
):

    # ------------------------------------------------------------
    # Ablation:
    #
    # augmentation ON
    #   → train_transform
    #
    # augmentation OFF
    #   → eval_transform
    #
    # Validation ใช้ eval_transform เสมอ
    # ------------------------------------------------------------

    if use_augmentation:

        selected_train_transform = (
            train_transform
        )

    else:

        selected_train_transform = (
            eval_transform
        )


    train_dataset = CassavaDataset(
        dataframe=train_df,
        root=ROOT,
        transform=selected_train_transform,
    )


    val_dataset = CassavaDataset(
        dataframe=val_df,
        root=ROOT,
        transform=eval_transform,
    )


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


    return (
        train_loader,
        val_loader,
    )


# ================================================================
# CREATE LOSS
# ================================================================

def create_criterion(
    use_class_weighting,
):

    if use_class_weighting:

        criterion = (
            nn.CrossEntropyLoss(
                weight=CLASS_WEIGHTS
            )
        )

    else:

        criterion = (
            nn.CrossEntropyLoss()
        )


    return criterion


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


        progress.set_postfix(

            loss=f"{total_loss / total_samples:.4f}",

            acc=f"{total_correct / total_samples * 100:.2f}%",

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
    experiment,
    epoch,
    model,
    val_metrics,
):

    checkpoint = {

        "experiment":
            experiment["name"],

        "augmentation":
            experiment[
                "augmentation"
            ],

        "class_weighting":
            experiment[
                "class_weighting"
            ],

        "epoch":
            epoch,

        "model_name":
            MODEL_NAME,

        "model_state_dict":
            model.state_dict(),

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
# READ EXISTING BASELINE
# ================================================================

results = []


if BASELINE_CHECKPOINT.exists():

    baseline = torch.load(
        BASELINE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )


    results.append({

        "experiment":
            "aug_on_weight_on",

        "augmentation":
            True,

        "class_weighting":
            True,

        "best_epoch":
            baseline[
                "epoch"
            ],

        "best_val_macro_f1":
            baseline[
                "val_macro_f1"
            ],

        "best_val_accuracy":
            baseline[
                "val_accuracy"
            ],

        "best_val_loss":
            baseline[
                "val_loss"
            ],

        "training_minutes":
            np.nan,

    })


    print()
    print(
        "Baseline loaded:"
    )

    print(
        "aug_on_weight_on"
    )

    print(
        f"Val Macro-F1 = "
        f"{baseline['val_macro_f1']:.4f}"
    )


else:

    print()
    print(
        "WARNING:"
        " baseline checkpoint not found"
    )


# ================================================================
# RUN EACH EXPERIMENT
# ================================================================

for experiment_index, experiment in enumerate(
    EXPERIMENTS,
    start=1,
):

    # ------------------------------------------------------------
    # RESET SEED
    #
    # เพื่อให้ experiment แต่ละตัวเริ่มต้นภายใต้
    # random state เดียวกันมากที่สุด
    # ------------------------------------------------------------

    set_seed(
        SEED
    )


    experiment_name = (
        experiment[
            "name"
        ]
    )


    use_augmentation = (
        experiment[
            "augmentation"
        ]
    )


    use_class_weighting = (
        experiment[
            "class_weighting"
        ]
    )


    experiment_dir = (
        OUTPUT_ROOT
        / experiment_name
    )


    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    history_path = (
        experiment_dir
        / "history.csv"
    )


    best_model_path = (
        experiment_dir
        / "best.pt"
    )


    print()
    print("=" * 72)

    print(
        f"EXPERIMENT "
        f"{experiment_index}/"
        f"{len(EXPERIMENTS)}"
    )

    print("=" * 72)

    print(
        "Name             :",
        experiment_name
    )

    print(
        "Augmentation     :",
        use_augmentation
    )

    print(
        "Class weighting  :",
        use_class_weighting
    )


    # ============================================================
    # DATALOADERS
    # ============================================================

    train_loader, val_loader = (
        create_dataloaders(
            use_augmentation
        )
    )


    # ============================================================
    # MODEL
    # ============================================================

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=len(
            CLASS_NAMES
        ),
    )


    model = model.to(
        DEVICE
    )


    # ============================================================
    # LOSS
    # ============================================================

    criterion = create_criterion(
        use_class_weighting
    )


    # ============================================================
    # OPTIMIZER
    # ============================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # ============================================================
    # SCHEDULER
    # ============================================================

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )
    )


    # ============================================================
    # AMP
    # ============================================================

    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=AMP_INITIAL_SCALE,
        enabled=USE_AMP,
    )


    # ============================================================
    # TRAINING STATE
    # ============================================================

    best_macro_f1 = -1.0

    best_epoch = 0

    best_accuracy = 0.0

    best_loss = 0.0


    history = []


    experiment_start = (
        time.time()
    )


    # ============================================================
    # EPOCH LOOP
    # ============================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        epoch_start = (
            time.time()
        )


        print()
        print(
            "-" * 72
        )

        print(
            f"{experiment_name} "
            f"| Epoch "
            f"{epoch}/{EPOCHS}"
        )

        print(
            "-" * 72
        )


        # --------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------

        train_loss, train_accuracy = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
            )
        )


        # --------------------------------------------------------
        # VALIDATE
        # --------------------------------------------------------

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


        # --------------------------------------------------------
        # LR SCHEDULER
        # --------------------------------------------------------

        scheduler.step(
            val_macro_f1
        )


        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )


        epoch_seconds = (
            time.time()
            - epoch_start
        )


        # --------------------------------------------------------
        # PRINT SUMMARY
        # --------------------------------------------------------

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
            f"{val_loss:.4f}"
        )

        print(
            f"Val accuracy   : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Val macro-F1   : "
            f"{val_macro_f1:.4f}"
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
            "Per-class validation F1:"
        )


        for class_name in CLASS_NAMES:

            class_f1 = (
                val_metrics[
                    "per_class"
                ][
                    class_name
                ][
                    "f1"
                ]
            )


            print(
                f"  "
                f"{class_name:<8}"
                f"{class_f1:.4f}"
            )


        # --------------------------------------------------------
        # HISTORY ROW
        # --------------------------------------------------------

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
                epoch_seconds,
        }


        for class_name in CLASS_NAMES:

            class_metrics = (
                val_metrics[
                    "per_class"
                ][
                    class_name
                ]
            )


            history_row[
                f"{class_name}_precision"
            ] = (
                class_metrics[
                    "precision"
                ]
            )


            history_row[
                f"{class_name}_recall"
            ] = (
                class_metrics[
                    "recall"
                ]
            )


            history_row[
                f"{class_name}_f1"
            ] = (
                class_metrics[
                    "f1"
                ]
            )


        history.append(
            history_row
        )


        pd.DataFrame(
            history
        ).to_csv(
            history_path,
            index=False,
        )


        # --------------------------------------------------------
        # BEST MODEL
        # --------------------------------------------------------

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

            best_accuracy = (
                val_accuracy
            )

            best_loss = (
                val_loss
            )


            save_checkpoint(
                best_model_path,
                experiment,
                epoch,
                model,
                val_metrics,
            )


            print(
                "✅ New best "
                f"{experiment_name}: "
                f"Macro-F1="
                f"{best_macro_f1:.4f}"
            )


    # ============================================================
    # EXPERIMENT FINISHED
    # ============================================================

    experiment_seconds = (
        time.time()
        - experiment_start
    )


    experiment_minutes = (
        experiment_seconds
        / 60
    )


    print()
    print("=" * 72)

    print(
        f"FINISHED: "
        f"{experiment_name}"
    )

    print("=" * 72)

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        f"Best Val Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Best Val Accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Training time: "
        f"{experiment_minutes:.2f} min"
    )


    results.append({

        "experiment":
            experiment_name,

        "augmentation":
            use_augmentation,

        "class_weighting":
            use_class_weighting,

        "best_epoch":
            best_epoch,

        "best_val_macro_f1":
            best_macro_f1,

        "best_val_accuracy":
            best_accuracy,

        "best_val_loss":
            best_loss,

        "training_minutes":
            experiment_minutes,
    })


    # ------------------------------------------------------------
    # Save running summary หลังจบแต่ละ experiment
    # เผื่อเครื่องดับ/หยุดกลางทาง
    # ------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        OUTPUT_ROOT
        / "ablation_summary.csv",
        index=False,
    )


    # ------------------------------------------------------------
    # Clear GPU cache
    # ------------------------------------------------------------

    del model

    del optimizer

    del scheduler

    del scaler

    del criterion

    del train_loader

    del val_loader


    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()


# ================================================================
# FINAL SUMMARY
# ================================================================

results_df = pd.DataFrame(
    results
)


results_df = (
    results_df
    .sort_values(
        "best_val_macro_f1",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


summary_path = (
    OUTPUT_ROOT
    / "ablation_summary.csv"
)


results_df.to_csv(
    summary_path,
    index=False,
)


print()
print("=" * 72)
print("ABLATION STUDY FINISHED")
print("=" * 72)

print()

print(
    results_df.to_string(
        index=False
    )
)


print()
print(
    "Summary saved:",
    summary_path
)


best_row = (
    results_df.iloc[0]
)


print()
print(
    "BEST CONFIGURATION"
)

print(
    "Experiment       :",
    best_row[
        "experiment"
    ]
)

print(
    "Augmentation     :",
    best_row[
        "augmentation"
    ]
)

print(
    "Class weighting  :",
    best_row[
        "class_weighting"
    ]
)

print(
    f"Val Macro-F1     : "
    f"{best_row['best_val_macro_f1']:.4f}"
)
