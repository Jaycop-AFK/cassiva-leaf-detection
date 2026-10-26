from pathlib import Path

import pandas as pd
import torch


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parent.parent

ABLATION_DIR = (
    ROOT
    / "outputs"
    / "ablation"
)

SUMMARY_PATH = (
    ABLATION_DIR
    / "ablation_summary.csv"
)


EXPERIMENTS = [
    "aug_off_weight_off",
    "aug_off_weight_on",
    "aug_on_weight_off",
    "aug_on_weight_on",
]


# ================================================================
# LOAD SUMMARY
# ================================================================

print("=" * 80)
print("ABLATION CHECKPOINT VERIFICATION")
print("=" * 80)

print()

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(
        f"Summary not found:\n{SUMMARY_PATH}"
    )


summary = pd.read_csv(
    SUMMARY_PATH
)


print("Summary:")
print(
    summary.to_string(
        index=False
    )
)


# ================================================================
# CHECK EACH EXPERIMENT
# ================================================================

print()
print("=" * 80)
print("CHECKPOINTS")
print("=" * 80)


for experiment in EXPERIMENTS:

    print()
    print("-" * 80)
    print(experiment)
    print("-" * 80)

    checkpoint_path = (
        ABLATION_DIR
        / experiment
        / "best.pt"
    )


    if not checkpoint_path.exists():

        print(
            "Checkpoint: NOT FOUND"
        )

        continue


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )


    print(
        "Checkpoint path :",
        checkpoint_path
    )

    print(
        "Epoch           :",
        checkpoint.get(
            "epoch"
        )
    )

    print(
        "Val Macro-F1    :",
        checkpoint.get(
            "val_macro_f1"
        )
    )

    print(
        "Val Accuracy    :",
        checkpoint.get(
            "val_accuracy"
        )
    )

    print(
        "Augmentation    :",
        checkpoint.get(
            "augmentation"
        )
    )

    print(
        "Class weighting :",
        checkpoint.get(
            "class_weighting"
        )
    )


    # ============================================================
    # COMPARE AGAINST SUMMARY
    # ============================================================

    match = summary[
        summary[
            "experiment"
        ]
        == experiment
    ]


    if len(match) == 0:

        print(
            "Summary row     : NOT FOUND"
        )

        continue


    row = match.iloc[0]


    expected_epoch = (
        row[
            "best_epoch"
        ]
    )

    expected_f1 = (
        row[
            "best_val_macro_f1"
        ]
    )


    actual_epoch = (
        checkpoint.get(
            "epoch"
        )
    )

    actual_f1 = (
        checkpoint.get(
            "val_macro_f1"
        )
    )


    epoch_match = (
        int(actual_epoch)
        == int(expected_epoch)
    )


    f1_match = (
        abs(
            float(actual_f1)
            - float(expected_f1)
        )
        < 1e-6
    )


    print()
    print(
        "Expected epoch  :",
        expected_epoch
    )

    print(
        "Expected F1     :",
        expected_f1
    )

    print(
        "Epoch matches   :",
        epoch_match
    )

    print(
        "F1 matches      :",
        f1_match
    )


# ================================================================
# FINISHED
# ================================================================

print()
print("=" * 80)
print("VERIFICATION FINISHED")
print("=" * 80)
