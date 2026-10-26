from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = (
    ROOT
    / "outputs"
    / "evaluation"
    / "test_predictions.csv"
)

CLASS_NAMES = [
    "cbb",
    "cbs",
    "cmd",
    "healthy",
]


# ================================================================
# LOAD PREDICTIONS
# ================================================================

df = pd.read_csv(
    PREDICTIONS_PATH
)


print("=" * 70)
print("STEP 7C - SOURCE BIAS ANALYSIS")
print("=" * 70)

print(
    "Rows:",
    len(df)
)

print()

print(
    "Sources:"
)

print(
    df["source"]
    .value_counts()
)


# ================================================================
# SOURCE EVALUATION FUNCTION
# ================================================================

def evaluate_source(
    source_name,
    source_df,
):

    true_labels = (
        source_df[
            "true_idx"
        ]
        .to_numpy()
    )

    pred_labels = (
        source_df[
            "pred_idx"
        ]
        .to_numpy()
    )


    accuracy = accuracy_score(
        true_labels,
        pred_labels,
    )


    macro_f1 = f1_score(
        true_labels,
        pred_labels,
        average="macro",
        zero_division=0,
    )


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            true_labels,
            pred_labels,
            labels=list(
                range(
                    len(CLASS_NAMES)
                )
            ),
            zero_division=0,
        )
    )


    print()
    print("=" * 70)
    print(
        f"SOURCE: {source_name.upper()}"
    )
    print("=" * 70)

    print(
        "Samples:",
        len(source_df)
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1 : {macro_f1:.4f}"
    )


    print()
    print("Per-class metrics")
    print("-" * 70)


    for idx, class_name in enumerate(
        CLASS_NAMES
    ):

        if support[idx] == 0:

            print(
                f"{class_name:<8} | "
                "not available in this source"
            )

            continue


        print(
            f"{class_name:<8} | "
            f"P={precision[idx]:.4f} | "
            f"R={recall[idx]:.4f} | "
            f"F1={f1[idx]:.4f} | "
            f"N={support[idx]}"
        )


    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(
        true_labels,
        pred_labels,
        labels=list(
            range(
                len(CLASS_NAMES)
            )
        ),
    )


    print()
    print("Confusion matrix:")
    print(cm)


    # ============================================================
    # ERROR SUMMARY
    # ============================================================

    wrong_df = source_df[
        source_df["correct"] == False
    ].copy()


    print()
    print(
        "Correct:",
        source_df["correct"].sum()
    )

    print(
        "Wrong  :",
        len(wrong_df)
    )


    if len(wrong_df) > 0:

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


        print()
        print("Most common mistakes:")

        print(
            mistakes.head(10)
        )


# ================================================================
# ANALYZE EACH SOURCE
# ================================================================

for source_name in sorted(
    df["source"].unique()
):

    source_df = df[
        df["source"] == source_name
    ].copy()

    evaluate_source(
        source_name,
        source_df,
    )


# ================================================================
# CLASS + SOURCE ACCURACY
# ================================================================

print()
print("=" * 70)
print("ACCURACY BY CLASS AND SOURCE")
print("=" * 70)


summary = (
    df
    .groupby(
        [
            "source",
            "label",
        ]
    )
    .agg(
        samples=(
            "correct",
            "size"
        ),
        correct=(
            "correct",
            "sum"
        ),
        accuracy=(
            "correct",
            "mean"
        ),
        mean_confidence=(
            "confidence",
            "mean"
        ),
    )
    .reset_index()
)


summary[
    "accuracy"
] *= 100


print(
    summary.to_string(
        index=False
    )
)


# ================================================================
# SAVE SOURCE ANALYSIS
# ================================================================

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "evaluation"
    / "source_analysis.csv"
)


summary.to_csv(
    OUTPUT_PATH,
    index=False,
)


print()
print("=" * 70)
print("SOURCE ANALYSIS FINISHED")
print("=" * 70)

print(
    "Saved:",
    OUTPUT_PATH
)
