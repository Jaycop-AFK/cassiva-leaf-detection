from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import timm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from dataset import (
    eval_transform,
    CLASS_NAMES,
)


# ================================================================
# CONFIG
# ================================================================


EXTERNAL_ROOT = (
    ROOT
    / "external_test_leaf_only"
    / "labeled"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "external_test_leaf_only"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "ablation"
    / "aug_off_weight_off_rerun"
    / "best.pt"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    class_names = checkpoint.get(
        "class_names",
        CLASS_NAMES,
    )

    model = timm.create_model(
        checkpoint.get(
            "model_name",
            "efficientnet_b0",
        ),
        pretrained=False,
        num_classes=len(class_names),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print("=" * 72)
    print("LEAF-ONLY EXTERNAL TEST")
    print("=" * 72)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch"
        )
    )

    print(
        "Validation Macro-F1:",
        checkpoint.get(
            "val_macro_f1"
        )
    )

    print(
        "Classes:",
        class_names
    )

    return (
        model,
        class_names,
    )


# ================================================================
# FIND IMAGES
# ================================================================

def find_images(
    directory,
):

    if not directory.exists():
        return []

    return sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


# ================================================================
# PREDICT
# ================================================================

def predict_image(
    model,
    image_path,
):

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        tensor = eval_transform(
            image
        )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():

        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(
                DEVICE.type == "cuda"
            ),
        ):

            logits = model(
                tensor
            )

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )[0]

    pred_idx = int(
        probabilities.argmax().item()
    )

    confidence = float(
        probabilities[
            pred_idx
        ].item()
    )

    return (
        pred_idx,
        confidence,
        probabilities
        .cpu()
        .numpy(),
    )


# ================================================================
# MAIN
# ================================================================

def main():

    model, class_names = (
        load_model()
    )

    rows = []

    print()
    print("=" * 72)
    print("DATASET")
    print("=" * 72)

    for true_idx, true_label in enumerate(
        class_names
    ):

        class_dir = (
            EXTERNAL_ROOT
            / true_label
        )

        images = find_images(
            class_dir
        )

        print(
            f"{true_label:<8}: "
            f"{len(images)} images"
        )

        for image_path in images:

            (
                pred_idx,
                confidence,
                probabilities,
            ) = predict_image(
                model,
                image_path,
            )

            pred_label = (
                class_names[
                    pred_idx
                ]
            )

            row = {

                "filepath":
                    image_path.relative_to(
                        ROOT
                    ).as_posix(),

                "true_label":
                    true_label,

                "true_idx":
                    true_idx,

                "pred_label":
                    pred_label,

                "pred_idx":
                    pred_idx,

                "confidence":
                    confidence,

                "correct":
                    true_idx
                    == pred_idx,
            }

            for idx, class_name in enumerate(
                class_names
            ):

                row[
                    f"prob_{class_name}"
                ] = float(
                    probabilities[
                        idx
                    ]
                )

            rows.append(
                row
            )

    if len(rows) == 0:

        print()
        print(
            "No images found."
        )

        return

    results_df = pd.DataFrame(
        rows
    )

    predictions_path = (
        OUTPUT_DIR
        / "leaf_only_predictions.csv"
    )

    results_df.to_csv(
        predictions_path,
        index=False,
    )

    # ============================================================
    # METRICS
    # ================================================================

    y_true = results_df[
        "true_idx"
    ].to_numpy()

    y_pred = results_df[
        "pred_idx"
    ].to_numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(
            range(
                len(class_names)
            )
        ),
        zero_division=0,
    )

    print()
    print("=" * 72)
    print("LEAF-ONLY EXTERNAL RESULTS")
    print("=" * 72)

    print(
        "Samples:",
        len(results_df)
    )

    print(
        f"Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1 : "
        f"{macro_f1:.4f}"
    )

    print()
    print(
        "Per-class metrics"
    )

    print(
        "-" * 72
    )

    for idx, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_name:<8} | "
            f"P={precision[idx]:.4f} | "
            f"R={recall[idx]:.4f} | "
            f"F1={f1[idx]:.4f} | "
            f"N={support[idx]}"
        )

    # ============================================================
    # CONFUSION MATRIX
    # ================================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                len(class_names)
            )
        ),
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        cm
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        cm
    )

    ax.set_xticks(
        np.arange(
            len(class_names)
        )
    )

    ax.set_yticks(
        np.arange(
            len(class_names)
        )
    )

    ax.set_xticklabels(
        class_names
    )

    ax.set_yticklabels(
        class_names
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_title(
        "Leaf-only External Confusion Matrix"
    )

    for i in range(
        len(class_names)
    ):

        for j in range(
            len(class_names)
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

    confusion_path = (
        OUTPUT_DIR
        / "leaf_only_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # ============================================================
    # WRONG PREDICTIONS
    # ================================================================

    wrong_df = (
        results_df[
            results_df[
                "correct"
            ] == False
        ]
        .copy()
    )

    wrong_path = (
        OUTPUT_DIR
        / "leaf_only_wrong_predictions.csv"
    )

    wrong_df.to_csv(
        wrong_path,
        index=False,
    )

    print()
    print("=" * 72)
    print("ERROR SUMMARY")
    print("=" * 72)

    print(
        "Correct:",
        int(
            results_df[
                "correct"
            ].sum()
        )
    )

    print(
        "Wrong:",
        len(
            wrong_df
        )
    )

    print()
    print(
        "Predictions:",
        predictions_path
    )

    print(
        "Wrong predictions:",
        wrong_path
    )

    print(
        "Confusion matrix:",
        confusion_path
    )

    print()
    print("=" * 72)
    print("LEAF-ONLY EXTERNAL TEST FINISHED")
    print("=" * 72)


if __name__ == "__main__":
    main()
