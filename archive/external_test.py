from pathlib import Path
import sys
import argparse

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
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from dataset import (
    eval_transform,
    CLASS_NAMES,
)


# ================================================================
# CONFIG
# ================================================================


CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "ablation"
    / "aug_off_weight_off_rerun"
    / "best.pt"
)

DEFAULT_EXTERNAL_ROOT = (
    ROOT
    / "external_test"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "external_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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


THAI_NAMES = {
    "cbb": "ใบไหม้ (Cassava Bacterial Blight)",
    "cbs": "ใบจุดสีน้ำตาล (Brown Leaf Spot)",
    "cmd": "ใบด่าง (Cassava Mosaic Disease)",
    "healthy": "ใบปกติ (Healthy)",
}


# ================================================================
# ARGUMENTS
# ================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "External evaluation for "
            "Cassava Leaf Disease Classification"
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_EXTERNAL_ROOT),
        help="External test root directory",
    )

    return parser.parse_args()


# ================================================================
# LOAD MODEL
# ================================================================

def load_model():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


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
        num_classes=len(
            class_names
        ),
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
    print("EXTERNAL TEST")
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
        "Experiment:",
        checkpoint.get(
            "experiment"
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
# PREDICT ONE IMAGE
# ================================================================

def predict_image(
    model,
    class_names,
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
                DEVICE.type
                == "cuda"
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


    pred_label = (
        class_names[
            pred_idx
        ]
    )


    confidence = float(
        probabilities[
            pred_idx
        ].item()
    )


    probabilities = (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )


    return (
        pred_idx,
        pred_label,
        confidence,
        probabilities,
    )


# ================================================================
# LABELED EXTERNAL TEST
# ================================================================

def evaluate_labeled(
    model,
    class_names,
    external_root,
):

    labeled_root = (
        external_root
        / "labeled"
    )


    rows = []


    print()
    print("=" * 72)
    print("LABELED EXTERNAL DATA")
    print("=" * 72)


    for true_idx, true_label in enumerate(
        class_names
    ):

        class_dir = (
            labeled_root
            / true_label
        )


        image_files = find_images(
            class_dir
        )


        print(
            f"{true_label:<8}: "
            f"{len(image_files)} images"
        )


        for image_path in image_files:

            (
                pred_idx,
                pred_label,
                confidence,
                probabilities,
            ) = predict_image(
                model,
                class_names,
                image_path,
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
                    probabilities[idx]
                )


            rows.append(
                row
            )


    if len(rows) == 0:

        print()
        print(
            "No labeled external images found."
        )

        return None


    results_df = pd.DataFrame(
        rows
    )


    results_path = (
        OUTPUT_DIR
        / "external_labeled_predictions.csv"
    )


    results_df.to_csv(
        results_path,
        index=False,
    )


    # ============================================================
    # METRICS
    # ============================================================

    y_true = (
        results_df[
            "true_idx"
        ]
        .to_numpy()
    )


    y_pred = (
        results_df[
            "pred_idx"
        ]
        .to_numpy()
    )


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


    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(
                range(
                    len(class_names)
                )
            ),
            zero_division=0,
        )
    )


    print()
    print("=" * 72)
    print("EXTERNAL LABELED RESULTS")
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


    print()
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(
                range(
                    len(class_names)
                )
            ),
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )


    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                len(class_names)
            )
        ),
    )


    print(
        "Confusion matrix:"
    )

    print(
        cm
    )


    fig, ax = plt.subplots(
        figsize=(
            8,
            7,
        )
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
        "External Test Confusion Matrix"
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
        / "external_confusion_matrix.png"
    )


    plt.savefig(
        confusion_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print()
    print(
        "Predictions:",
        results_path
    )

    print(
        "Confusion matrix:",
        confusion_path
    )


    return results_df


# ================================================================
# UNLABELED EXTERNAL IMAGES
# ================================================================

def predict_unlabeled(
    model,
    class_names,
    external_root,
):

    unlabeled_root = (
        external_root
        / "unlabeled"
    )


    image_files = find_images(
        unlabeled_root
    )


    print()
    print("=" * 72)
    print("UNLABELED EXTERNAL DATA")
    print("=" * 72)

    print(
        "Images:",
        len(image_files)
    )


    if len(image_files) == 0:

        print(
            "No unlabeled images found."
        )

        return None


    rows = []


    for image_path in image_files:

        (
            pred_idx,
            pred_label,
            confidence,
            probabilities,
        ) = predict_image(
            model,
            class_names,
            image_path,
        )


        row = {

            "filepath":
                image_path.relative_to(
                    ROOT
                ).as_posix(),

            "pred_label":
                pred_label,

            "confidence":
                confidence,
        }


        for idx, class_name in enumerate(
            class_names
        ):

            row[
                f"prob_{class_name}"
            ] = float(
                probabilities[idx]
            )


        rows.append(
            row
        )


        print(
            f"{image_path.name:<35} "
            f"→ "
            f"{pred_label:<8} "
            f"{confidence * 100:6.2f}%"
        )


    results_df = pd.DataFrame(
        rows
    )


    output_path = (
        OUTPUT_DIR
        / "external_unlabeled_predictions.csv"
    )


    results_df.to_csv(
        output_path,
        index=False,
    )


    print()
    print(
        "Saved:",
        output_path
    )


    return results_df


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()


    external_root = Path(
        args.root
    )


    if not external_root.is_absolute():

        external_root = (
            ROOT
            / external_root
        )


    print(
        "External root:",
        external_root
    )


    model, class_names = (
        load_model()
    )


    evaluate_labeled(
        model,
        class_names,
        external_root,
    )


    predict_unlabeled(
        model,
        class_names,
        external_root,
    )


    print()
    print("=" * 72)
    print("EXTERNAL TEST FINISHED")
    print("=" * 72)


if __name__ == "__main__":
    main()
