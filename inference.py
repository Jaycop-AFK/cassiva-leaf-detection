from pathlib import Path
import argparse

import torch
import timm

from PIL import Image

from dataset import (
    eval_transform,
    CLASS_NAMES,
)


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(__file__).resolve().parent

CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "ablation"
    / "aug_off_weight_off_rerun"
    / "best.pt"
)

MODEL_NAME = "efficientnet_b0"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# THAI LABELS
# ================================================================

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
            "Cassava Leaf Disease Classification"
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )

    return parser.parse_args()


# ================================================================
# LOAD CHECKPOINT
# ================================================================

def load_model():

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    print("=" * 70)
    print("LOAD MODEL")
    print("=" * 70)

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Device:",
        DEVICE
    )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )


    checkpoint_classes = checkpoint.get(
        "class_names",
        CLASS_NAMES,
    )


    print(
        "Checkpoint experiment:",
        checkpoint.get(
            "experiment",
            "unknown"
        )
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get(
            "epoch",
            "unknown"
        )
    )

    print(
        "Validation Macro-F1:",
        checkpoint.get(
            "val_macro_f1",
            "unknown"
        )
    )

    print(
        "Validation Accuracy:",
        checkpoint.get(
            "val_accuracy",
            "unknown"
        )
    )

    print(
        "Augmentation:",
        checkpoint.get(
            "augmentation",
            "unknown"
        )
    )

    print(
        "Class weighting:",
        checkpoint.get(
            "class_weighting",
            "unknown"
        )
    )

    print(
        "Classes:",
        checkpoint_classes
    )


    model = timm.create_model(
        checkpoint.get(
            "model_name",
            MODEL_NAME,
        ),
        pretrained=False,
        num_classes=len(
            checkpoint_classes
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


    print(
        "Model loaded successfully"
    )


    return (
        model,
        checkpoint_classes,
    )


# ================================================================
# LOAD IMAGE
# ================================================================

def load_image(
    image_path,
):

    image_path = Path(
        image_path
    )


    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found:\n"
            f"{image_path}"
        )


    if not image_path.is_file():

        raise ValueError(
            f"Not a file:\n"
            f"{image_path}"
        )


    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )


        original_size = (
            image.size
        )


        tensor = eval_transform(
            image
        )


    tensor = (
        tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    return (
        image_path,
        original_size,
        tensor,
    )


# ================================================================
# PREDICT
# ================================================================

def predict(
    model,
    image_tensor,
    class_names,
):

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
                image_tensor
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


    return (
        pred_idx,
        pred_label,
        confidence,
        probabilities
        .detach()
        .cpu(),
    )


# ================================================================
# PRINT RESULT
# ================================================================

def print_result(
    image_path,
    original_size,
    pred_label,
    confidence,
    probabilities,
    class_names,
):

    print()
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        "Image:",
        image_path
    )

    print(
        "Original size:",
        original_size
    )

    print()

    print(
        "Prediction :",
        pred_label
    )

    print(
        "Thai       :",
        THAI_NAMES.get(
            pred_label,
            pred_label
        )
    )

    print(
        "Confidence :",
        f"{confidence * 100:.2f}%"
    )


    print()
    print("Probabilities")
    print("-" * 70)


    rows = []


    for idx, class_name in enumerate(
        class_names
    ):

        probability = float(
            probabilities[
                idx
            ].item()
        )


        rows.append(
            (
                class_name,
                probability,
            )
        )


    rows.sort(
        key=lambda x: x[1],
        reverse=True,
    )


    for class_name, probability in rows:

        print(
            f"{class_name:<8} "
            f"{probability * 100:>7.2f}% "
            f"| "
            f"{THAI_NAMES.get(class_name, class_name)}"
        )


# ================================================================
# MAIN
# ================================================================

def main():

    args = parse_args()


    model, class_names = (
        load_model()
    )


    (
        image_path,
        original_size,
        image_tensor,
    ) = load_image(
        args.image
    )


    (
        pred_idx,
        pred_label,
        confidence,
        probabilities,
    ) = predict(
        model,
        image_tensor,
        class_names,
    )


    print_result(
        image_path=image_path,
        original_size=original_size,
        pred_label=pred_label,
        confidence=confidence,
        probabilities=probabilities,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()