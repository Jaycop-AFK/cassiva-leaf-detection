from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import timm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

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

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "external_gradcam"
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


# ================================================================
# POSSIBLE PREDICTION CSV LOCATIONS
# ================================================================

PREDICTION_CANDIDATES = [
    ROOT
    / "outputs"
    / "external_test_leaf_only"
    / "leaf_only_predictions.csv",

    ROOT
    / "outputs"
    / "external_test_leaf_only"
    / "predictions.csv",

    ROOT
    / "external_test_leaf_only"
    / "predictions.csv",
]


# ================================================================
# FIND PREDICTIONS CSV
# ================================================================

def find_predictions_csv():

    for path in PREDICTION_CANDIDATES:

        if path.exists():
            return path

    candidates = list(
        ROOT.rglob("*predictions*.csv")
    )

    candidates = [
        path
        for path in candidates
        if "leaf" in str(path).lower()
    ]

    if len(candidates) == 0:

        raise FileNotFoundError(
            "Leaf-only predictions CSV not found."
        )

    return candidates[0]


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
    print("EXTERNAL GRAD-CAM")
    print("=" * 72)

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Epoch:",
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
        "Device:",
        DEVICE
    )

    print(
        "Classes:",
        class_names
    )

    print(
        "Target layer:",
        model.conv_head
    )

    return (
        model,
        class_names,
    )


# ================================================================
# GRAD-CAM CLASS
# ================================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            self.target_layer
            .register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer
            .register_full_backward_hook(
                self._backward_hook
            )
        )


    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = (
            output.detach()
        )


    def _backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = (
            grad_output[0].detach()
        )


    def generate(
        self,
        image_tensor,
        target_idx,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        logits = self.model(
            image_tensor
        )

        score = logits[
            0,
            target_idx
        ]

        score.backward()


        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )


        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )


        weights = (
            self.gradients
            .mean(
                dim=(2, 3),
                keepdim=True,
            )
        )


        cam = (
            weights
            * self.activations
        ).sum(
            dim=1,
            keepdim=True,
        )


        cam = F.relu(
            cam
        )


        cam = F.interpolate(
            cam,
            size=(
                image_tensor.shape[2],
                image_tensor.shape[3],
            ),
            mode="bilinear",
            align_corners=False,
        )


        cam = cam[
            0,
            0
        ]


        cam_min = float(
            cam.min().item()
        )

        cam_max = float(
            cam.max().item()
        )


        if (
            cam_max - cam_min
        ) > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = torch.zeros_like(
                cam
            )


        return (
            cam
            .detach()
            .cpu()
            .numpy()
        )


    def close(
        self,
    ):

        self.forward_handle.remove()

        self.backward_handle.remove()


# ================================================================
# RESOLVE IMAGE PATH
# ================================================================

def resolve_image(
    path_value,
):

    raw_path = Path(
        str(path_value)
    )


    # Absolute or already valid path
    if raw_path.exists():

        return raw_path


    # Relative to project root
    project_path = (
        ROOT
        / raw_path
    )


    if project_path.exists():

        return project_path


    # Search by filename
    matches = []

    for path in ROOT.rglob(
        raw_path.name
    ):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            matches.append(
                path
            )


    if len(matches) == 0:

        raise FileNotFoundError(
            f"Cannot find image:\n"
            f"{path_value}"
        )


    return matches[0]


# ================================================================
# LOAD IMAGE
# ================================================================

def load_image(
    image_path,
):

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        original_image = (
            image.copy()
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
        original_image,
        tensor,
    )


# ================================================================
# CREATE DISPLAY IMAGE
#
# dataset.py eval_transform:
# Resize((256, 256))
# CenterCrop(224)
#
# ทำเหมือน preprocessing ของ model เพื่อให้ CAM ตรงกับ input
# ================================================================

def create_model_input_display(
    image,
):

    resized = image.resize(
        (256, 256)
    )

    left = (
        256 - 224
    ) // 2

    top = (
        256 - 224
    ) // 2

    right = (
        left + 224
    )

    bottom = (
        top + 224
    )

    cropped = resized.crop(
        (
            left,
            top,
            right,
            bottom,
        )
    )

    image_array = np.asarray(
        cropped
    ).astype(
        np.float32
    ) / 255.0


    return image_array


# ================================================================
# OVERLAY
# ================================================================

def make_overlay(
    image_array,
    cam,
):

    heatmap = plt.cm.jet(
        cam
    )[
        :,
        :,
        :3
    ]


    overlay = (
        0.55
        * image_array
        + 0.45
        * heatmap
    )


    overlay = np.clip(
        overlay,
        0.0,
        1.0,
    )


    return overlay


# ================================================================
# NORMAL MODEL PREDICTION
# ================================================================

def predict(
    model,
    tensor,
):

    with torch.inference_mode():

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
        .detach()
        .cpu()
        .numpy(),
    )


# ================================================================
# GENERATE ONE GRAD-CAM FIGURE
# ================================================================

def generate_figure(
    model,
    gradcam,
    class_names,
    image_path,
    true_label,
    case_name,
):

    (
        original_image,
        tensor,
    ) = load_image(
        image_path
    )


    (
        pred_idx,
        confidence,
        probabilities,
    ) = predict(
        model,
        tensor,
    )


    pred_label = (
        class_names[
            pred_idx
        ]
    )


    true_idx = (
        class_names.index(
            true_label
        )
    )


    # Grad-CAM for predicted class
    predicted_cam = (
        gradcam.generate(
            tensor,
            pred_idx,
        )
    )


    # Grad-CAM for true class
    true_cam = (
        gradcam.generate(
            tensor,
            true_idx,
        )
    )


    image_array = (
        create_model_input_display(
            original_image
        )
    )


    predicted_overlay = (
        make_overlay(
            image_array,
            predicted_cam,
        )
    )


    true_overlay = (
        make_overlay(
            image_array,
            true_cam,
        )
    )


    # ============================================================
    # FIGURE
    # ================================================================

    fig = plt.figure(
        figsize=(
            13,
            4.5,
        )
    )


    ax1 = fig.add_subplot(
        1,
        3,
        1,
    )

    ax1.imshow(
        image_array
    )

    ax1.set_title(
        "Model input"
    )

    ax1.axis(
        "off"
    )


    ax2 = fig.add_subplot(
        1,
        3,
        2,
    )

    ax2.imshow(
        predicted_overlay
    )

    ax2.set_title(
        f"Predicted CAM\n"
        f"{pred_label} "
        f"({confidence:.3f})"
    )

    ax2.axis(
        "off"
    )


    ax3 = fig.add_subplot(
        1,
        3,
        3,
    )

    ax3.imshow(
        true_overlay
    )

    ax3.set_title(
        f"True-class CAM\n"
        f"{true_label}"
    )

    ax3.axis(
        "off"
    )


    fig.suptitle(
        f"{case_name}\n"
        f"true={true_label} | "
        f"pred={pred_label}"
    )


    plt.tight_layout()


    safe_name = (
        case_name
        .replace(
            "/",
            "_"
        )
        .replace(
            "\\",
            "_"
        )
        .replace(
            " ",
            "_"
        )
    )


    output_path = (
        OUTPUT_DIR
        / f"{safe_name}.png"
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    result = {

        "case":
            case_name,

        "filepath":
            str(
                image_path
            ),

        "true_label":
            true_label,

        "pred_label":
            pred_label,

        "confidence":
            confidence,
    }


    for idx, class_name in enumerate(
        class_names
    ):

        result[
            f"prob_{class_name}"
        ] = float(
            probabilities[
                idx
            ]
        )


    result[
        "gradcam_path"
    ] = str(
        output_path
    )


    return result


# ================================================================
# FIND NAMED IMAGE
# ================================================================

def find_named_images(
    name,
):

    matches = []


    for path in ROOT.rglob(
        "*"
    ):

        if not path.is_file():
            continue


        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue


        # Ignore Grad-CAM output itself
        if (
            OUTPUT_DIR
            in path.parents
        ):
            continue


        if (
            name.lower()
            in path.stem.lower()
        ):

            matches.append(
                path
            )


    return sorted(
        matches
    )


# ================================================================
# CLASSIFY IMAGE VERSION
# ================================================================

def classify_version(
    path,
):

    path_text = (
        str(path)
        .lower()
        .replace(
            "\\",
            "/"
        )
    )


    if (
        "external_test_leaf_only"
        in path_text
    ):

        return "leaf_only"


    if (
        "/external_test/"
        in path_text
    ):

        return "original"


    return "other"


# ================================================================
# MAIN
# ================================================================

def main():

    model, class_names = (
        load_model()
    )


    gradcam = GradCAM(
        model=model,
        target_layer=model.conv_head,
    )


    prediction_path = (
        find_predictions_csv()
    )


    print()
    print(
        "Predictions CSV:",
        prediction_path
    )


    predictions_df = pd.read_csv(
        prediction_path
    )


    cases = []


    # ============================================================
    # CBB_04 AND CBB_08
    #
    # หา original และ leaf-only version
    # ================================================================

    for image_name in [
        "cbb_04",
        "cbb_08",
    ]:

        matches = find_named_images(
            image_name
        )


        print()
        print(
            f"Found {image_name}: "
            f"{len(matches)} file(s)"
        )


        for path in matches:

            version = classify_version(
                path
            )


            # สนใจเฉพาะ original / leaf-only
            if version not in {
                "original",
                "leaf_only",
            }:

                continue


            print(
                " ",
                version,
                "->",
                path
            )


            cases.append(
                {
                    "image_path":
                        path,

                    "true_label":
                        "cbb",

                    "case_name":
                        (
                            f"{image_name}_"
                            f"{version}"
                        ),
                }
            )


    # ============================================================
    # SELECT TOP 3 WRONG CBS
    # ================================================================

    required_columns = {
        "filepath",
        "true_label",
        "pred_label",
        "confidence",
    }


    if required_columns.issubset(
        predictions_df.columns
    ):

        wrong_cbs = predictions_df[
            (predictions_df["true_label"] == "cbs")
            & (predictions_df["pred_label"] != "cbs")
        ].copy()
        
        priorrity = wrong_cbs[
             wrong_cbs["filepath"]
    .astype(str)
    .str.contains(
        "cbs_04",
        case=False,
        na=False,
    )
        ]
        
        


        print()
        print("=" * 72)
        print("SELECTED WRONG CBS")
        print("=" * 72)


        if len(wrong_cbs) == 0:

            print(
                "No wrong CBS images found."
            )


        for _, row in wrong_cbs.iterrows():

            image_path = resolve_image(
                row[
                    "filepath"
                ]
            )


            print(
                image_path.name,
                "| true=cbs",
                f"| pred={row['pred_label']}",
                f"| confidence={row['confidence']:.4f}",
            )


            cases.append(
                {
                    "image_path":
                        image_path,

                    "true_label":
                        "cbs",

                    "case_name":
                        (
                            "cbs_wrong_"
                            + image_path.stem
                        ),
                }
            )


    else:

        print()
        print(
            "WARNING: predictions CSV "
            "does not contain required columns."
        )


    # ================================================================
    # REMOVE DUPLICATE CASES
    # ================================================================

    unique_cases = []

    seen = set()


    for case in cases:

        key = (
            str(
                case[
                    "image_path"
                ].resolve()
            ),
            case[
                "true_label"
            ],
        )


        if key in seen:
            continue


        seen.add(
            key
        )

        unique_cases.append(
            case
        )


    # ================================================================
    # GENERATE CAMS
    # ================================================================

    print()
    print("=" * 72)
    print("GENERATING EXTERNAL GRAD-CAM")
    print("=" * 72)

    print(
        "Cases:",
        len(unique_cases)
    )


    results = []


    for index, case in enumerate(
        unique_cases,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(unique_cases)}] "
            f"{case['case_name']}"
        )

        print(
            "Image:",
            case[
                "image_path"
            ]
        )


        result = generate_figure(
            model=model,
            gradcam=gradcam,
            class_names=class_names,
            image_path=case[
                "image_path"
            ],
            true_label=case[
                "true_label"
            ],
            case_name=case[
                "case_name"
            ],
        )


        results.append(
            result
        )


        print(
            "Prediction:",
            result[
                "pred_label"
            ]
        )

        print(
            "Confidence:",
            f"{result['confidence']:.4f}"
        )


    gradcam.close()


    # ================================================================
    # SAVE SUMMARY
    # ================================================================

    summary_path = (
        OUTPUT_DIR
        / "external_gradcam_summary.csv"
    )


    results_df = pd.DataFrame(
        results
    )


    results_df.to_csv(
        summary_path,
        index=False,
    )


    # ================================================================
    # FINAL OUTPUT
    # ================================================================

    print()
    print("=" * 72)
    print("EXTERNAL GRAD-CAM FINISHED")
    print("=" * 72)

    print(
        "Cases generated:",
        len(results)
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print(
        "Summary CSV:",
        summary_path
    )


    if len(results_df) > 0:

        print()
        print(
            results_df[
                [
                    "case",
                    "true_label",
                    "pred_label",
                    "confidence",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
