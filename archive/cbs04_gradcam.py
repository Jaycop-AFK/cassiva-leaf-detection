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
    / "external_gradcam_cbs04"
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
    print("CBS_04 EXTERNAL GRAD-CAM")
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
# GRAD-CAM
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
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
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
                "No activations captured."
            )


        if self.gradients is None:

            raise RuntimeError(
                "No gradients captured."
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
# FIND CBS_04 IMAGES
# ================================================================

def find_cbs04_images():

    results = []


    # ------------------------------------------------------------
    # ORIGINAL
    # ------------------------------------------------------------

    original_dir = (
        ROOT
        / "external_test"
        / "labeled"
        / "cbs"
    )


    if original_dir.exists():

        for path in sorted(
            original_dir.iterdir()
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            if (
                "cbs_04"
                in path.stem.lower()
            ):

                results.append(
                    {
                        "path":
                            path,

                        "version":
                            "original",
                    }
                )


    # ------------------------------------------------------------
    # LEAF ONLY
    # ------------------------------------------------------------

    leaf_dir = (
        ROOT
        / "external_test_leaf_only"
        / "labeled"
        / "cbs"
    )


    if leaf_dir.exists():

        for path in sorted(
            leaf_dir.iterdir()
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            if (
                "cbs_04"
                in path.stem.lower()
            ):

                results.append(
                    {
                        "path":
                            path,

                        "version":
                            "leaf_only",
                    }
                )


    return results


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
# CREATE MODEL INPUT DISPLAY
#
# eval_transform:
# Resize((256, 256))
# CenterCrop(224)
# ================================================================

def create_model_input_display(
    image,
):

    try:

        resample = (
            Image.Resampling.BILINEAR
        )

    except AttributeError:

        resample = (
            Image.BILINEAR
        )


    resized = image.resize(
        (
            256,
            256,
        ),
        resample=resample,
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


    array = np.asarray(
        cropped
    ).astype(
        np.float32
    ) / 255.0


    return array


# ================================================================
# MODEL PREDICTION
# ================================================================

def predict(
    model,
    tensor,
):

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
        .detach()
        .cpu()
        .numpy(),
    )


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
# GENERATE FIGURE
# ================================================================

def generate_gradcam(
    model,
    gradcam,
    class_names,
    image_path,
    version,
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


    true_label = "cbs"

    true_idx = (
        class_names.index(
            true_label
        )
    )


    # ------------------------------------------------------------
    # PREDICTED CLASS CAM
    # ------------------------------------------------------------

    predicted_cam = gradcam.generate(
        tensor,
        pred_idx,
    )


    # ------------------------------------------------------------
    # TRUE CBS CAM
    # ------------------------------------------------------------

    true_cam = gradcam.generate(
        tensor,
        true_idx,
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


    # ------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # PREDICTED CAM
    # ------------------------------------------------------------

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


    # ------------------------------------------------------------
    # TRUE CLASS CAM
    # ------------------------------------------------------------

    ax3 = fig.add_subplot(
        1,
        3,
        3,
    )

    ax3.imshow(
        true_overlay
    )

    ax3.set_title(
        "True-class CAM\n"
        "cbs"
    )

    ax3.axis(
        "off"
    )


    fig.suptitle(
        f"cbs_04_{version}\n"
        f"true=cbs | pred={pred_label}"
    )


    plt.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"cbs_04_{version}.png"
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
            f"cbs_04_{version}",

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

        "correct":
            pred_label
            == true_label,

        "gradcam_path":
            str(
                output_path
            ),
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


    return result


# ================================================================
# MAIN
# ================================================================

def main():

    model, class_names = (
        load_model()
    )


    images = (
        find_cbs04_images()
    )


    if len(images) == 0:

        raise FileNotFoundError(
            "Cannot find cbs_04 image.\n"
            "Expected under:\n"
            "external_test/labeled/cbs/\n"
            "or\n"
            "external_test_leaf_only/labeled/cbs/"
        )


    print()
    print("=" * 72)
    print("FOUND CBS_04 IMAGES")
    print("=" * 72)


    for item in images:

        print(
            item[
                "version"
            ],
            "->",
            item[
                "path"
            ]
        )


    gradcam = GradCAM(
        model=model,
        target_layer=model.conv_head,
    )


    results = []


    print()
    print("=" * 72)
    print("GENERATING GRAD-CAM")
    print("=" * 72)


    for index, item in enumerate(
        images,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(images)}] "
            f"cbs_04_{item['version']}"
        )

        print(
            "Image:",
            item[
                "path"
            ]
        )


        result = generate_gradcam(
            model=model,
            gradcam=gradcam,
            class_names=class_names,
            image_path=item[
                "path"
            ],
            version=item[
                "version"
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

        print(
            "Correct:",
            result[
                "correct"
            ]
        )


        print(
            "Probabilities:"
        )


        for class_name in class_names:

            print(
                f"  "
                f"{class_name:<8}"
                f"{result[f'prob_{class_name}'] * 100:.2f}%"
            )


    gradcam.close()


    # ============================================================
    # SAVE SUMMARY
    # ================================================================

    results_df = pd.DataFrame(
        results
    )


    summary_path = (
        OUTPUT_DIR
        / "cbs04_gradcam_summary.csv"
    )


    results_df.to_csv(
        summary_path,
        index=False,
    )


    # ============================================================
    # FINAL
    # ================================================================

    print()
    print("=" * 72)
    print("CBS_04 GRAD-CAM FINISHED")
    print("=" * 72)

    print(
        "Cases:",
        len(results)
    )

    print(
        "Output directory:",
        OUTPUT_DIR
    )

    print(
        "Summary:",
        summary_path
    )


    print()

    print(
        results_df[
            [
                "case",
                "true_label",
                "pred_label",
                "confidence",
                "correct",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
