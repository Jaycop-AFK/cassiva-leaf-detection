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

from torchvision import transforms
from torchvision.transforms import functional as TF

from dataset import (
    CLASS_NAMES,
    LABEL_TO_IDX,
)


# ================================================================
# CONFIG
# ================================================================


CHECKPOINT_PATH = (
    ROOT
    / "outputs"
    / "checkpoints"
    / "efficientnet_b0_best.pt"
)

PREDICTIONS_PATH = (
    ROOT
    / "outputs"
    / "evaluation"
    / "test_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "gradcam"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SELECTION_PATH = (
    OUTPUT_DIR
    / "gradcam_selection.csv"
)


MODEL_NAME = "efficientnet_b0"

IMAGE_SIZE = 224


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ================================================================
# MODEL INPUT TRANSFORM
#
# IMPORTANT:
# ต้องตรงกับ eval_transform ที่เราใช้ตอน validation/test
# ================================================================

model_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406,
        ],
        std=[
            0.229,
            0.224,
            0.225,
        ],
    ),
])


# ================================================================
# DISPLAY TRANSFORM
#
# ใช้ spatial transform เดียวกัน
# แต่ไม่ Normalize
# ================================================================

display_transform = transforms.Compose([

    transforms.Resize(
        (256, 256)
    ),

    transforms.CenterCrop(
        IMAGE_SIZE
    ),
])


# ================================================================
# LOAD MODEL
# ================================================================

print("=" * 70)
print("STEP 7D - GRAD-CAM")
print("=" * 70)

print(
    "Device:",
    DEVICE
)


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


model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=len(CLASS_NAMES),
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


# ================================================================
# TARGET LAYER
#
# EfficientNet-B0:
# conv_head คือ convolution layer สุดท้ายก่อน global pooling
# ================================================================

target_layer = model.conv_head


print(
    "Grad-CAM target layer:",
    target_layer
)


# ================================================================
# LOAD TEST PREDICTIONS
# ================================================================

predictions_df = pd.read_csv(
    PREDICTIONS_PATH
)


print()
print(
    "Prediction rows:",
    len(predictions_df)
)


# ================================================================
# SELECT REPRESENTATIVE CASES
# ================================================================

def select_case(
    case_name,
    source,
    true_label,
    pred_label=None,
    correct=None,
):

    subset = predictions_df[
        predictions_df[
            "source"
        ].eq(source)
        &
        predictions_df[
            "label"
        ].eq(true_label)
    ].copy()


    if pred_label is not None:

        subset = subset[
            subset[
                "pred_label"
            ].eq(pred_label)
        ]


    if correct is not None:

        subset = subset[
            subset[
                "correct"
            ].eq(correct)
        ]


    if len(subset) == 0:

        print(
            f"WARNING: no sample for {case_name}"
        )

        return None


    # เลือกตัวที่โมเดลมั่นใจสูง
    # เพื่อดูว่าโมเดลใช้หลักฐานอะไรในการตัดสินใจ
    subset = subset.sort_values(
        "confidence",
        ascending=False,
    )


    row = subset.iloc[0].copy()

    row[
        "case_name"
    ] = case_name


    return row


cases = []


case_specs = [

    (
        "kaggle_cbb_correct",
        "kaggle",
        "cbb",
        "cbb",
        True,
    ),

    (
        "kaggle_cbb_to_healthy",
        "kaggle",
        "cbb",
        "healthy",
        False,
    ),

    (
        "kaggle_healthy_correct",
        "kaggle",
        "healthy",
        "healthy",
        True,
    ),

    (
        "kaggle_healthy_to_cbb",
        "kaggle",
        "healthy",
        "cbb",
        False,
    ),

    (
        "kaggle_cmd_correct",
        "kaggle",
        "cmd",
        "cmd",
        True,
    ),

    (
        "kaggle_cmd_to_healthy",
        "kaggle",
        "cmd",
        "healthy",
        False,
    ),

    (
        "ccmt_cbb_correct",
        "ccmt",
        "cbb",
        "cbb",
        True,
    ),

    (
        "ccmt_cbb_to_cbs",
        "ccmt",
        "cbb",
        "cbs",
        False,
    ),

    (
        "ccmt_cbs_correct",
        "ccmt",
        "cbs",
        "cbs",
        True,
    ),

    (
        "ccmt_cbs_to_cbb",
        "ccmt",
        "cbs",
        "cbb",
        False,
    ),

    (
        "ccmt_healthy_correct",
        "ccmt",
        "healthy",
        "healthy",
        True,
    ),

    (
        "ccmt_cmd_correct",
        "ccmt",
        "cmd",
        "cmd",
        True,
    ),
]


for spec in case_specs:

    row = select_case(
        case_name=spec[0],
        source=spec[1],
        true_label=spec[2],
        pred_label=spec[3],
        correct=spec[4],
    )

    if row is not None:

        cases.append(
            row
        )


selection_df = pd.DataFrame(
    cases
)


selection_df.to_csv(
    SELECTION_PATH,
    index=False,
)


print()
print(
    "Selected Grad-CAM cases:",
    len(selection_df)
)

print(
    "Selection saved:",
    SELECTION_PATH
)


# ================================================================
# GRAD-CAM FUNCTION
# ================================================================

def generate_gradcam(
    image_tensor,
    target_class_idx,
):

    activation_holder = {}


    def forward_hook(
        module,
        inputs,
        output,
    ):

        activation_holder[
            "activation"
        ] = output

        output.retain_grad()


    hook_handle = (
        target_layer
        .register_forward_hook(
            forward_hook
        )
    )


    try:

        model.zero_grad(
            set_to_none=True
        )


        logits = model(
            image_tensor
        )


        score = logits[
            0,
            target_class_idx
        ]


        score.backward()


        activations = (
            activation_holder[
                "activation"
            ]
        )


        gradients = (
            activations.grad
        )


        # --------------------------------------------------------
        # Global average pooling ของ gradient
        # --------------------------------------------------------

        weights = gradients.mean(
            dim=(
                2,
                3,
            ),
            keepdim=True,
        )


        # --------------------------------------------------------
        # Weighted combination
        # --------------------------------------------------------

        cam = (
            weights
            * activations
        ).sum(
            dim=1
        )


        cam = torch.relu(
            cam
        )


        # --------------------------------------------------------
        # Resize CAM → 224x224
        # --------------------------------------------------------

        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(1),
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            mode="bilinear",
            align_corners=False,
        )


        cam = cam[
            0,
            0
        ]


        # --------------------------------------------------------
        # Normalize 0 → 1
        # --------------------------------------------------------

        cam_min = cam.min()

        cam_max = cam.max()


        if (
            cam_max
            - cam_min
        ) > 0:

            cam = (
                cam
                - cam_min
            ) / (
                cam_max
                - cam_min
            )


        return (
            cam
            .detach()
            .cpu()
            .numpy()
        )


    finally:

        hook_handle.remove()


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
        ...,
        :3
    ]


    overlay = (
        0.55
        * image_array
        +
        0.45
        * heatmap
    )


    overlay = np.clip(
        overlay,
        0,
        1,
    )


    return overlay


# ================================================================
# PROCESS EACH SAMPLE
# ================================================================

print()
print("=" * 70)
print("GENERATING GRAD-CAM")
print("=" * 70)


for index, row in selection_df.iterrows():

    case_name = (
        row[
            "case_name"
        ]
    )


    filepath = (
        ROOT
        / row[
            "filepath"
        ]
    )


    true_label = (
        row[
            "label"
        ]
    )


    pred_label = (
        row[
            "pred_label"
        ]
    )


    confidence = float(
        row[
            "confidence"
        ]
    )


    source = (
        row[
            "source"
        ]
    )


    # ------------------------------------------------------------
    # OPEN IMAGE
    # ------------------------------------------------------------

    with Image.open(
        filepath
    ) as image:

        image = image.convert(
            "RGB"
        )


        display_image = (
            display_transform(
                image
            )
        )


        model_image = (
            model_transform(
                image
            )
        )


    # ------------------------------------------------------------
    # NUMPY DISPLAY IMAGE
    # ------------------------------------------------------------

    display_array = np.asarray(
        display_image,
        dtype=np.float32,
    ) / 255.0


    # ------------------------------------------------------------
    # MODEL INPUT
    # ------------------------------------------------------------

    image_tensor = (
        model_image
        .unsqueeze(0)
        .to(DEVICE)
    )


    # ------------------------------------------------------------
    # PREDICTED CLASS CAM
    # ------------------------------------------------------------

    pred_idx = LABEL_TO_IDX[
        pred_label
    ]


    pred_cam = generate_gradcam(
        image_tensor,
        pred_idx,
    )


    pred_overlay = make_overlay(
        display_array,
        pred_cam,
    )


    # ------------------------------------------------------------
    # TRUE CLASS CAM
    # ------------------------------------------------------------

    true_idx = LABEL_TO_IDX[
        true_label
    ]


    true_cam = generate_gradcam(
        image_tensor,
        true_idx,
    )


    true_overlay = make_overlay(
        display_array,
        true_cam,
    )


    # ------------------------------------------------------------
    # CREATE FIGURE
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(
            15,
            5,
        ),
    )


    # ORIGINAL
    axes[0].imshow(
        display_array
    )

    axes[0].set_title(
        "Model input"
    )

    axes[0].axis(
        "off"
    )


    # PREDICTED
    axes[1].imshow(
        pred_overlay
    )

    axes[1].set_title(
        f"Predicted CAM\n"
        f"{pred_label} "
        f"({confidence:.3f})"
    )

    axes[1].axis(
        "off"
    )


    # TRUE
    axes[2].imshow(
        true_overlay
    )

    axes[2].set_title(
        f"True-class CAM\n"
        f"{true_label}"
    )

    axes[2].axis(
        "off"
    )


    fig.suptitle(
        f"{case_name}\n"
        f"source={source} | "
        f"true={true_label} | "
        f"pred={pred_label}"
    )


    plt.tight_layout()


    output_path = (
        OUTPUT_DIR
        / f"{index + 1:02d}_{case_name}.png"
    )


    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    print(
        f"[{index + 1:02d}/"
        f"{len(selection_df):02d}] "
        f"{case_name}"
    )


# ================================================================
# FINISHED
# ================================================================

print()
print("=" * 70)
print("GRAD-CAM FINISHED")
print("=" * 70)

print(
    "Output directory:",
    OUTPUT_DIR
)

print(
    "Selection CSV:",
    SELECTION_PATH
)
