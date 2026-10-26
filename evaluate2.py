"""Evaluate the ResNet18 comparison model on the held-out test set."""

import os
import runpy
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
os.environ["CASSAVA_MODEL_NAME"] = "resnet18"
os.environ["CASSAVA_EVAL_TAG"] = "resnet18"
os.environ["CASSAVA_CHECKPOINT"] = str(
    ROOT / "outputs" / "checkpoints" / "resnet18_best.pt"
)

runpy.run_path(str(ROOT / "evaluate.py"), run_name="__main__")
