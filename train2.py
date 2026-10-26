import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
os.environ["CASSAVA_MODEL_NAME"] = "resnet18"
os.environ["CASSAVA_RUN_TAG"] = "resnet18"

runpy.run_path(str(ROOT / "train.py"), run_name="__main__")
