# Project file map

## Core pipeline

- `build_manifest.py` - build the combined Kaggle/CCMT manifest.
- `clean_data.py` - validate images and remove exact/perceptual duplicates from the manifest.
- `split_data.py` - create the stratified train/validation/test split.
- `dataset.py` - class names, transforms, and PyTorch Dataset.
- `train.py` - main EfficientNet-B0 training script.
- `evaluate.py` - test metrics and confusion matrices.
- `inference.py` - command-line prediction for one new image.
- `app.py` - Streamlit demo GUI.
- `run_gui.ps1` - start the GUI.

## Experiments

The scripts in `experiments/` are reproducible experiment and interpretation code. They are not required to start the GUI.

- `ablation_train.py` - augmentation/class-weighting ablation.
- `rerun_aug_off_weight_off.py` - verified rerun of the selected experiment.
- `verify_ablation.py` - checkpoint/history consistency check.
- `analyze_source.py` - source-bias analysis.
- `gradcam.py` - Grad-CAM on selected internal test images.

## Archive

The scripts in `archive/` are one-time checks or exploratory external-test work. They are retained for reproducibility and evidence, but are not part of the normal GUI path.

The moved scripts resolve the project root from their new location, so they can still be run from the project root when needed.
