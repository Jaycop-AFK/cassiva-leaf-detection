# Cassava leaf-only external evaluation

## Setup and safeguards

- Model: uploaded `best.pt`, EfficientNet-B0, epoch 10, experiment `aug_off_weight_off`.
- Inference: resize to 256×256, center crop 224, ImageNet normalization; CPU inference.
- Original 33 images are left untouched. No retraining or threshold tuning was performed.
- One reviewed crop per included source image. Three source images were excluded for the reasons in `external_test_leaf_only/metadata.csv`.
- Source URLs and original source names were not supplied with the ZIP files; those metadata fields remain empty. The labels here are inherited from the supplied class folders, not independently authenticated.
- Without the training images or their hashes, overlap with Kaggle/CCMT cannot be verified. The source hashes in metadata permit a later audit.

## Results

| Dataset | Images | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Validation (checkpoint metadata) | 3,474 | 92.31% | 0.8882 |
| Original held-out test (prior reported) | 3,474 | 89.55% | 0.8677 |
| External Web Test, reproduced on uploads | 33 | 75.76% | 0.7210 |
| Original images for the same 30 retained cases | 30 | 76.67% | 0.7127 |
| Leaf-only crops, paired 30 cases | 30 | 80.00% | 0.7446 |

The paired difference is **+3.33 percentage points accuracy** and **+0.0319 Macro-F1**. Compared with the full 33-image external set, leaf-only Macro-F1 is higher by 0.0236, but that unpaired comparison also changes the case mix. The small paired improvement is descriptive, not evidence of a reliable population-wide gain.

## Leaf-only class metrics

| Class | N | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| CBB | 7 | 0.6667 | 0.5714 | 0.6154 |
| CBS | 5 | 1.0000 | 0.4000 | 0.5714 |
| CMD | 11 | 0.8462 | 1.0000 | 0.9167 |
| Healthy | 7 | 0.7778 | 1.0000 | 0.8750 |

Rows=true, columns=predicted, order CBB, CBS, CMD, Healthy:

```
[[4, 0, 2, 1],
 [2, 2, 0, 1],
 [0, 0,11, 0],
 [0, 0, 0, 7]]
```

Only two predictions changed on paired cases: `cbb_04.webp` changed from CMD to CBB, and `cbb_08.webp` changed from CMD to Healthy. The other 28 predictions stayed the same. CBS has the lowest cropped-set F1 (0.5714), while CBB still has two CMD errors. A crop improves one CBB case but does not solve the generalization problem.

## Interpretation

This is an **exploratory crop sensitivity analysis**. Selecting a leaf region also removes context and changes the image content; the present result cannot isolate image composition as the cause of error. Crops from scientific figures can retain arrows or annotations where those overlap the leaf. The 30 source images are few, selected from web material, and their labels and provenance need verification before use as a definitive external benchmark. Check metadata against source publications and training-image hashes before drawing a stronger conclusion.

## Files

- `external_test_leaf_only/labeled/`: reviewed crops, one per source image.
- `external_test_leaf_only/metadata.csv`: coordinate boxes, source SHA-256, notes and exclusions.
- `leaf_only_results/predictions.csv`: image probabilities for 33 originals and 30 crops.
- `leaf_only_results/metrics.json`: detailed metrics and confusion matrices, including paired figures.
- `leaf_only_results/*_confusion_matrix.png`: two confusion-matrix images.
- `archive/make_leaf_only.py`, `archive/evaluate_leaf_only.py`: reproducible processing scripts. To rerun, place the original class images under `source/{cbb,cbs,cmd,healthy}` and the checkpoint at `upload/01-best.pt` next to the project root, then run the scripts from the project root.
