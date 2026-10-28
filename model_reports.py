"""Shared model registry and reproducible presentation artifacts."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "outputs" / "comparison"
MODELS = {
    "EfficientNet-B0": {
        "architecture": "efficientnet_b0",
        "checkpoint": ROOT / "outputs/checkpoints/efficientnet_b0_best.pt",
        "history": ROOT / "outputs/training_history.csv",
        "evaluation": ROOT / "outputs/evaluation",
    },
    "ResNet18": {
        "architecture": "resnet18",
        "checkpoint": ROOT / "outputs/checkpoints/resnet18_best.pt",
        "history": ROOT / "outputs/training_history_resnet18.csv",
        "evaluation": ROOT / "outputs/evaluation_resnet18",
    },
    "MobileNetV3-Large": {
        "architecture": "mobilenetv3_large_100",
        "checkpoint": ROOT / "outputs/checkpoints/mobilenetv3_large_100_best.pt",
        "history": ROOT / "outputs/training_history_mobilenetv3_large_100.csv",
        "evaluation": ROOT / "outputs/evaluation_mobilenetv3_large_100",
    },
}


def comparison_table():
    rows = []
    reference = None
    for name, config in MODELS.items():
        predictions = pd.read_csv(config["evaluation"] / "test_predictions.csv")
        identities = predictions[["filepath", "true_idx"]].sort_values("filepath").reset_index(drop=True)
        if reference is not None and not reference.equals(identities):
            raise ValueError("Model evaluations do not use identical test images and labels")
        reference = identities
        precision, recall, f1, _ = precision_recall_fscore_support(
            predictions.true_idx, predictions.pred_idx, average="macro", zero_division=0
        )
        history = pd.read_csv(config["history"])
        best = history.loc[history.val_macro_f1.idxmax()]
        rows.append({"Model": name, "Accuracy": accuracy_score(predictions.true_idx, predictions.pred_idx),
                     "Macro Precision": precision, "Macro Recall": recall, "Macro-F1": f1,
                     "Test images": len(predictions), "Best epoch": int(best.epoch),
                     "Validation Macro-F1": best.val_macro_f1})
    return pd.DataFrame(rows)


def export_reports():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table = comparison_table()
    table.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    for name, config in MODELS.items():
        history = pd.read_csv(config["history"])
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, columns, title in zip(axes, [("train_loss", "val_loss"), ("train_accuracy", "val_accuracy"), ("val_macro_f1",)], ["Loss", "Accuracy", "Validation Macro-F1"]):
            for column in columns:
                ax.plot(history.epoch, history[column], marker="o", label=column)
                for epoch, value in zip(history.epoch, history[column]):
                    ax.annotate(f"{value:.3f}", (epoch, value), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
            ax.set(title=title, xlabel="Epoch")
            ax.grid(alpha=0.25)
            ax.legend()
        fig.suptitle(name)
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"learning_curve_{config['architecture']}.png", dpi=200)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics = ["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1"]
    plotted = table.set_index("Model")[metrics].plot.bar(ax=ax, rot=0, stacked=False)
    ax.set(ylabel="Score (0–1)", ylim=(0, 1), title="Model comparison - same held-out test set")
    ax.grid(axis="y", alpha=0.25)
    for container in plotted.containers:
        ax.bar_label(container, fmt="%.4f", padding=3, fontsize=8)
    ax.legend(title="Metric", loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)
    return table


if __name__ == "__main__":
    print(export_reports().to_string(index=False))
