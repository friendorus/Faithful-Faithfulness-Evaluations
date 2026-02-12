from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classification(df: pd.DataFrame):
    """
    Compute core classification metrics.
    Expects df with columns:
        - GT
        - NN
        - prob_0, prob_1, ..., prob_C
    """

    df = df.copy()
    df["GT"] = df["GT"].astype(int)
    df["NN"] = df["NN"].astype(int)

    labels = sorted(df["GT"].unique())
    class_names = [str(l) for l in labels]

    cm = confusion_matrix(df["GT"], df["NN"], labels=labels)
    acc = accuracy_score(df["GT"], df["NN"])
    macro_f1 = f1_score(df["GT"], df["NN"], average="macro")

    report = classification_report(
        df["GT"], df["NN"], labels=labels, digits=4, zero_division=0
    )

    # Build probability matrix
    prob_cols = [f"prob_{c}" for c in labels]
    y_score = df[prob_cols].values
    y_true = df["GT"].values

    # One-hot encode GT
    y_true_oh = np.zeros_like(y_score)
    for i, t in enumerate(y_true):
        y_true_oh[i, t] = 1

    auc_per_class = {}

    for c in labels:
        try:
            auc_val = roc_auc_score(y_true_oh[:, c], y_score[:, c])
            auc_per_class[c] = auc_val
        except ValueError:
            auc_per_class[c] = np.nan

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "auc_per_class": auc_per_class,
        "labels": labels,
        "class_names": class_names,
        "y_true_oh": y_true_oh,
        "y_score": y_score,
    }

    return metrics


def plot_confusion_matrix(cm, class_names, output_path, accuracy=None):
    """
    Save confusion matrix figure.
    """

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )

    if accuracy is not None:
        plt.title(f"Confusion Matrix (Acc={accuracy:.3f})")
    else:
        plt.title("Confusion Matrix")

    plt.xlabel("Prediction")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_roc_curves(labels, y_true_oh, y_score, output_path):
    """
    Save multi-class one-vs-rest ROC curve.
    """

    plt.figure(figsize=(6, 6))

    for c in labels:
        try:
            fpr, tpr, _ = roc_curve(y_true_oh[:, c], y_score[:, c])
            auc_val = roc_auc_score(y_true_oh[:, c], y_score[:, c])
            plt.plot(fpr, tpr, label=f"Class {c} (AUC={auc_val:.3f})")
        except ValueError:
            continue

    plt.plot([0, 1], [0, 1], "--")
    plt.title("Multi-class ROC (One-vs-Rest)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_and_plot(df: pd.DataFrame, output_dir: str):
    """
    Full evaluation pipeline:
        - compute metrics
        - print summary
        - save confusion matrix
        - save ROC curves
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_classification(df)

    clean_auc = {int(k): float(v) for k, v in metrics["auc_per_class"].items()}

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nClassification Report:\n")
    print(metrics["classification_report"])

    print("AUC per class:", clean_auc)

    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=metrics["class_names"],
        output_path=output_dir / "confusion_matrix.png",
        accuracy=metrics["accuracy"],
    )

    plot_roc_curves(
        labels=metrics["labels"],
        y_true_oh=metrics["y_true_oh"],
        y_score=metrics["y_score"],
        output_path=output_dir / "roc_multiclass.png",
    )

    return metrics
