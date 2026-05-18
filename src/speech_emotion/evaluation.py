from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("Results/matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def ensure_results_dirs() -> None:
    for folder in ["Results/checkpoints", "Results/tables", "Results/plots", "Results/matplotlib_cache"]:
        Path(folder).mkdir(parents=True, exist_ok=True)


def save_metrics(name: str, y_true, y_pred, labels: list[str]) -> float:
    ensure_results_dirs()

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    pd.DataFrame(report).transpose().to_csv(f"Results/tables/{name}_classification_report.csv")
    pd.DataFrame([{"model": name, "accuracy": accuracy}]).to_csv(
        f"Results/tables/{name}_accuracy.csv", index=False
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"Results/plots/{name}_confusion_matrix.png", dpi=160)
    plt.close()
    return accuracy
