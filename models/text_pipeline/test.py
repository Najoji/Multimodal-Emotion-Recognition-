"""
Test the trained text-only emotion classifier.

This script loads pre-trained models from checkpoints and re-evaluates them
on the speaker-holdout split by recomputing predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe


def infer_test_speaker(
    train_speaker: str | None,
    test_speaker: str | None,
    speakers: list[str],
) -> str | None:
    if test_speaker:
        return test_speaker
    if train_speaker:
        for speaker in speakers:
            if speaker != train_speaker:
                return speaker
    return None


def ensure_results_dirs() -> None:
    (ROOT / "Results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "Results" / "plots").mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(
    name: str,
    y_true,
    y_pred,
    labels: list[str],
) -> Path:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plot_path = ROOT / "Results" / "plots" / f"{name}_confusion_matrix.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    return plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained text-only emotion classifier."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="Results/checkpoints",
        help="Directory containing trained models",
    )
    parser.add_argument("--data-dir", default="data", help="Path to TESS dataset")
    args = parser.parse_args()

    checkpoint_dir = ROOT / args.checkpoint_dir

    text_checkpoints = sorted(checkpoint_dir.glob("text_only_*_train.joblib"))
    if not text_checkpoints:
        print(f"No text checkpoints found in {checkpoint_dir}")
        return

    try:
        df = load_tess_dataframe(args.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return

    df["speaker"] = df["audio_path"].str.extract(r"([OY]AF)_")[0]
    speakers = sorted(df["speaker"].dropna().unique().tolist())
    if not speakers:
        print("No speakers found in dataset.")
        return

    ensure_results_dirs()

    print("=" * 60)
    print("TEXT-ONLY MODEL EVALUATION")
    print("=" * 60)

    for checkpoint_path in text_checkpoints:
        print(f"\nLoading: {checkpoint_path.name}")
        checkpoint = joblib.load(checkpoint_path)

        model = checkpoint.get("model")
        train_speaker = checkpoint.get("train_speaker")
        test_speaker = infer_test_speaker(
            train_speaker,
            checkpoint.get("test_speaker"),
            speakers,
        )

        if model is None:
            print("  Missing model in checkpoint; skipping.")
            continue
        if not test_speaker:
            print("  Unable to infer test speaker; skipping.")
            continue

        test_df = df[df["speaker"] == test_speaker]
        if test_df.empty:
            print(f"  No samples for speaker {test_speaker}; skipping.")
            continue

        test_text = test_df["transcript"].fillna("")
        predictions = model.predict(test_text)
        accuracy = accuracy_score(test_df["emotion"], predictions)
        labels = checkpoint.get("labels") or sorted(pd.unique(test_df["emotion"]))
        report = classification_report(
            test_df["emotion"],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_path = ROOT / "Results" / "tables" / f"text_only_{train_speaker}_test_report.csv"
        report_df.to_csv(report_path)
        plot_path = save_confusion_matrix(
            f"text_only_{train_speaker}_test",
            test_df["emotion"],
            predictions,
            labels,
        )

        print(f"  Trained on: {train_speaker}")
        print(f"  Tested on: {test_speaker}")
        print(f"  Accuracy: {accuracy:.4f}")
        try:
            report_display = report_path.relative_to(ROOT)
        except ValueError:
            report_display = report_path
        try:
            plot_display = plot_path.relative_to(ROOT)
        except ValueError:
            plot_display = plot_path

        print(f"  Saved report: {report_display}")
        print(f"  Saved plot: {plot_display}")

    train_script = Path(__file__).with_name("train.py")
    try:
        train_command = train_script.relative_to(ROOT)
    except ValueError:
        train_command = train_script

    print("\n" + "=" * 60)
    print(f"Retrain command: python {train_command}")
    print("Use the command above to update checkpoints.")
    print("=" * 60)


if __name__ == "__main__":
    main()
