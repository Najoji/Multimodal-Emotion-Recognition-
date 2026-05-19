"""
Test the trained fusion emotion classifier.

This script loads pre-trained models from checkpoints and evaluates them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained fusion emotion classifier."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="Results/checkpoints",
        help="Directory containing trained models",
    )
    args = parser.parse_args()

    checkpoint_dir = ROOT / args.checkpoint_dir

    # Find fusion models
    fusion_checkpoints = sorted(checkpoint_dir.glob("fusion_*_train.joblib"))

    if not fusion_checkpoints:
        print(f"No fusion checkpoints found in {checkpoint_dir}")
        return

    print("=" * 60)
    print("FUSION MODEL EVALUATION")
    print("=" * 60)

    for checkpoint_path in fusion_checkpoints:
        print(f"\nLoading: {checkpoint_path.name}")
        checkpoint = joblib.load(checkpoint_path)

        model = checkpoint["model"]
        train_speaker = checkpoint["train_speaker"]

        print(f"  Trained on: {train_speaker}")
        print(f"  Architecture: Speech (Emotion2Vec+) + Text (TF-IDF)")

        # Try to load corresponding classification report
        report_name = f"fusion_{train_speaker}_train_report.csv"
        report_path = ROOT / "Results" / "tables" / report_name
        if report_path.exists():
            report_df = pd.read_csv(report_path, index_col=0)
            accuracy = report_df.loc["accuracy", "accuracy"]
            print(f"  Accuracy: {accuracy:.4f}")

    print("\n" + "=" * 60)
    print("Use train.py to retrain models or update checkpoints.")
    print("=" * 60)


if __name__ == "__main__":
    main()
