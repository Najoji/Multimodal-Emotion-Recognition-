"""
Text-only emotion recognition using speaker-holdout evaluation.

This demonstrates that TESS transcripts (isolated words) carry almost no emotion signal.
Result: ~14.29% accuracy (chance level for 7 emotions).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train text-only emotion classifier with speaker-holdout evaluation."
    )
    parser.add_argument("--data-dir", default="data", help="Path to TESS dataset")
    args = parser.parse_args()

    ensure_results_dirs()
    
    # Load dataset
    print("Loading TESS dataset...")
    df = load_tess_dataframe(args.data_dir)
    df["speaker"] = df["audio_path"].str.extract(r"([OY]AF)_")[0]
    print(f"  Loaded {len(df)} samples from {df['speaker'].nunique()} speakers")
    print(f"  Emotions: {sorted(df['emotion'].unique().tolist())}")

    # Get emotion labels in sorted order
    labels = sorted(df["emotion"].unique().tolist())

    # Speaker-holdout evaluation (train on OAF, test on YAF, then vice versa)
    print("\n" + "=" * 60)
    print("TEXT-ONLY SPEAKER-HOLDOUT EVALUATION")
    print("=" * 60)
    print("Note: TESS transcripts are isolated words with no inherent emotion.")
    print("Expected accuracy: ~14.29% (chance level for 7 emotions)")
    print("=" * 60)

    results = []

    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        print(f"\nTraining on {train_speaker}, testing on {test_speaker}...")

        # Split by speaker
        train_df = df[df["speaker"] == train_speaker]
        test_df = df[df["speaker"] == test_speaker]

        # Train model
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )
        model.fit(train_df["transcript"], train_df["emotion"])

        # Evaluate
        predictions = model.predict(test_df["transcript"])
        accuracy = accuracy_score(test_df["emotion"], predictions)
        print(f"  Accuracy: {accuracy:.4f}")

        results.append(
            {
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
                "representation": "TF-IDF(ngrams=1-2)",
                "classifier": "LogisticRegression(balanced)",
                "accuracy": accuracy,
            }
        )

        # Save model
        checkpoint_name = f"text_only_{train_speaker}_train.joblib"
        checkpoint_path = ROOT / "Results" / "checkpoints" / checkpoint_name
        joblib.dump(
            {
                "model": model,
                "labels": labels,
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
            },
            checkpoint_path,
        )
        print(f"  Saved checkpoint: {checkpoint_path.name}")

        # Save classification report
        report = classification_report(
            test_df["emotion"],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_path = ROOT / "Results" / "tables" / f"text_only_{train_speaker}_train_report.csv"
        report_df.to_csv(report_path)
        print(f"  Saved report: {report_path.name}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    avg_accuracy = results_df["accuracy"].mean()
    print(f"\nAverage speaker-holdout accuracy: {avg_accuracy:.4f}")
    print(f"Chance level (7 emotions): {1/7:.4f}")

    # Save summary
    summary_path = ROOT / "Results" / "tables" / "text_only_accuracy.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path.name}")


if __name__ == "__main__":
    main()
