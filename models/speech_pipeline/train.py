"""
Speech-only emotion recognition using Emotion2Vec+ embeddings and speaker-holdout evaluation.

This script demonstrates the final speech-only model with proper cross-speaker evaluation.
It uses Emotion2Vec+ (state-of-the-art emotion-specialized speech model) instead of handcrafted MFCCs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from funasr import AutoModel
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


MODEL_NAME = "iic/emotion2vec_plus_base"


def load_waveform(path: str, sample_rate: int) -> np.ndarray:
    """Load audio file, trim silence, and return as float32."""
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)
    return y.astype(np.float32)


def build_or_load_embeddings(
    df: pd.DataFrame,
    sample_rate: int,
    cache_path: Path,
) -> pd.DataFrame:
    """Build Emotion2Vec+ embeddings or load from cache."""
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        return joblib.load(cache_path)

    print(f"Building Emotion2Vec+ embeddings (model: {MODEL_NAME})...")
    model = AutoModel(model=MODEL_NAME, disable_update=True)
    rows = []

    for index, row in df.reset_index(drop=True).iterrows():
        waveform = load_waveform(row["audio_path"], sample_rate=sample_rate)
        result = model.generate(
            waveform,
            granularity="utterance",
            extract_embedding=True,
        )[0]
        rows.append(
            {
                "audio_path": row["audio_path"],
                "transcript": row["transcript"],
                "emotion": row["emotion"],
                "speaker": row["speaker"],
                "embedding": np.asarray(result["feats"], dtype=np.float32),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"  embedded {index + 1}/{len(df)} files")

    result_df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result_df, cache_path)
    print(f"Saved embeddings to {cache_path}")
    return result_df


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Convert dataframe of embeddings to numpy matrix."""
    return np.vstack(df["embedding"].to_list())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train speech-only emotion classifier with speaker-holdout evaluation."
    )
    parser.add_argument("--data-dir", default="data", help="Path to TESS dataset")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument(
        "--cache-path",
        default="Results/embedding_cache/emotion2vec_plus_base_embeddings.joblib",
        help="Path to embedding cache",
    )
    args = parser.parse_args()

    ensure_results_dirs()
    
    # Load dataset
    print("Loading TESS dataset...")
    df = load_tess_dataframe(args.data_dir)
    df["speaker"] = df["audio_path"].str.extract(r"([OY]AF)_")[0]
    print(f"  Loaded {len(df)} samples from {df['speaker'].nunique()} speakers")

    # Build or load embeddings
    embeddings_df = build_or_load_embeddings(
        df,
        sample_rate=args.sample_rate,
        cache_path=Path(args.cache_path),
    )

    # Get emotion labels in sorted order
    labels = sorted(embeddings_df["emotion"].unique().tolist())
    print(f"  Emotions: {labels}")

    # Speaker-holdout evaluation (train on OAF, test on YAF, then vice versa)
    print("\n" + "=" * 60)
    print("SPEAKER-HOLDOUT EVALUATION")
    print("=" * 60)

    results = []

    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        print(f"\nTraining on {train_speaker}, testing on {test_speaker}...")

        # Split by speaker
        train_df = embeddings_df[embeddings_df["speaker"] == train_speaker]
        test_df = embeddings_df[embeddings_df["speaker"] == test_speaker]

        x_train = matrix_from_embeddings(train_df)
        x_test = matrix_from_embeddings(test_df)
        y_train = train_df["emotion"].values
        y_test = test_df["emotion"].values

        # Train classifier
        model = make_pipeline(
            StandardScaler(),
            Normalizer(norm="l2"),
            LinearSVC(
                C=0.1,
                class_weight="balanced",
                random_state=42,
                dual="auto",
                max_iter=10000,
            ),
        )
        model.fit(x_train, y_train)

        # Evaluate
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"  Accuracy: {accuracy:.4f}")

        results.append(
            {
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
                "model": MODEL_NAME,
                "classifier": "LinearSVC(C=0.1, balanced)",
                "accuracy": accuracy,
            }
        )

        # Save model
        checkpoint_name = f"speech_only_{train_speaker}_train.joblib"
        checkpoint_path = ROOT / "Results" / "checkpoints" / checkpoint_name
        joblib.dump(
            {
                "model": model,
                "labels": labels,
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
                "model_name": MODEL_NAME,
            },
            checkpoint_path,
        )
        print(f"  Saved checkpoint: {checkpoint_path.name}")

        # Save classification report
        report = classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_path = ROOT / "Results" / "tables" / f"speech_only_{train_speaker}_train_report.csv"
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

    # Save summary
    summary_path = ROOT / "Results" / "tables" / "speech_only_accuracy.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path.name}")


if __name__ == "__main__":
    main()
