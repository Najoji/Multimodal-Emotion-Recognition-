"""
Test the trained speech-only emotion classifier.

This script loads pre-trained models from checkpoints and re-evaluates them
on the speaker-holdout split by recomputing predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe


MODEL_NAME = "iic/emotion2vec_plus_base"


def load_waveform(path: str, sample_rate: int) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)
    return y.astype(np.float32)


def build_or_load_embeddings(
    df: pd.DataFrame,
    sample_rate: int,
    cache_path: Path,
) -> pd.DataFrame | None:
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        return joblib.load(cache_path)

    print("Embedding cache not found; building Emotion2Vec+ embeddings...")
    try:
        from funasr import AutoModel
    except Exception as exc:
        print(f"Unable to import AutoModel ({exc}). Run train.py first or install dependencies.")
        return None

    model = AutoModel(model=MODEL_NAME, disable_update=True)
    rows = []

    for index, row in df.reset_index(drop=True).iterrows():
        try:
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
        except Exception as exc:
            print(f"  Skipping {row['audio_path']}: {exc}")
            continue

        if (index + 1) % 100 == 0:
            print(f"  embedded {index + 1}/{len(df)} files")

    if not rows:
        print("No embeddings were built. Check dataset paths.")
        return None

    result_df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result_df, cache_path)
    print(f"Saved embeddings to {cache_path}")
    return result_df


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_list())


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
        description="Evaluate trained speech-only emotion classifier."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="Results/checkpoints",
        help="Directory containing trained models",
    )
    parser.add_argument("--data-dir", default="data", help="Path to TESS dataset")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate")
    parser.add_argument(
        "--cache-path",
        default="Results/embedding_cache/emotion2vec_plus_base_embeddings.joblib",
        help="Path to embedding cache",
    )
    args = parser.parse_args()

    checkpoint_dir = ROOT / args.checkpoint_dir

    speech_checkpoints = sorted(checkpoint_dir.glob("speech_only_*_train.joblib"))
    if not speech_checkpoints:
        print(f"No speech checkpoints found in {checkpoint_dir}")
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

    embeddings_df = build_or_load_embeddings(
        df,
        sample_rate=args.sample_rate,
        cache_path=Path(args.cache_path),
    )
    if embeddings_df is None:
        return

    if "speaker" not in embeddings_df.columns:
        embeddings_df["speaker"] = embeddings_df["audio_path"].str.extract(r"([OY]AF)_")[0]
    if "emotion" not in embeddings_df.columns:
        embeddings_df = embeddings_df.merge(
            df[["audio_path", "emotion"]],
            on="audio_path",
            how="left",
        )

    print("=" * 60)
    print("SPEECH-ONLY MODEL EVALUATION")
    print("=" * 60)

    for checkpoint_path in speech_checkpoints:
        print(f"\nLoading: {checkpoint_path.name}")
        checkpoint = joblib.load(checkpoint_path)

        model = checkpoint.get("model")
        train_speaker = checkpoint.get("train_speaker")
        test_speaker = infer_test_speaker(
            train_speaker,
            checkpoint.get("test_speaker"),
            speakers,
        )
        model_name = checkpoint.get("model_name", MODEL_NAME)

        if model is None:
            print("  Missing model in checkpoint; skipping.")
            continue
        if not test_speaker:
            print("  Unable to infer test speaker; skipping.")
            continue

        test_df = embeddings_df[embeddings_df["speaker"] == test_speaker]
        if test_df.empty:
            print(f"  No samples for speaker {test_speaker}; skipping.")
            continue

        x_test = matrix_from_embeddings(test_df)
        y_test = test_df["emotion"].values
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        labels = checkpoint.get("labels") or sorted(pd.unique(y_test))
        report = classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_path = ROOT / "Results" / "tables" / f"speech_only_{train_speaker}_test_report.csv"
        report_df.to_csv(report_path)
        plot_path = save_confusion_matrix(
            f"speech_only_{train_speaker}_test",
            y_test,
            predictions,
            labels,
        )

        print(f"  Model: {model_name}")
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
