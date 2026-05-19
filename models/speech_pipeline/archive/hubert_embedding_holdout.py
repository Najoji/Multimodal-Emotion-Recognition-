from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


def load_waveform(path: str, sample_rate: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)
    return y.astype(np.float32)


def extract_embedding(
    path: str,
    processor,
    model,
    sample_rate: int,
    device: torch.device,
) -> np.ndarray:
    waveform = load_waveform(path, sample_rate)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
    return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)]).astype(np.float32)


def build_or_load_embeddings(
    df: pd.DataFrame,
    model_name: str,
    sample_rate: int,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        return joblib.load(cache_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    rows = []
    for index, row in df.reset_index(drop=True).iterrows():
        embedding = extract_embedding(
            row["audio_path"],
            processor=processor,
            model=model,
            sample_rate=sample_rate,
            device=device,
        )
        rows.append(
            {
                "audio_path": row["audio_path"],
                "transcript": row["transcript"],
                "emotion": row["emotion"],
                "speaker": row["speaker"],
                "embedding": embedding,
            }
        )
        if (index + 1) % 100 == 0:
            print(f"embedded {index + 1}/{len(df)} files")

    result = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, cache_path)
    return result


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_list())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-name", default="microsoft/wavlm-base")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--classifier", choices=["linear_svm", "logistic_regression"], default="linear_svm")
    parser.add_argument("--cache-path", default="Results/embedding_cache/wavlm_base_embeddings.joblib")
    args = parser.parse_args()

    ensure_results_dirs()
    df = load_tess_dataframe(args.data_dir)
    df["speaker"] = df["audio_path"].str.extract(r"([OY]AF)_")[0]

    embeddings = build_or_load_embeddings(
        df,
        model_name=args.model_name,
        sample_rate=args.sample_rate,
        cache_path=Path(args.cache_path),
    )

    labels = sorted(embeddings["emotion"].unique().tolist())
    rows = []
    all_predictions = []

    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        train_df = embeddings[embeddings["speaker"] == train_speaker]
        test_df = embeddings[embeddings["speaker"] == test_speaker]
        x_train = matrix_from_embeddings(train_df)
        x_test = matrix_from_embeddings(test_df)

        if args.classifier == "logistic_regression":
            classifier = LogisticRegression(max_iter=3000, class_weight="balanced")
        else:
            classifier = LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=10000)

        model = make_pipeline(StandardScaler(), classifier)
        model.fit(x_train, train_df["emotion"])
        predictions = model.predict(x_test)
        accuracy = accuracy_score(test_df["emotion"], predictions)
        rows.append(
            {
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
                "model_name": args.model_name,
                "classifier": args.classifier,
                "accuracy": accuracy,
            }
        )
        all_predictions.append(
            pd.DataFrame(
                {
                    "audio_path": test_df["audio_path"],
                    "true": test_df["emotion"],
                    "predicted": predictions,
                    "train_speaker": train_speaker,
                    "test_speaker": test_speaker,
                }
            )
        )

        report = classification_report(
            test_df["emotion"],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            f"Results/tables/wavlm_{train_speaker}_to_{test_speaker}_classification_report.csv"
        )
        pd.DataFrame(confusion_matrix(test_df["emotion"], predictions, labels=labels)).to_csv(
            f"Results/tables/wavlm_{train_speaker}_to_{test_speaker}_confusion_matrix.csv",
            index=False,
        )

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/wavlm_speaker_holdout.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        "Results/tables/wavlm_speaker_holdout_predictions.csv", index=False
    )
    print(result.to_string(index=False))
    print(f"average accuracy: {result['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()
