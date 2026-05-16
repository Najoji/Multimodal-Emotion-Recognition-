from __future__ import annotations

import argparse
import os
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
from transformers import AutoFeatureExtractor, AutoModel

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
    feature_extractor,
    model,
    sample_rate: int,
    device: torch.device,
) -> np.ndarray:
    """Extract embedding using feature extractor + model."""
    waveform = load_waveform(path, sample_rate)
    
    # Process with feature extractor
    inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Extract embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    hidden = outputs.last_hidden_state.squeeze(0).detach().cpu().numpy()
    return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)]).astype(np.float32)


def build_or_load_embeddings(
    df: pd.DataFrame,
    model_name: str,
    sample_rate: int,
    cache_path: Path,
    hf_token: str | None,
) -> pd.DataFrame:
    """Load cached embeddings or extract new ones."""
    if cache_path.exists():
        print(f"  Loading cached embeddings from {cache_path.name}...")
        return joblib.load(cache_path)

    print(f"  Downloading and loading model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    # Load feature extractor and model
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, token=hf_token
        )
        print(f"  [OK] Feature extractor loaded")
    except Exception as e:
        print(f"  [WARN] Feature extractor loading failed: {e}")
        print(f"  Trying with sampling_rate parameter...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, sampling_rate=sample_rate, token=hf_token
        )
    
    model = AutoModel.from_pretrained(model_name, token=hf_token).to(device)
    model.eval()
    print(f"  [OK] Model loaded ({model_name})")

    rows = []
    for index, row in df.reset_index(drop=True).iterrows():
        try:
            embedding = extract_embedding(
                row["audio_path"],
                feature_extractor=feature_extractor,
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
        except Exception as e:
            print(f"  [WARN] Error processing {row['audio_path']}: {e}")
            continue
        
        if (index + 1) % 100 == 0:
            print(f"  extracted {index + 1}/{len(df)} files")

    result = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, cache_path)
    print(f"  [OK] Cached {len(result)} embeddings to {cache_path.name}")
    return result


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_list())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--model-name",
        default="microsoft/wavlm-base-plus",
        help="Pretrained model name (wav2vec2, hubert, wavlm)",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--classifier", choices=["linear_svm", "logistic_regression"], default="linear_svm")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    # Auto-generate cache path from model name if not provided
    if args.cache_path is None:
        model_short = args.model_name.split("/")[-1].replace("-", "_")
        args.cache_path = f"Results/embedding_cache/{model_short}_embeddings.joblib"

    ensure_results_dirs()
    
    print(f"\n{'='*70}")
    print(f"Pretrained Model Evaluation: {args.model_name}")
    print(f"{'='*70}\n")
    
    print("Loading dataset...")
    df = load_tess_dataframe(args.data_dir)
    df["speaker"] = df["audio_path"].str.extract(r"([OY]AF)_")[0]
    print(f"  [OK] Loaded {len(df)} audio files from {len(df['speaker'].unique())} speakers\n")

    print("Extracting embeddings...")
    embeddings = build_or_load_embeddings(
        df,
        model_name=args.model_name,
        sample_rate=args.sample_rate,
        cache_path=Path(args.cache_path),
        hf_token=hf_token,
    )
    print(f"  [OK] Using {len(embeddings)} embeddings for evaluation\n")

    labels = sorted(embeddings["emotion"].unique().tolist())
    rows = []
    all_predictions = []

    print("Running speaker-holdout evaluation...\n")
    print(f"{'Train':^8} | {'Test':^8} | {'Accuracy':^10} | {'Model':<30}")
    print("-" * 60)

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
                    "audio_path": test_df["audio_path"].values,
                    "true": test_df["emotion"].values,
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
        
        model_short = args.model_name.split("/")[-1]
        pd.DataFrame(report).transpose().to_csv(
            f"Results/tables/{model_short}_{train_speaker}_to_{test_speaker}_classification_report.csv"
        )
        pd.DataFrame(confusion_matrix(test_df["emotion"], predictions, labels=labels)).to_csv(
            f"Results/tables/{model_short}_{train_speaker}_to_{test_speaker}_confusion_matrix.csv",
            index=False,
        )
        
        print(f" {train_speaker:^8} | {test_speaker:^8} | {accuracy:^10.2%} | {args.model_name:<30}")

    result = pd.DataFrame(rows)
    model_short = args.model_name.split("/")[-1]
    result.to_csv(f"Results/tables/{model_short}_speaker_holdout.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        f"Results/tables/{model_short}_speaker_holdout_predictions.csv", index=False
    )
    
    avg_accuracy = result["accuracy"].mean()
    print("-" * 60)
    print(f"\n{'='*70}")
    print(f"AVERAGE ACCURACY: {avg_accuracy:.2%}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
