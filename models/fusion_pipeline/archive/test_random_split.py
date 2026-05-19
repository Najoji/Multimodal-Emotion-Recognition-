from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.audio_features import build_audio_feature_matrix
from speech_emotion.evaluation import save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Results/checkpoints/fusion.joblib")
    parser.add_argument("--test-split", default="Results/tables/fusion_test_split.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    df = pd.read_csv(args.test_split)

    speech = bundle["scaler"].transform(
        build_audio_feature_matrix(df["audio_path"].tolist(), sample_rate=bundle["sample_rate"])
    )
    text = bundle["vectorizer"].transform(df["transcript"])
    x = hstack([csr_matrix(speech), text])

    predictions = bundle["classifier"].predict(x)
    accuracy = save_metrics("fusion_test", df["emotion"], predictions, bundle["labels"])
    print(f"Loaded model: {args.model_path}")
    print(f"Evaluated saved test split: {args.test_split}")
    print(f"Recomputed accuracy: {accuracy:.4f}")
    print("Saved metrics: Results/tables/fusion_test_accuracy.csv and classification report.")


if __name__ == "__main__":
    main()
