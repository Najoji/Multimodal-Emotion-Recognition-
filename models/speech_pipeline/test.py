from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.audio_features import build_feature_matrix
from speech_emotion.evaluation import save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Results/checkpoints/speech_only.joblib")
    parser.add_argument("--test-split", default="Results/tables/speech_only_test_split.csv")
    parser.add_argument("--output-name", default="speech_only_test")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    df = pd.read_csv(args.test_split)
    x = build_feature_matrix(
        df["audio_path"].tolist(),
        sample_rate=bundle["sample_rate"],
        feature_set=bundle.get("feature_set", "mfcc"),
    )
    predictions = bundle["model"].predict(x)
    accuracy = save_metrics(args.output_name, df["emotion"], predictions, bundle["labels"])
    print(f"Loaded model: {args.model_path}")
    print(f"Evaluated saved test split: {args.test_split}")
    print(f"Recomputed accuracy: {accuracy:.4f}")
    print(f"Saved metrics: Results/tables/{args.output_name}_accuracy.csv and classification report.")


if __name__ == "__main__":
    main()
