from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.evaluation import save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Results/checkpoints/text_only.joblib")
    parser.add_argument("--test-split", default="Results/tables/text_only_test_split.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    df = pd.read_csv(args.test_split)
    predictions = bundle["model"].predict(df["transcript"])
    save_metrics("text_only_test", df["emotion"], predictions, bundle["labels"])
    print("Saved text-only test metrics.")


if __name__ == "__main__":
    main()
