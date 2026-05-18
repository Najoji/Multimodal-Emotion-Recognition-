from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    ensure_results_dirs()
    df = load_tess_dataframe(args.data_dir)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["emotion"]
    )

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(train_df["transcript"], train_df["emotion"])

    labels = sorted(df["emotion"].unique().tolist())
    predictions = model.predict(test_df["transcript"])
    accuracy = save_metrics("text_only", test_df["emotion"], predictions, labels)

    joblib.dump({"model": model, "labels": labels}, "Results/checkpoints/text_only.joblib")
    test_df.to_csv("Results/tables/text_only_test_split.csv", index=False)
    print(f"Trained text_only on {len(train_df)} samples and evaluated on {len(test_df)} samples.")
    print(f"Random-split accuracy: {accuracy:.4f}")
    print("Saved checkpoint: Results/checkpoints/text_only.joblib")
    print("Saved test split: Results/tables/text_only_test_split.csv")
    print("Saved metrics: Results/tables/text_only_accuracy.csv and classification report.")


if __name__ == "__main__":
    main()
