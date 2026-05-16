from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    ensure_results_dirs()
    df = load_tess_dataframe(args.data_dir)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["emotion"]
    )

    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=42),
        "tfidf_logistic_regression": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "tfidf_naive_bayes": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
            MultinomialNB(),
        ),
        "tfidf_linear_svm": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
            LinearSVC(class_weight="balanced", random_state=42, dual="auto"),
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(train_df["transcript"], train_df["emotion"])
        predictions = model.predict(test_df["transcript"])
        rows.append({"model": name, "accuracy": accuracy_score(test_df["emotion"], predictions)})

    pd.DataFrame(rows).to_csv("Results/tables/text_baseline_comparison.csv", index=False)
    print("Saved text baseline comparison.")


if __name__ == "__main__":
    main()
