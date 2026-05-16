from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.audio_features import build_audio_feature_matrix
from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    ensure_results_dirs()
    df = load_tess_dataframe(args.data_dir)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["emotion"]
    )

    scaler = StandardScaler()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)

    speech_train = scaler.fit_transform(
        build_audio_feature_matrix(train_df["audio_path"].tolist(), sample_rate=args.sample_rate)
    )
    speech_test = scaler.transform(
        build_audio_feature_matrix(test_df["audio_path"].tolist(), sample_rate=args.sample_rate)
    )
    text_train = vectorizer.fit_transform(train_df["transcript"])
    text_test = vectorizer.transform(test_df["transcript"])

    x_train = hstack([csr_matrix(speech_train), text_train])
    x_test = hstack([csr_matrix(speech_test), text_test])

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(x_train, train_df["emotion"])

    labels = sorted(df["emotion"].unique().tolist())
    predictions = classifier.predict(x_test)
    save_metrics("fusion", test_df["emotion"], predictions, labels)

    joblib.dump(
        {
            "scaler": scaler,
            "vectorizer": vectorizer,
            "classifier": classifier,
            "labels": labels,
            "sample_rate": args.sample_rate,
        },
        "Results/checkpoints/fusion.joblib",
    )
    test_df.to_csv("Results/tables/fusion_test_split.csv", index=False)
    print("Saved fusion model and metrics.")


if __name__ == "__main__":
    main()
