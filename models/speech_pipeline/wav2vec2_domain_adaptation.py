from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.evaluation import ensure_results_dirs


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_list())


def center_separately(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x_train - x_train.mean(axis=0), x_test - x_test.mean(axis=0)


def zscore_separately(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_std = np.maximum(x_train.std(axis=0), 1e-8)
    test_std = np.maximum(x_test.std(axis=0), 1e-8)
    return (
        (x_train - x_train.mean(axis=0)) / train_std,
        (x_test - x_test.mean(axis=0)) / test_std,
    )


def covariance_sqrt(matrix: np.ndarray, inverse: bool = False) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 1e-6)
    if inverse:
        values = 1.0 / np.sqrt(values)
    else:
        values = np.sqrt(values)
    return vectors @ np.diag(values) @ vectors.T


def coral_source_to_target(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_mean = x_train.mean(axis=0)
    test_mean = x_test.mean(axis=0)
    train_centered = x_train - train_mean
    test_centered = x_test - test_mean

    cov_train = np.cov(train_centered, rowvar=False) + np.eye(x_train.shape[1]) * 1e-3
    cov_test = np.cov(test_centered, rowvar=False) + np.eye(x_test.shape[1]) * 1e-3

    transform = covariance_sqrt(cov_train, inverse=True) @ covariance_sqrt(cov_test)
    aligned_train = train_centered @ transform
    return aligned_train, test_centered


def fit_predict(x_train: np.ndarray, y_train, x_test: np.ndarray, use_normalizer: bool) -> np.ndarray:
    steps = [StandardScaler()]
    if use_normalizer:
        steps.append(Normalizer(norm="l2"))
    steps.append(LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=10000))
    model = make_pipeline(*steps)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-path",
        default="Results/embedding_cache/wav2vec2_base_embeddings.joblib",
    )
    parser.add_argument("--output-prefix", default="speech_stage2_wav2vec2")
    args = parser.parse_args()

    ensure_results_dirs()
    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Expected cached Wav2Vec2 embeddings at {cache_path}. "
            "Run pretrained_embedding_holdout.py first."
        )

    embeddings = joblib.load(cache_path)
    rows = []

    transforms = {
        "baseline": lambda train, test: (train, test),
        "separate_centering": center_separately,
        "separate_zscore": zscore_separately,
        "coral_source_to_target": coral_source_to_target,
    }

    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        train_df = embeddings[embeddings["speaker"] == train_speaker]
        test_df = embeddings[embeddings["speaker"] == test_speaker]
        x_train = matrix_from_embeddings(train_df)
        x_test = matrix_from_embeddings(test_df)

        for transform_name, transform in transforms.items():
            transformed_train, transformed_test = transform(x_train, x_test)
            for use_normalizer in [False, True]:
                predictions = fit_predict(
                    transformed_train,
                    train_df["emotion"],
                    transformed_test,
                    use_normalizer=use_normalizer,
                )
                rows.append(
                    {
                        "train_speaker": train_speaker,
                        "test_speaker": test_speaker,
                        "transform": transform_name,
                        "l2_normalize": use_normalizer,
                        "accuracy": accuracy_score(test_df["emotion"], predictions),
                    }
                )

    result = pd.DataFrame(rows)
    detail_path = f"Results/tables/{args.output_prefix}_details.csv"
    summary_path = f"Results/tables/{args.output_prefix}_adaptation.csv"
    result.to_csv(detail_path, index=False)
    summary = (
        result.groupby(["transform", "l2_normalize"])["accuracy"]
        .mean()
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
