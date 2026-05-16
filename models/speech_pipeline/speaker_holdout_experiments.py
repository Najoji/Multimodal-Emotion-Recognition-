from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.audio_features import build_audio_feature_matrix, build_mfcc_prosody_feature_matrix
from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


def per_sample_standardize(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-8)


def select_features(feature_cache: dict[str, np.ndarray], paths: list[str]) -> np.ndarray:
    return np.vstack([feature_cache[path] for path in paths])


def evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cache: dict[str, np.ndarray],
    feature_set: str,
    normalize_per_sample: bool,
):
    x_train = select_features(feature_cache, train_df["audio_path"].tolist())
    x_test = select_features(feature_cache, test_df["audio_path"].tolist())

    if normalize_per_sample:
        x_train = per_sample_standardize(x_train)
        x_test = per_sample_standardize(x_test)

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=10000),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    rows = []
    for name, model in models.items():
        model.fit(x_train, train_df["emotion"])
        predictions = model.predict(x_test)
        rows.append(
            {
                "model": name,
                "feature_set": feature_set,
                "normalize_per_sample": normalize_per_sample,
                "accuracy": accuracy_score(test_df["emotion"], predictions),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    ensure_results_dirs()
    df = load_tess_dataframe(args.data_dir)
    df["speaker"] = df["audio_path"].str.extract(r"([OY]AF)_")[0]

    rows = []
    feature_caches = {
        "mfcc": dict(
            zip(
                df["audio_path"],
                build_audio_feature_matrix(df["audio_path"].tolist()),
            )
        ),
        "mfcc_prosody": dict(
            zip(
                df["audio_path"],
                build_mfcc_prosody_feature_matrix(df["audio_path"].tolist()),
            )
        ),
    }

    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        train_df = df[df["speaker"] == train_speaker]
        test_df = df[df["speaker"] == test_speaker]
        for feature_set in ["mfcc", "mfcc_prosody"]:
            for normalize_per_sample in [False, True]:
                for row in evaluate(
                    train_df,
                    test_df,
                    feature_caches[feature_set],
                    feature_set,
                    normalize_per_sample,
                ):
                    row["train_speaker"] = train_speaker
                    row["test_speaker"] = test_speaker
                    rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/speaker_holdout_experiments.csv", index=False)

    summary = (
        result.groupby(["model", "feature_set", "normalize_per_sample"])["accuracy"]
        .mean()
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    summary.to_csv("Results/tables/speaker_holdout_experiments_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
