from __future__ import annotations

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

from speech_emotion.audio_features import build_feature_matrix


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_list())


def main() -> None:
    cache_path = Path(
        "Results/embedding_cache/wav2vec2_large_robust_12_ft_emotion_msp_dim_embeddings.joblib"
    )
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    embeddings = joblib.load(cache_path)
    handcrafted = build_feature_matrix(
        embeddings["audio_path"].tolist(),
        feature_set="enhanced",
    )
    handcrafted_by_path = dict(zip(embeddings["audio_path"], handcrafted))

    rows = []
    for feature_mode in ["embedding_only", "handcrafted_only", "embedding_plus_handcrafted"]:
        for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
            train_df = embeddings[embeddings["speaker"] == train_speaker]
            test_df = embeddings[embeddings["speaker"] == test_speaker]

            embedding_train = matrix_from_embeddings(train_df)
            embedding_test = matrix_from_embeddings(test_df)
            handcrafted_train = np.vstack(
                [handcrafted_by_path[path] for path in train_df["audio_path"]]
            )
            handcrafted_test = np.vstack(
                [handcrafted_by_path[path] for path in test_df["audio_path"]]
            )

            if feature_mode == "embedding_only":
                x_train = embedding_train
                x_test = embedding_test
            elif feature_mode == "handcrafted_only":
                x_train = handcrafted_train
                x_test = handcrafted_test
            else:
                x_train = np.hstack([embedding_train, handcrafted_train])
                x_test = np.hstack([embedding_test, handcrafted_test])

            model = make_pipeline(
                StandardScaler(),
                Normalizer(norm="l2"),
                LinearSVC(
                    C=0.1,
                    class_weight="balanced",
                    random_state=42,
                    dual="auto",
                    max_iter=10000,
                ),
            )
            model.fit(x_train, train_df["emotion"])
            predictions = model.predict(x_test)
            rows.append(
                {
                    "feature_mode": feature_mode,
                    "train_speaker": train_speaker,
                    "test_speaker": test_speaker,
                    "accuracy": accuracy_score(test_df["emotion"], predictions),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/emotion_ft_feature_fusion.csv", index=False)
    summary = (
        result.groupby("feature_mode")["accuracy"]
        .mean()
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    summary.to_csv("Results/tables/emotion_ft_feature_fusion_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

