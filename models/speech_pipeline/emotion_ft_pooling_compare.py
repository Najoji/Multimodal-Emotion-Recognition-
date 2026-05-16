from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC


def split_embedding_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    matrix = np.vstack(df["embedding"].to_list())
    half = matrix.shape[1] // 2
    return {
        "mean_only": matrix[:, :half],
        "std_only": matrix[:, half:],
        "mean_plus_std": matrix,
    }


def main() -> None:
    cache_path = Path(
        "Results/embedding_cache/wav2vec2_large_robust_12_ft_emotion_msp_dim_embeddings.joblib"
    )
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    embeddings = joblib.load(cache_path)
    rows = []

    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        train_df = embeddings[embeddings["speaker"] == train_speaker]
        test_df = embeddings[embeddings["speaker"] == test_speaker]
        train_views = split_embedding_matrix(train_df)
        test_views = split_embedding_matrix(test_df)

        for pooling_name in train_views:
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
            model.fit(train_views[pooling_name], train_df["emotion"])
            predictions = model.predict(test_views[pooling_name])
            rows.append(
                {
                    "pooling": pooling_name,
                    "train_speaker": train_speaker,
                    "test_speaker": test_speaker,
                    "accuracy": accuracy_score(test_df["emotion"], predictions),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/emotion_ft_pooling_compare.csv", index=False)
    summary = (
        result.groupby("pooling")["accuracy"]
        .mean()
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    summary.to_csv("Results/tables/emotion_ft_pooling_compare_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

