from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_list())


def main() -> None:
    cache_path = Path(
        "Results/embedding_cache/wav2vec2_large_robust_12_ft_emotion_msp_dim_embeddings.joblib"
    )
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    embeddings = joblib.load(cache_path)
    rows = []
    c_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    for c_value in c_values:
        for l2_normalize in [False, True]:
            for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
                train_df = embeddings[embeddings["speaker"] == train_speaker]
                test_df = embeddings[embeddings["speaker"] == test_speaker]
                x_train = matrix_from_embeddings(train_df)
                x_test = matrix_from_embeddings(test_df)

                steps = [StandardScaler()]
                if l2_normalize:
                    steps.append(Normalizer(norm="l2"))
                steps.append(
                    LinearSVC(
                        C=c_value,
                        class_weight="balanced",
                        random_state=42,
                        dual="auto",
                        max_iter=10000,
                    )
                )
                model = make_pipeline(*steps)
                model.fit(x_train, train_df["emotion"])
                predictions = model.predict(x_test)
                rows.append(
                    {
                        "c": c_value,
                        "l2_normalize": l2_normalize,
                        "train_speaker": train_speaker,
                        "test_speaker": test_speaker,
                        "accuracy": accuracy_score(test_df["emotion"], predictions),
                    }
                )

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/emotion_ft_svm_sweep.csv", index=False)
    summary = (
        result.groupby(["c", "l2_normalize"])["accuracy"]
        .mean()
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    summary.to_csv("Results/tables/emotion_ft_svm_sweep_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

