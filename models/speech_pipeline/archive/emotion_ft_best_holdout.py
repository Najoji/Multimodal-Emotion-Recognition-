from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
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
    labels = sorted(embeddings["emotion"].unique().tolist())
    rows = []

    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        train_df = embeddings[embeddings["speaker"] == train_speaker]
        test_df = embeddings[embeddings["speaker"] == test_speaker]
        x_train = matrix_from_embeddings(train_df)
        x_test = matrix_from_embeddings(test_df)

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
        accuracy = accuracy_score(test_df["emotion"], predictions)

        rows.append(
            {
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
                "model_name": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                "classifier": "linear_svm_c_0.1_l2",
                "accuracy": accuracy,
            }
        )

        report = classification_report(
            test_df["emotion"],
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            f"Results/tables/speech_stage3_emotion_finetuned_{train_speaker}_to_{test_speaker}_classification_report.csv"
        )

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/speech_stage3_emotion_finetuned_holdout.csv", index=False)
    print(result.to_string(index=False))
    print(f"average accuracy: {result['accuracy'].mean():.4f}")


if __name__ == "__main__":
    main()
