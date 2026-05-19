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


def build_model() -> any:
    return make_pipeline(
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


def select_top_k_per_class(
    predictions: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    top_k: int,
) -> np.ndarray:
    if scores.ndim == 1:
        margins = np.abs(scores)
    else:
        sorted_scores = np.sort(scores, axis=1)
        margins = sorted_scores[:, -1] - sorted_scores[:, -2]

    chosen = []
    for class_name in classes:
        indices = np.where(predictions == class_name)[0]
        if len(indices) == 0:
            continue
        ranked = indices[np.argsort(margins[indices])[::-1]]
        chosen.extend(ranked[:top_k].tolist())
    return np.array(sorted(set(chosen)), dtype=int)


def main() -> None:
    cache_path = Path(
        "Results/embedding_cache/wav2vec2_large_robust_12_ft_emotion_msp_dim_embeddings.joblib"
    )
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    embeddings = joblib.load(cache_path)
    rows = []

    for top_k in [10, 20, 30, 40, 50, 75, 100]:
        for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
            train_df = embeddings[embeddings["speaker"] == train_speaker]
            test_df = embeddings[embeddings["speaker"] == test_speaker]
            x_train = matrix_from_embeddings(train_df)
            x_test = matrix_from_embeddings(test_df)

            first_model = build_model()
            first_model.fit(x_train, train_df["emotion"])
            first_predictions = first_model.predict(x_test)
            scores = first_model.decision_function(x_test)
            selected = select_top_k_per_class(
                first_predictions,
                scores,
                first_model.classes_,
                top_k=top_k,
            )

            x_adapt = np.vstack([x_train, x_test[selected]])
            y_adapt = np.concatenate([train_df["emotion"].to_numpy(), first_predictions[selected]])

            adapted_model = build_model()
            adapted_model.fit(x_adapt, y_adapt)
            adapted_predictions = adapted_model.predict(x_test)

            rows.append(
                {
                    "top_k_per_predicted_class": top_k,
                    "train_speaker": train_speaker,
                    "test_speaker": test_speaker,
                    "selected_target_samples": len(selected),
                    "baseline_accuracy": accuracy_score(test_df["emotion"], first_predictions),
                    "adapted_accuracy": accuracy_score(test_df["emotion"], adapted_predictions),
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/emotion_ft_pseudolabel_adaptation.csv", index=False)
    summary = (
        result.groupby("top_k_per_predicted_class")[["baseline_accuracy", "adapted_accuracy"]]
        .mean()
        .reset_index()
        .sort_values("adapted_accuracy", ascending=False)
    )
    summary.to_csv("Results/tables/emotion_ft_pseudolabel_adaptation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

