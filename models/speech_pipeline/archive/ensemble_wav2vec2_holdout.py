from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.evaluation import ensure_results_dirs


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_list())


def main() -> None:
    ensure_results_dirs()

    # Load cached wav2vec2 embeddings
    cache_path = Path("Results/embedding_cache/wav2vec2_base_embeddings.joblib")
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cached embeddings not found at {cache_path}. "
            "Please run pretrained_embedding_holdout.py first."
        )

    print("Loading cached wav2vec2 embeddings...")
    embeddings = joblib.load(cache_path)

    labels = sorted(embeddings["emotion"].unique().tolist())
    rows = []
    all_predictions = []

    print("\nTesting ensemble classifiers on speaker-holdout evaluation...")
    print("=" * 60)

    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        print(f"\nTraining on {train_speaker}, testing on {test_speaker}...")

        train_df = embeddings[embeddings["speaker"] == train_speaker]
        test_df = embeddings[embeddings["speaker"] == test_speaker]
        x_train = matrix_from_embeddings(train_df)
        x_test = matrix_from_embeddings(test_df)
        y_train = train_df["emotion"]
        y_test = test_df["emotion"]

        # Create ensemble with multiple classifiers
        ensemble = VotingClassifier(
            estimators=[
                ("svm", LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=10000)),
                ("lr", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)),
            ],
            voting="hard",  # Use majority voting instead of soft voting
        )

        # Fit pipeline with scaler + ensemble
        model = make_pipeline(StandardScaler(), ensemble)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        rows.append(
            {
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
                "model_name": "ensemble_wav2vec2",
                "classifier": "Voting (SVM+LR+RF)",
                "accuracy": accuracy,
            }
        )

        all_predictions.append(
            pd.DataFrame(
                {
                    "audio_path": test_df["audio_path"].values,
                    "true": y_test.values,
                    "predicted": predictions,
                    "train_speaker": train_speaker,
                    "test_speaker": test_speaker,
                }
            )
        )

        report = classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            f"Results/tables/ensemble_wav2vec2_{train_speaker}_to_{test_speaker}_classification_report.csv"
        )
        pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels)).to_csv(
            f"Results/tables/ensemble_wav2vec2_{train_speaker}_to_{test_speaker}_confusion_matrix.csv",
            index=False,
        )

        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/ensemble_wav2vec2_speaker_holdout.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(
        "Results/tables/ensemble_wav2vec2_speaker_holdout_predictions.csv", index=False
    )

    print("\n" + "=" * 60)
    print(result.to_string(index=False))
    avg_accuracy = result["accuracy"].mean()
    print(f"\nAverage accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
