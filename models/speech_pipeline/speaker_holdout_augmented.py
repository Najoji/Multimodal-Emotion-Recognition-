from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.audio_features import build_augmented_enhanced_feature_matrix, build_feature_matrix
from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    ensure_results_dirs()
    df = load_tess_dataframe(args.data_dir)
    df["speaker"] = df["audio_path"].str.extract(r"([OY]AF)_")[0]

    rows = []
    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        train_df = df[df["speaker"] == train_speaker]
        test_df = df[df["speaker"] == test_speaker]

        x_train, y_train = build_augmented_enhanced_feature_matrix(
            train_df["audio_path"].tolist(),
            train_df["emotion"].tolist(),
            sample_rate=args.sample_rate,
        )
        x_test = build_feature_matrix(
            test_df["audio_path"].tolist(),
            sample_rate=args.sample_rate,
            feature_set="enhanced",
        )

        model = make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=10000),
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append(
            {
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
                "model": "linear_svm",
                "feature_set": "enhanced",
                "augmentation": "noise_gain_pitch_shift_time_stretch",
                "accuracy": accuracy_score(test_df["emotion"], predictions),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv("Results/tables/speaker_holdout_augmented.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

