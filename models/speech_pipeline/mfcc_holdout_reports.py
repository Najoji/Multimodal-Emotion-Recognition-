from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.audio_features import build_audio_feature_matrix
from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


def main() -> None:
    ensure_results_dirs()
    df = load_tess_dataframe("data")
    df["speaker"] = df["audio_path"].str.extract(r"([OY]AF)_")[0]
    labels = sorted(df["emotion"].unique().tolist())

    feature_cache = dict(
        zip(
            df["audio_path"],
            build_audio_feature_matrix(df["audio_path"].tolist()),
        )
    )

    rows = []
    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        train_df = df[df["speaker"] == train_speaker]
        test_df = df[df["speaker"] == test_speaker]
        x_train = pd.DataFrame(
            [feature_cache[path] for path in train_df["audio_path"].tolist()]
        ).to_numpy()
        x_test = pd.DataFrame(
            [feature_cache[path] for path in test_df["audio_path"].tolist()]
        ).to_numpy()

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"),
        )
        model.fit(x_train, train_df["emotion"])
        predictions = model.predict(x_test)
        accuracy = accuracy_score(test_df["emotion"], predictions)
        rows.append(
            {
                "train_speaker": train_speaker,
                "test_speaker": test_speaker,
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
            ROOT
            / "Results"
            / "tables"
            / f"speech_stage1_mfcc_baseline_{train_speaker}_to_{test_speaker}_classification_report.csv"
        )

    pd.DataFrame(rows).to_csv(
        ROOT / "Results" / "tables" / "speech_stage1_mfcc_baseline.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
