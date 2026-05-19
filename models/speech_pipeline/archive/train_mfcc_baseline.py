from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.audio_features import build_augmented_enhanced_feature_matrix, build_feature_matrix
from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--feature-set",
        choices=["mfcc", "mfcc_prosody", "enhanced", "enhanced_pitch"],
        default="mfcc",
    )
    parser.add_argument("--classifier", choices=["logistic_regression", "linear_svm"], default="logistic_regression")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--output-name", default="speech_only")
    args = parser.parse_args()

    ensure_results_dirs()
    df = load_tess_dataframe(args.data_dir)

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["emotion"]
    )

    if args.augment and args.feature_set == "enhanced":
        x_train, y_train = build_augmented_enhanced_feature_matrix(
            train_df["audio_path"].tolist(),
            train_df["emotion"].tolist(),
            sample_rate=args.sample_rate,
        )
    else:
        x_train = build_feature_matrix(
            train_df["audio_path"].tolist(),
            sample_rate=args.sample_rate,
            feature_set=args.feature_set,
        )
        y_train = train_df["emotion"]

    x_test = build_feature_matrix(
        test_df["audio_path"].tolist(),
        sample_rate=args.sample_rate,
        feature_set=args.feature_set,
    )

    if args.classifier == "linear_svm":
        classifier = LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=10000)
    else:
        classifier = LogisticRegression(max_iter=3000, class_weight="balanced")

    model = make_pipeline(StandardScaler(), classifier)
    model.fit(x_train, y_train)

    labels = sorted(df["emotion"].unique().tolist())
    predictions = model.predict(x_test)
    accuracy = save_metrics(args.output_name, test_df["emotion"], predictions, labels)

    joblib.dump(
        {
            "model": model,
            "labels": labels,
            "sample_rate": args.sample_rate,
            "feature_set": args.feature_set,
            "classifier": args.classifier,
            "augment": args.augment,
        },
        f"Results/checkpoints/{args.output_name}.joblib",
    )
    test_df.to_csv(f"Results/tables/{args.output_name}_test_split.csv", index=False)
    print(f"Trained {args.output_name} on {len(train_df)} samples and evaluated on {len(test_df)} samples.")
    print(f"Random-split accuracy: {accuracy:.4f}")
    print(f"Saved checkpoint: Results/checkpoints/{args.output_name}.joblib")
    print(f"Saved test split: Results/tables/{args.output_name}_test_split.csv")
    print(f"Saved metrics: Results/tables/{args.output_name}_accuracy.csv and classification report.")


if __name__ == "__main__":
    main()
