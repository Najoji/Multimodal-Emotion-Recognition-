from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def main() -> None:
    df = joblib.load("Results/embedding_cache/wav2vec2_base_embeddings.joblib")
    x = np.vstack(df["embedding"].to_list())
    y = df["emotion"].values
    groups = df["speaker"].values

    output_dir = Path("Results/tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Random 80/20 split
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(class_weight="balanced", random_state=42, dual="auto", max_iter=10000),
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    metrics = {
        "split": "random_80_20",
        "accuracy": accuracy_score(y_test, preds),
        "balanced_accuracy": balanced_accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
    }
    pd.DataFrame([metrics]).to_csv(
        output_dir / "wav2vec2_base_random80_20_metrics.csv", index=False
    )

    labels = sorted(np.unique(y))
    pd.DataFrame(
        confusion_matrix(y_test, preds, labels=labels),
        index=labels,
        columns=labels,
    ).to_csv(output_dir / "wav2vec2_base_random80_20_confusion_matrix.csv")

    # GroupKFold by speaker (2 folds)
    rows = []
    gkf = GroupKFold(n_splits=2)
    for fold, (tr, te) in enumerate(gkf.split(x, y, groups=groups), start=1):
        model = make_pipeline(
            StandardScaler(),
            LinearSVC(
                class_weight="balanced", random_state=42, dual="auto", max_iter=10000
            ),
        )
        model.fit(x[tr], y[tr])
        p = model.predict(x[te])
        rows.append(
            {
                "split": f"groupkfold_speaker_fold{fold}",
                "accuracy": accuracy_score(y[te], p),
                "balanced_accuracy": balanced_accuracy_score(y[te], p),
                "macro_f1": f1_score(y[te], p, average="macro"),
            }
        )

    pd.DataFrame(rows).to_csv(
        output_dir / "wav2vec2_base_groupkfold_metrics.csv", index=False
    )

    print(pd.DataFrame([metrics]))
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
