from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("Results/matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.audio_features import build_audio_feature_matrix
from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


def balanced_sample(df: pd.DataFrame, samples_per_emotion: int, random_state: int = 42) -> pd.DataFrame:
    sampled_parts = []
    for _, group in df.groupby("emotion"):
        sample_size = min(len(group), samples_per_emotion)
        sampled_parts.append(group.sample(sample_size, random_state=random_state))
    return pd.concat(sampled_parts, ignore_index=True)


def plot_2d(points, labels, title: str, output_path: str) -> None:
    plot_df = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "emotion": labels})
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="emotion",
        s=42,
        alpha=0.85,
        edgecolor="none",
    )
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_accuracy_bar_chart() -> None:
    comparison_path = Path("Results/tables/model_comparison.csv")
    if not comparison_path.exists():
        return

    comparison = pd.read_csv(comparison_path)
    comparison["accuracy_percent"] = comparison["accuracy"] * 100

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(data=comparison, x="model", y="accuracy_percent", color="#4f7cac")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Baseline Model Comparison")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", padding=3)
    plt.tight_layout()
    plt.savefig("Results/plots/model_comparison_bar.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--samples-per-emotion", type=int, default=100)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    ensure_results_dirs()
    df = load_tess_dataframe(args.data_dir)
    df = balanced_sample(df, args.samples_per_emotion)

    labels = df["emotion"].tolist()

    speech_features = build_audio_feature_matrix(
        df["audio_path"].tolist(), sample_rate=args.sample_rate
    )
    speech_scaled = StandardScaler().fit_transform(speech_features)
    speech_points = PCA(n_components=2, random_state=42).fit_transform(speech_scaled)
    plot_2d(
        speech_points,
        labels,
        "Speech Feature Space (MFCC PCA)",
        "Results/plots/speech_representation_pca.png",
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    text_features = vectorizer.fit_transform(df["transcript"])
    text_points = TruncatedSVD(n_components=2, random_state=42).fit_transform(text_features)
    plot_2d(
        text_points,
        labels,
        "Text Feature Space (TF-IDF SVD)",
        "Results/plots/text_representation_svd.png",
    )

    fusion_features = hstack([speech_scaled, text_features]).toarray()
    fusion_points = PCA(n_components=2, random_state=42).fit_transform(fusion_features)
    plot_2d(
        fusion_points,
        labels,
        "Fusion Feature Space (Speech + Text PCA)",
        "Results/plots/fusion_representation_pca.png",
    )

    save_accuracy_bar_chart()

    pd.DataFrame(
        [
            {"representation": "speech_mfcc_pca", "samples": len(df)},
            {"representation": "text_tfidf_svd", "samples": len(df)},
            {"representation": "fusion_pca", "samples": len(df)},
        ]
    ).to_csv("Results/tables/representation_plots.csv", index=False)

    print("Saved representation plots and model comparison chart.")


if __name__ == "__main__":
    main()
