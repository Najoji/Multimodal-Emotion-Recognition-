import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, Normalizer
from scipy.sparse import hstack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.evaluation import ensure_results_dirs
from speech_emotion.audio_features import build_audio_feature_matrix


def load_embeddings():
    """Load the cached Emotion2Vec+ embeddings."""
    cache_path = ROOT / "Results" / "embedding_cache" / "emotion2vec_plus_base_embeddings_fusion.joblib"
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    df = joblib.load(cache_path)
    print(f"Loaded {len(df)} embeddings from cache")
    return df


def plot_speech_representations(df):
    """Plot 1: Speech embeddings via PCA (Temporal Modelling Block)."""
    print("\nGenerating Speech Representation Plot (PCA)...")
    
    # Extract embeddings
    embeddings = np.vstack(df["embedding"].to_list())
    print(f"  Speech embedding shape: {embeddings.shape}")
    
    # Normalize
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    normalizer = Normalizer(norm="l2")
    embeddings_normalized = normalizer.fit_transform(embeddings_scaled)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_normalized)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    emotions = df["emotion"].values
    unique_emotions = sorted(set(emotions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_emotions)))
    emotion_to_color = {emotion: colors[i] for i, emotion in enumerate(unique_emotions)}
    
    for emotion in unique_emotions:
        mask = emotions == emotion
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                  label=emotion, alpha=0.6, s=50, color=emotion_to_color[emotion])
    
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Temporal Modelling Block: Emotion2Vec+ Speech Representations (PCA)")
    ax.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = ROOT / "Results" / "plots" / "speech_representation_pca.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {output_path.relative_to(ROOT)}")
    plt.close()


def plot_text_representations(df):
    """Plot 2: Text TF-IDF via SVD (Contextual Modelling Block)."""
    print("\nGenerating Text Representation Plot (SVD)...")
    
    # Extract and vectorize text
    transcripts = df["transcript"].values
    tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    tfidf_matrix = tfidf.fit_transform(transcripts)
    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # SVD to 2D
    svd = TruncatedSVD(n_components=2)
    tfidf_2d = svd.fit_transform(tfidf_matrix)
    print(f"  SVD explained variance: {svd.explained_variance_ratio_}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    emotions = df["emotion"].values
    unique_emotions = sorted(set(emotions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_emotions)))
    emotion_to_color = {emotion: colors[i] for i, emotion in enumerate(unique_emotions)}
    
    for emotion in unique_emotions:
        mask = emotions == emotion
        ax.scatter(tfidf_2d[mask, 0], tfidf_2d[mask, 1], 
                  label=emotion, alpha=0.6, s=50, color=emotion_to_color[emotion])
    
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    ax.set_title("Contextual Modelling Block: TF-IDF Text Representations (SVD)")
    ax.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = ROOT / "Results" / "plots" / "text_representation_svd.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {output_path.relative_to(ROOT)}")
    plt.close()


def plot_fusion_representations(df):
    """Plot 3: Fusion (Speech + Text) via PCA (Fusion Block)."""
    print("\nGenerating Fusion Representation Plot (PCA)...")
    
    # Speech embeddings (dense)
    embeddings = np.vstack(df["embedding"].to_list())
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    normalizer = Normalizer(norm="l2")
    embeddings_normalized = normalizer.fit_transform(embeddings_scaled)
    print(f"  Speech embedding shape: {embeddings_normalized.shape}")
    
    # Text TF-IDF (sparse)
    transcripts = df["transcript"].values
    tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    tfidf_matrix = tfidf.fit_transform(transcripts)
    print(f"  Text TF-IDF shape: {tfidf_matrix.shape}")
    
    # Fusion: concatenate dense and sparse
    fused_matrix = hstack([embeddings_normalized, tfidf_matrix])
    fused_dense = fused_matrix.toarray()
    print(f"  Fused matrix shape (dense): {fused_dense.shape}")
    
    # PCA to 2D
    pca = PCA(n_components=2)
    fused_2d = pca.fit_transform(fused_dense)
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    emotions = df["emotion"].values
    unique_emotions = sorted(set(emotions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_emotions)))
    emotion_to_color = {emotion: colors[i] for i, emotion in enumerate(unique_emotions)}
    
    for emotion in unique_emotions:
        mask = emotions == emotion
        ax.scatter(fused_2d[mask, 0], fused_2d[mask, 1], 
                  label=emotion, alpha=0.6, s=50, color=emotion_to_color[emotion])
    
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("Multimodal Fusion Block: Combined Speech & Text Representations (PCA)")
    ax.legend(title="Emotion", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = ROOT / "Results" / "plots" / "fusion_representation_pca.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {output_path.relative_to(ROOT)}")
    plt.close()


def project_dense_features(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    normalizer = Normalizer(norm="l2")
    features_normalized = normalizer.fit_transform(features_scaled)
    return PCA(n_components=2).fit_transform(features_normalized)


def plot_speech_model_evolution(df):
    """Plot 4: Compare MFCC, Generic Wav2Vec2, Emotion-Finetuned Wav2Vec2, and Emotion2Vec+ feature geometry."""
    print("\nGenerating Speech Model Evolution Plot (All 4 Stages)...")

    mfcc_features = build_audio_feature_matrix(df["audio_path"].tolist())
    wav2vec_df = joblib.load(ROOT / "Results" / "embedding_cache" / "wav2vec2_base_embeddings.joblib")
    emotion_ft_df = joblib.load(ROOT / "Results" / "embedding_cache" / "wav2vec2_large_robust_12_ft_emotion_msp_dim_embeddings.joblib")
    emotion2vec_features = np.vstack(df["embedding"].to_list())
    
    wav2vec_features = np.vstack(wav2vec_df["embedding"].to_list())
    emotion_ft_features = np.vstack(emotion_ft_df["embedding"].to_list())

    projections = [
        ("Stage 1: MFCC Baseline", project_dense_features(mfcc_features)),
        ("Stage 2: Generic Wav2Vec2", project_dense_features(wav2vec_features)),
        ("Stage 3: Emotion-Finetuned Wav2Vec2", project_dense_features(emotion_ft_features)),
        ("Stage 4: Emotion2Vec+ Champion", project_dense_features(emotion2vec_features)),
    ]

    emotions = df["emotion"].values
    unique_emotions = sorted(set(emotions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_emotions)))
    emotion_to_color = {emotion: colors[i] for i, emotion in enumerate(unique_emotions)}

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=False, sharey=False)
    for ax, (title, points) in zip(axes, projections):
        for emotion in unique_emotions:
            mask = emotions == emotion
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                label=emotion,
                alpha=0.55,
                s=18,
                color=emotion_to_color[emotion],
            )
        ax.set_title(title)
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.grid(True, alpha=0.25)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, title="Emotion", loc="center right", bbox_to_anchor=(1.12, 0.5))
    fig.suptitle("Speech Representation Evolution Across All 4 Model Stages")
    plt.tight_layout(rect=[0, 0, 0.92, 0.94])

    output_path = ROOT / "Results" / "plots" / "speech_model_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved to: {output_path.relative_to(ROOT)}")
    plt.close()


def main():
    print("Visualizing Final Advanced Pipeline Representations...")
    print("=" * 60)
    
    ensure_results_dirs()
    
    # Load data
    df = load_embeddings()
    
    # Generate all three plots
    plot_speech_representations(df)
    plot_text_representations(df)
    plot_fusion_representations(df)
    plot_speech_model_evolution(df)
    
    print("\n" + "=" * 60)
    print("All visualizations completed successfully!")
    print(f"Plots saved to: {(ROOT / 'Results' / 'plots').relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
