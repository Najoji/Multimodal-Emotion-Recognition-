import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from funasr import AutoModel
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs


MODEL_NAME = "iic/emotion2vec_plus_base"


def load_waveform(path: str, sample_rate: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if y.size == 0:
        y = np.zeros(sample_rate, dtype=np.float32)
    return y.astype(np.float32)


def build_or_load_embeddings(
    df: pd.DataFrame,
    sample_rate: int,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        return joblib.load(cache_path)

    model = AutoModel(model=MODEL_NAME, disable_update=True)
    rows = []

    for index, row in df.reset_index(drop=True).iterrows():
        waveform = load_waveform(row["audio_path"], sample_rate=sample_rate)
        result = model.generate(
            waveform,
            granularity="utterance",
            extract_embedding=True,
        )[0]
        rows.append(
            {
                "audio_path": row["audio_path"],
                "transcript": row["transcript"],
                "emotion": row["emotion"],
                "speaker": row["speaker"],
                "embedding": np.asarray(result["feats"], dtype=np.float32),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"embedded {index + 1}/{len(df)} files")

    result_df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result_df, cache_path)
    return result_df


def matrix_from_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.vstack(df["embedding"].to_list())


def main() -> None:
    print("Running strict speaker holdout test for Fusion Pipeline (Speech + Text)...")
    
    ensure_results_dirs()
    
    # Load dataset
    df = load_tess_dataframe("data")
    
    # 1. Look into the index first (in case paths are stored as the dataframe index)
    index_str = df.index.to_series().astype(str)
    if index_str.str.contains('OAF|YAF').any():
        df['speaker'] = index_str.apply(lambda x: 'OAF' if 'OAF' in str(x) else 'YAF')
    else:
        # 2. Look into every column to find where 'OAF' or 'YAF' is mentioned
        speaker_col = None
        for col in df.columns:
            if df[col].astype(str).str.contains('OAF|YAF').any():
                speaker_col = col
                break
        
        if speaker_col:
            df['speaker'] = df[speaker_col].apply(lambda x: 'OAF' if 'OAF' in str(x) else 'YAF')
        else:
            raise ValueError(f"Could not find 'OAF' or 'YAF' in columns or index. Available columns: {list(df.columns)}")
    
    print(f"Loaded {len(df)} TESS samples")
    print(f"OAF samples: {len(df[df['speaker'] == 'OAF'])}")
    print(f"YAF samples: {len(df[df['speaker'] == 'YAF'])}")
    
    # Build or load speech embeddings
    embedding_cache = ROOT / "Results" / "embedding_cache" / "emotion2vec_plus_base_embeddings_fusion.joblib"
    embeddings_df = build_or_load_embeddings(
        df,
        sample_rate=16000,
        cache_path=embedding_cache,
    )
    
    labels = sorted(embeddings_df["emotion"].unique().tolist())
    results = []
    
    # Evaluate both directions
    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        print(f"\n{'='*60}")
        print(f"Train on {train_speaker} -> Test on {test_speaker}")
        print(f"{'='*60}")
        
        train_df = embeddings_df[embeddings_df["speaker"] == train_speaker]
        test_df = embeddings_df[embeddings_df["speaker"] == test_speaker]
        
        # --- Speech Features Block ---
        x_train_speech = matrix_from_embeddings(train_df)
        x_test_speech = matrix_from_embeddings(test_df)
        
        # Normalize speech embeddings
        speech_scaler = StandardScaler()
        x_train_speech = speech_scaler.fit_transform(x_train_speech)
        x_test_speech = speech_scaler.transform(x_test_speech)
        
        speech_normalizer = Normalizer(norm="l2")
        x_train_speech = speech_normalizer.fit_transform(x_train_speech)
        x_test_speech = speech_normalizer.transform(x_test_speech)
        
        print(f"Speech features shape - Train: {x_train_speech.shape}, Test: {x_test_speech.shape}")
        
        # --- Text Features Block ---
        tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        x_train_text = tfidf.fit_transform(train_df["transcript"].values)
        x_test_text = tfidf.transform(test_df["transcript"].values)
        
        print(f"Text features shape - Train: {x_train_text.shape}, Test: {x_test_text.shape}")
        
        # --- Fusion Block ---
        # Concatenate speech (dense) and text (sparse) features
        x_train_fused = hstack([x_train_speech, x_train_text])
        x_test_fused = hstack([x_test_speech, x_test_text])
        
        print(f"Fused features shape - Train: {x_train_fused.shape}, Test: {x_test_fused.shape}")
        
        # --- Classifier ---
        model = LinearSVC(
            C=0.1,
            class_weight="balanced",
            random_state=42,
            dual="auto",
            max_iter=10000,
        )
        model.fit(x_train_fused, train_df["emotion"].values)
        predictions = model.predict(x_test_fused)
        
        # --- Evaluation ---
        accuracy = accuracy_score(test_df["emotion"].values, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        
        results.append({
            "train_speaker": train_speaker,
            "test_speaker": test_speaker,
            "accuracy": accuracy,
        })
        
        # Classification report
        report = classification_report(
            test_df["emotion"].values,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_path = ROOT / "Results" / "tables" / f"fusion_speaker_holdout_{train_speaker}_to_{test_speaker}_classification_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path)
        print(f"Saved classification report to: {report_path.name}")
    
    # --- Summary ---
    results_df = pd.DataFrame(results)
    average_accuracy = results_df["accuracy"].mean()
    print(f"\n{'='*60}")
    print(f"Final Average Speaker-Holdout Accuracy: {average_accuracy:.4f}")
    print(f"{'='*60}")
    
    # Save results
    output_path = ROOT / "Results" / "tables" / "fusion_speaker_holdout_accuracy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
