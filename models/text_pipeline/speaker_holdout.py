import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

# Setup paths for custom imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from speech_emotion.dataset import load_tess_dataframe
from speech_emotion.evaluation import ensure_results_dirs

def main():
    print("Running strict speaker holdout test for Text Baseline...")
    
    # Ensure our results directories exist just in case
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
            # Diagnostic fallback if it absolutely can't find it
            raise ValueError(f"Could not find 'OAF' or 'YAF' in columns or index. Available columns: {list(df.columns)}")

    # Proceed with the speaker splits
    oaf_df = df[df['speaker'] == 'OAF']
    yaf_df = df[df['speaker'] == 'YAF']
    
    results = []
    
    # Text baseline pipeline as requested
    def build_model():
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
            LogisticRegression(max_iter=2000, class_weight="balanced")
        )
    
    # Evaluate both directions
    for train_speaker, test_speaker in [("OAF", "YAF"), ("YAF", "OAF")]:
        train_df = df[df["speaker"] == train_speaker]
        test_df = df[df["speaker"] == test_speaker]
        
        x_train = train_df["transcript"]
        y_train = train_df["emotion"]
        
        x_test = test_df["transcript"]
        y_test = test_df["emotion"]
        
        # Train and Predict
        model = build_model()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, predictions)
        print(f"Train on {train_speaker} -> Test on {test_speaker} | Accuracy: {accuracy:.4f}")
        
        results.append({
            "train_speaker": train_speaker,
            "test_speaker": test_speaker,
            "accuracy": accuracy
        })
        
    # Summarize and Save
    results_df = pd.DataFrame(results)
    average_accuracy = results_df["accuracy"].mean()
    print(f"\nFinal Average Holdout Accuracy: {average_accuracy:.4f}")
    
    output_path = ROOT / "Results" / "tables" / "text_speaker_holdout_accuracy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Saved results cleanly to: {output_path.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
