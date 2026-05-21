# Speech Pipeline (Emotion2Vec+)

This pipeline is the final speech-only system. It uses Emotion2Vec+ embeddings and a linear SVM with speaker-holdout evaluation.

## How it works

1. Load the TESS audio files and infer speaker IDs (OAF/YAF).
2. Extract Emotion2Vec+ embeddings (`iic/emotion2vec_plus_base`).
3. Train a linear classifier: `StandardScaler -> L2 Normalizer -> LinearSVC`.
4. Evaluate on the held-out speaker.

Embeddings are cached to avoid recomputation.

## Run the model

Train (builds embeddings if missing, saves checkpoints and reports):

```powershell
python models\speech_pipeline\train.py
```

Test (loads checkpoints, recomputes accuracy on the holdout split, saves test reports and confusion matrices):

```powershell
python models\speech_pipeline\test.py
```

## Outputs

- Checkpoints: `Results/checkpoints/speech_only_*_train.joblib`
- Train reports: `Results/tables/speech_only_*_train_report.csv`
- Test reports: `Results/tables/speech_only_*_test_report.csv`
- Test confusion matrices: `Results/plots/speech_only_*_test_confusion_matrix.png`

## Notes

- You may see a notice about ffmpeg not being installed. This is a harmless message from the audio backend; torchaudio is used as a fallback.
- Checkpoints in `Results/checkpoints/` are generated locally and are not committed to GitHub.
- The `archive/` folder contains older baselines and experimental scripts.
