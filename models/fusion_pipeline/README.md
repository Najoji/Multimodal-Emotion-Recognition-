# Fusion Pipeline (Speech + Text)

This pipeline combines speech embeddings with TF-IDF text features. It is evaluated under the same speaker-holdout protocol as the other models.

## How it works

1. Extract Emotion2Vec+ embeddings for speech.
2. Standardize and L2-normalize the speech embeddings.
3. Compute TF-IDF text features (unigrams + bigrams).
4. Concatenate speech and text features (`hstack`).
5. Train a balanced Linear SVM.

## Run the model

Train:

```powershell
python models\fusion_pipeline\train.py
```

Test (reloads checkpoints, recomputes accuracy on the holdout split, and saves reports/plots):

```powershell
python models\fusion_pipeline\test.py
```

## Outputs

- Checkpoints: `Results/checkpoints/fusion_*_train.joblib`
- Train reports: `Results/tables/fusion_*_train_report.csv`
- Test reports: `Results/tables/fusion_*_test_report.csv`
- Test confusion matrices: `Results/plots/fusion_*_test_confusion_matrix.png`

## Notes

- The checkpoint stores the TF-IDF vectorizer. If it is missing, the test script refits TF-IDF on the training split to match feature dimensions.
- Checkpoints in `Results/checkpoints/` are generated locally and are not committed to GitHub.
- You may see an ffmpeg notice from the audio backend; it is harmless.
- The `archive/` folder contains earlier fusion baselines and random-split scripts.
