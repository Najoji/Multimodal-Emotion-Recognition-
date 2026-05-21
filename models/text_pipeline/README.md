# Text Pipeline (TF-IDF Baseline)

This pipeline is the text-only baseline. It uses TF-IDF features from transcripts and a balanced Logistic Regression classifier under speaker-holdout evaluation.

## How it works

1. Load TESS transcripts (single words derived from filenames).
2. Compute TF-IDF features with unigrams and bigrams.
3. Train a balanced Logistic Regression model.
4. Evaluate on the held-out speaker (OAF -> YAF, then YAF -> OAF).

## Run the model

Train:

```powershell
python models\text_pipeline\train.py
```

Test (reloads checkpoints and recomputes accuracy on the holdout split, saving test reports and confusion matrices):

```powershell
python models\text_pipeline\test.py
```

## Outputs

- Checkpoints: `Results/checkpoints/text_only_*_train.joblib`
- Train reports: `Results/tables/text_only_*_train_report.csv`
- Test reports: `Results/tables/text_only_*_test_report.csv`
- Test confusion matrices: `Results/plots/text_only_*_test_confusion_matrix.png`

## Notes

- TESS transcripts are isolated words, so text-only accuracy is expected to be near chance.
- Checkpoints in `Results/checkpoints/` are generated locally and are not committed to GitHub.
- The `archive/` folder contains older random-split scripts and experiments.
