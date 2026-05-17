# Multimodal Emotion Recognition

For a reviewer-friendly entry point, start with:

```text
REVIEWER_GUIDE.md
```

This project predicts emotion from the Toronto Emotional Speech Set (TESS) using:

1. Speech-only input
2. Text-only input
3. A fused speech + text model

The first goal is to build clean, explainable baselines. After that, we can upgrade the models.

## Dataset

Download TESS from Kaggle:

https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

Put the extracted dataset inside:

```text
data/
```

The code searches recursively for `.wav` files, so the exact folder nesting can vary.
If the dataset is accidentally extracted twice, duplicate filenames are ignored.

## Project Layout

```text
models/
  speech_pipeline/
    train.py
    test.py
  text_pipeline/
    train.py
    test.py
  fusion_pipeline/
    train.py
    test.py
src/
  speech_emotion/
    dataset.py
    audio_features.py
    evaluation.py
Results/
  checkpoints/
  tables/
  plots/
requirements.txt
START_HERE.md
```

## Setup

Install Python 3.10 or 3.11 first. Then:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Baselines

Speech-only:

```powershell
python models\speech_pipeline\train.py --data-dir data
python models\speech_pipeline\test.py
```

Enhanced speech upgrade:

```powershell
python models\speech_pipeline\train.py --data-dir data --feature-set enhanced --augment --classifier linear_svm --output-name speech_enhanced_aug_svm
python models\speech_pipeline\test.py --model-path Results\checkpoints\speech_enhanced_aug_svm.joblib --test-split Results\tables\speech_enhanced_aug_svm_test_split.csv --output-name speech_enhanced_aug_svm_test
python models\speech_pipeline\speaker_holdout_augmented.py --data-dir data
```

Text-only:

```powershell
python models\text_pipeline\train.py --data-dir data
python models\text_pipeline\test.py
```

Fusion:

```powershell
python models\fusion_pipeline\train.py --data-dir data
python models\fusion_pipeline\test.py
```

Generate report plots:

```powershell
python models\visualize_representations.py --data-dir data
```

## Current Baseline Choices

- Speech features: MFCC summary statistics using `librosa`
- Text features: TF-IDF over words extracted from TESS filenames
- Classifier: Logistic Regression
- Fusion: concatenate speech features and text features

These are intentionally simple. They make a strong first milestone because they produce accuracy tables, confusion matrices, and failure cases quickly.

## Report Artifacts

The scripts save outputs under:

- `Results/tables/`
- `Results/plots/`

Important plots include confusion matrices, the model comparison bar chart, and 2D representation plots for speech, text, and fusion.

For a plain-English explanation of the final tables and figures, see:

- `Results/tables/README.md`
- `Results/plots/README.md`
- `Results/FINAL_PROJECT_REPORT.md`
