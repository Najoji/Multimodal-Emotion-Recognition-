# Setup Instructions

This file is the main guide for understanding, setting up, and reviewing the project.

## 1. What This Project Does

This project studies emotion recognition on the Toronto Emotional Speech Set (TESS) using three systems:

1. speech only
2. text only
3. speech + text fusion

The project begins with simple baselines, then improves the speech model through several stages:

| Stage | Main model | Historical random-split accuracy | Average speaker-holdout accuracy |
| --- | --- | ---: | ---: |
| 1 | MFCC baseline | 99.82% | 49.50% |
| 2 | Generic Wav2Vec2 + adaptation | Not used as the main metric | 66.71% |
| 3 | Emotion-finetuned Wav2Vec2 | Not used as the main metric | 84.89% |
| 4 | Emotion2Vec+ champion | Not used as the main metric | 99.86% |

The main lesson from the project is that stronger emotion-aware representations helped far more than adding complicated classifiers.

The original random-split baselines were:

| Early baseline | Random-split accuracy |
| --- | ---: |
| Speech-only MFCC | 99.82% |
| Text-only TF-IDF | 0.00% |
| Fusion MFCC + TF-IDF | 99.82% |

These values are kept to show why the project moved to speaker holdout. They are historical context, not the final evaluation standard.

## 2. Recommended Reading Order

If you are reviewing the finished project, read the files in this order:

1. `README.md`
2. `FINAL_PROJECT_REPORT.md`
3. `Results/tables/README.md`
4. `Results/plots/README.md`
5. `PROJECT_LOG.md` only if you want the full development history

## 3. Project Layout

```text
models/
  speech_pipeline/
  text_pipeline/
  fusion_pipeline/
src/
  speech_emotion/
Results/
  tables/
  plots/
  archive/
  checkpoints/
  embedding_cache/
FINAL_PROJECT_REPORT.md
PROJECT_LOG.md
README.md
requirements.txt
SETUP_INSTRUCTIONS.md
```

Important folders:

- `models/`: training, testing, speaker-holdout, and experiment scripts
- `src/speech_emotion/`: shared dataset, feature, and evaluation helpers
- `Results/tables/`: final reviewer-facing CSV outputs
- `Results/plots/`: final reviewer-facing figures
- `Results/archive/`: older exploratory outputs kept for traceability

## 4. Dataset Setup

Download the TESS dataset from Kaggle and place the extracted files inside:

```text
data/
```

The loader searches recursively for `.wav` files, so the exact nested folder layout is not critical.

Example TESS filename:

```text
OAF_back_angry.wav
```

From that filename, the code reads:

- speaker: `OAF`
- transcript text: `back`
- emotion label: `angry`

The dataset loader also removes duplicate filenames if the archive was accidentally extracted twice.

## 5. Python Environment Setup

Use Python `3.10` or `3.11`.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 6. Core Reproduction Commands

All commands below use the **speaker-holdout evaluation** that the final report is based on. The first commands shown are the final models a reviewer is most likely to care about.

### Final speech champion result

```powershell
python models\speech_pipeline\emotion2vec_holdout.py
```

This evaluates the final Emotion2Vec+ speech model under speaker holdout:

```text
99.86% average speaker-holdout accuracy
```

For comparison, the first simple speech baseline had looked like `99.82%` under the old random split, but only `49.50%` under speaker holdout.

Because cached embeddings are already stored in `Results/embedding_cache/`, the script can reuse them instead of rebuilding them from scratch.

### Final text-only speaker-holdout result

```powershell
python models\text_pipeline\speaker_holdout.py
```

This trains on one speaker and tests on the other, then repeats in the opposite direction. It produces the final text-only result:

```text
14.29% average speaker-holdout accuracy
```

The earlier random-split TF-IDF text baseline was `0.00%`.

### Final fusion speaker-holdout result

```powershell
python models\fusion_pipeline\speaker_holdout.py
```

This evaluates the final speech + text fusion model under the same strict speaker-holdout setup.

For historical context, the original random-split MFCC + TF-IDF fusion baseline was `99.82%`.

### Final visualizations

```powershell
python models\visualize_representations.py
```

This regenerates the final report figures, including the speech-model evolution plot.

## 7. Reproduce The Speech Development Stages

The final speech model did not appear all at once. These commands reproduce the major progression shown in the report.

### Stage 1: MFCC baseline

```powershell
python models\speech_pipeline\mfcc_holdout_reports.py
```

Expected result:

```text
99.82% on the old random split, but 49.50% average speaker-holdout accuracy
```

Purpose:

- proves the first simple speech representation did not generalize well to an unseen speaker
- gives the baseline that later stages improved upon

### Stage 2A: Generic Wav2Vec2 baseline

```powershell
python models\speech_pipeline\pretrained_embedding_holdout.py
```

Expected result:

```text
62.89% average speaker-holdout accuracy
```

Purpose:

- shows that generic pretrained speech embeddings improve over handcrafted MFCCs

### Stage 2B: Generic Wav2Vec2 with adaptation

```powershell
python models\speech_pipeline\wav2vec2_domain_adaptation.py
```

Expected best result:

```text
66.71% average speaker-holdout accuracy
```

Purpose:

- shows that speaker-wise normalization can reduce some of the cross-speaker shift

### Stage 3: Emotion-finetuned Wav2Vec2

```powershell
python models\speech_pipeline\emotion_ft_best_holdout.py
```

Expected best result:

```text
84.89% average speaker-holdout accuracy
```

Purpose:

- shows that emotion-specific pretrained representations are much stronger than generic speech representations

### Stage 4: Emotion2Vec+ champion

```powershell
python models\speech_pipeline\emotion2vec_holdout.py
```

Expected result:

```text
99.86% average speaker-holdout accuracy
```

Purpose:

- reproduces the final best speech-only model used in the report

### Legacy random-split scripts

The older `train.py` and `test.py` files are still kept in each pipeline because they document the early baseline stage and satisfy the original deliverable structure from the project brief. They are **not** the main commands to use when reproducing the final report.

## 8. Main Results

Final speaker-holdout comparison:

| Model | Average accuracy |
| --- | ---: |
| Speech-only champion | 99.86% |
| Text-only | 14.29% |
| Fusion | 99.86% |

Historical random-split baselines:

| Early baseline | Random-split accuracy |
| --- | ---: |
| Speech-only MFCC | 99.82% |
| Text-only TF-IDF | 0.00% |
| Fusion MFCC + TF-IDF | 99.82% |

Important files:

- final report: `FINAL_PROJECT_REPORT.md`
- final tables guide: `Results/tables/README.md`
- final plots guide: `Results/plots/README.md`
- development history: `PROJECT_LOG.md`

## 9. Notes For Reviewers

- The primary evaluation uses **speaker holdout**, not only random splitting.
- TESS text consists of isolated words, so the text-only pipeline is intentionally weak.
- The final `99.86%` result is excellent on the controlled TESS speaker-holdout setting, but it should not be treated as guaranteed real-world performance on unrelated audio.
- The main folders were kept compact for review. Older exploratory outputs are preserved separately in `Results/archive/`.

