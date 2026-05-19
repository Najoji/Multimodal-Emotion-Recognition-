# Multimodal Emotion Recognition

> This report is best viewed in Markdown Preview . In VS Code, use `Ctrl+Shift+V`.

This project studies emotion recognition on the Toronto Emotional Speech Set (TESS) using three modalities:

1. **speech only**
2. **text only**
3. **speech + text fusion**

The project begins with simple baselines, then improves the speech model through several stages, culminating in a staged improvement in speaker-holdout speech accuracy:

| Stage | Model | Speaker-holdout accuracy |
| --- | --- | ---: |
| 1 | MFCC baseline | 49.50% |
| 2 | Generic Wav2Vec2 + adaptation | 66.71% |
| 3 | Emotion-finetuned Wav2Vec2 | 84.89% |
| 4 | Emotion2Vec+ champion | 99.86% |

For context, the original easy random-split baselines were:

| Early baseline | Random-split accuracy |
| --- | ---: |
| Speech-only MFCC | 99.82% |
| Text-only TF-IDF | 0.00% |
| Fusion MFCC + TF-IDF | 99.82% |

Those random-split values are kept as historical context; the final project comparison uses speaker holdout because it is a more honest cross-speaker test.

## Recommended Reading Order

If you are reviewing this project, read in this order:

1. This `README.md` (you are here)
2. `FINAL_PROJECT_REPORT.md` - final report with results and analysis
3. `Results/tables/README.md` - guide to result tables
4. `Results/plots/README.md` - guide to result plots
5. `PROJECT_LOG.md` - full development history (optional, for detailed context)

---

## Repository Layout

```text
project/
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py              (final Emotion2Vec+ model with speaker-holdout)
│   │   ├── test.py               (evaluation script)
│   │   ├── archive/              (all experimental and baseline scripts)
│   │   └── __pycache__/
│   ├── text_pipeline/
│   │   ├── train.py              (final TF-IDF model with speaker-holdout)
│   │   ├── test.py               (evaluation script)
│   │   ├── archive/              (old random-split scripts)
│   │   └── __pycache__/
│   ├── fusion_pipeline/
│   │   ├── train.py              (final fusion model with speaker-holdout)
│   │   ├── test.py               (evaluation script)
│   │   ├── archive/              (old random-split scripts)
│   │   └── __pycache__/
│   └── visualize_representations.py
├── src/
│   └── speech_emotion/
│       ├── dataset.py
│       ├── audio_features.py
│       ├── evaluation.py
│       └── ...
├── Results/
│   ├── tables/
│   │   ├── All 3 model variants accuracy tables
│   │   └── README.md
│   ├── plots/
│   │   ├── Visualization plots
│   │   └── README.md
│   ├── checkpoints/
│   ├── embedding_cache/
│   └── archive/
├── data/
│   └── [TESS dataset files]
├── FINAL_PROJECT_REPORT.md
├── PROJECT_LOG.md
├── README.md (this file)
└── requirements.txt
```

---

## 1. Dataset Setup

### Download and Extract

Download the TESS dataset from [Kaggle](https://www.kaggle.com/datasets/ejlok/toronto-emotional-speech-set-tess) and place all extracted files inside:

```text
data/
```

The loader searches recursively for `.wav` files, so the exact nested folder layout is not critical.

### Dataset Format

Example TESS filename:

```text
OAF_back_angry.wav
```

From this filename, the code extracts:

- **speaker**: `OAF` (Older Adult Female)
- **transcript**: `back` (spoken word)
- **emotion**: `angry` (emotion label)

TESS contains **7 emotions**: angry, disgust, fear, happy, neutral, pleasant surprise, sad

TESS contains **2 speakers**: OAF (Older Adult Female, 400 samples) and YAF (Young Adult Female, 400 samples)

**Total: 2,800 unique samples** (7 emotions × 2 speakers × 200 samples per emotion-speaker pair)

### Dataset Note

The dataset loader removes duplicate filenames if the archive was accidentally extracted twice, ensuring exactly 2,800 unique samples.

---

## 2. Python Environment Setup

### Prerequisites

- Python 3.10 or 3.11
- pip

### Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```

Core dependencies include:
- `librosa` - audio loading and processing
- `scikit-learn` - ML models and feature extraction
- `transformers` - pretrained speech models (Wav2Vec2, Emotion2Vec+)
- `pandas` / `numpy` - data handling
- `matplotlib` / `seaborn` - visualization

---

## 3. Final Results

### Speaker-Holdout Evaluation (Official)

| Model | Average accuracy | Notes |
| --- | ---: | --- |
| Speech-only (Emotion2Vec+) | 99.86% | Final champion |
| Text-only (TF-IDF) | 14.29% | Chance level (7 classes) |
| Fusion (Speech + Text) | 99.86% | Text adds no value |

### Historical Random-Split Baselines

| Early baseline | Random-split accuracy | Notes |
| --- | ---: | --- |
| Speech-only MFCC | 99.82% | Misleading on TESS |
| Text-only TF-IDF | 0.00% | Isolated words carry no emotion |
| Fusion MFCC + TF-IDF | 99.82% | Speaker holdout revealed overfitting |

---

## 4. Official Train/Test Scripts (PDF Deliverable)

The `train.py` and `test.py` files in each pipeline folder satisfy the PDF deliverable requirements. These scripts use the **final best models** with **speaker-holdout evaluation**.

### Train All Models

**Speech-only (Emotion2Vec+):**
```powershell
python models\speech_pipeline\train.py
```
- Trains on OAF speaker, tests on YAF speaker (and vice versa)  
- Uses Emotion2Vec+ (state-of-the-art emotion model)
- **Expected:** 99.86% average speaker-holdout accuracy
- Saves checkpoints and classification reports to `Results/`

**Text-only (TF-IDF):**
```powershell
python models\text_pipeline\train.py
```
- Trains on OAF speaker, tests on YAF speaker (and vice versa)
- Demonstrates that isolated words carry no emotion
- **Expected:** 14.29% average speaker-holdout accuracy (chance level)

**Fusion (Speech + Text):**
```powershell
python models\fusion_pipeline\train.py
```
- Combines Emotion2Vec+ embeddings + TF-IDF features  
- Shows that weak modalities don't help strong ones
- **Expected:** 99.86% average speaker-holdout accuracy (identical to speech-only)

### Evaluate All Models

**Speech-only:**
```powershell
python models\speech_pipeline\test.py
```

**Text-only:**
```powershell
python models\text_pipeline\test.py
```

**Fusion:**
```powershell
python models\fusion_pipeline\test.py
```

All test scripts load trained checkpoints from `Results/checkpoints/` and display accuracy summaries.

---

## 5. Explore Development Stages (Optional)

The final speech model evolved through four stages. If you want to explore how the model improved at each stage, the scripts are available in `models/speech_pipeline/archive/`:

**Stage 1: MFCC Baseline**
```powershell
python models\speech_pipeline\archive\mfcc_holdout_reports.py
```
49.50% speaker-holdout accuracy — shows why handcrafted features don't generalize.

**Stage 2A: Generic Wav2Vec2**
```powershell
python models\speech_pipeline\archive\pretrained_embedding_holdout.py
```
62.89% speaker-holdout accuracy — generic speech embeddings improve over MFCCs.

**Stage 2B: Wav2Vec2 with Adaptation**
```powershell
python models\speech_pipeline\archive\wav2vec2_domain_adaptation.py
```
66.71% speaker-holdout accuracy — speaker normalization reduces cross-speaker shift.

**Stage 3: Emotion-Finetuned Wav2Vec2**
```powershell
python models\speech_pipeline\archive\emotion_ft_best_holdout.py
```
84.89% speaker-holdout accuracy — emotion specialization helps much more than generic features.

**Stage 4: Emotion2Vec+ Champion** (also in archive)
```powershell
python models\speech_pipeline\archive\emotion2vec_holdout.py
```
99.86% speaker-holdout accuracy — state-of-the-art emotion speech model.

These scripts are provided for **educational exploration** only. For the official deliverable, use the `train.py` and `test.py` scripts in the main folder.

---

## 6. Final Outputs

Reviewer-facing results are stored in:

- **`Results/tables/`** - CSV accuracy tables and classification reports for all models
- **`Results/plots/`** - PNG visualization figures (representation plots, confusion matrices, etc.)

Older exploratory outputs are preserved in:

- **`Results/archive/`** - Early experimental results and discarded approaches

---

## 7. Key Insights

1. **Emotion is acoustic, not semantic**: Text-only achieves 14.29% (chance level), proving isolated words carry no emotion information.

2. **Representation quality >> Classifier complexity**: Stronger embeddings (Emotion2Vec+) gave 2x improvement over weak embeddings + complex classifiers.

3. **Speaker-holdout is essential**: Random-split gave misleading 99.82% early on, but speaker-holdout revealed true cross-speaker generalization (49.50% initially → 99.86% finally).

4. **Fusion with weak modalities doesn't help**: Text adds zero value to the fusion (99.86% identical to speech-only), confirming that weak features act as noise, not signal.

---

## 8. Notes For Reviewers

- The primary evaluation uses **speaker holdout** (OAF train / YAF test and vice versa), not random splitting.
- TESS text consists of isolated words, so intentionally weak text-only performance validates the acoustic nature of emotion.
- The 99.86% result is excellent on controlled TESS speaker-holdout, but should not be treated as guaranteed real-world performance on unrelated datasets.
- All code, data, and results are self-contained for reproducibility.

---

## Questions? 

Refer to:
- `FINAL_PROJECT_REPORT.md` for detailed analysis
- `PROJECT_LOG.md` for the full development journey
- `Results/tables/README.md` for result tables
- `Results/plots/README.md` for plot descriptions
