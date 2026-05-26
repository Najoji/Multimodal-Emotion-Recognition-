# Multimodal Emotion Recognition

> This report is best viewed in Markdown Preview . In VS Code, use `Ctrl+Shift+V`.

This project studies emotion recognition on the Toronto Emotional Speech Set (TESS) using three modalities:

1. **speech only**
2. **text only**
3. **speech + text fusion**

## Final Pipeline Results (Speaker-Holdout Evaluation)

| Pipeline | Representation | Accuracy |
| --- | --- | ---: |
| Speech-only | Emotion2Vec+ embeddings | **99.86%** |
| Text-only | TF-IDF features | **14.29%** |
| Fusion | Speech + Text combined | **99.86%** |

**Key Finding:** Text carries virtually no emotion information on TESS (14.29% = chance level). High-quality speech representations completely dominate; fusion adds zero value.

---

## Project Development: Four-Stage Speech Model Evolution

The project evolved the speech model through four stages to reach the final 99.86% accuracy:

| Stage | Model | Speaker-holdout accuracy |
| --- | --- | ---: |
| 1 | MFCC baseline | 49.50% |
| 2 | Generic Wav2Vec2 + adaptation | 66.71% |
| 3 | Emotion-finetuned Wav2Vec2 | 84.89% |
| 4 | Emotion2Vec+ champion | 99.86% |

**Lesson:** Better representations matter far more than classifier complexity.

---

## Evaluation Methodology: Speaker-Holdout

All models use **speaker-holdout cross-validation** to ensure robust generalization. We train on one speaker (OAF: older adult female) and test on the other (YAF: younger adult female), then reverse. This prevents overfitting to speaker identity and provides a realistic measure of how well the model generalizes to unseen speakers. The reported accuracies (99.86%, 14.29%, 99.86%) represent the average across both speaker pairs.

---

## Recommended Reading Order

If you are reviewing this project, read in this order:

1. This `README.md` (you are here)
2. `FINAL_PROJECT_REPORT.md` - final report with results and analysis
3. `Results/tables/README.md` - guide to result tables
4. `Results/plots/README.md` - guide to result plots
5. `PROJECT_LOG.md` - full development history (optional, for detailed context)

---

## How to Navigate the Project

Use this section as a quick map when reviewing or presenting the repository.

| Goal | Where to look | What you will find |
| --- | --- | --- |
| Understand the complete project story | `FINAL_PROJECT_REPORT.md` | Architecture decisions, experiments, analysis, error cases, and figures |
| Run the final official pipelines | `models/speech_pipeline/`, `models/text_pipeline/`, `models/fusion_pipeline/` | Each folder has a `train.py` and `test.py` for the required deliverables |
| Inspect reusable project code | `src/speech_emotion/` | Dataset parsing, audio feature loading, and evaluation helpers |
| Check final numeric results | `Results/tables/` | Accuracy summaries and classification reports for all three model variants |
| Check final visual results | `Results/plots/` | Confusion matrices and representation visualizations |
| Understand old experiments | `Results/archive/` and `models/*/archive/` | Historical baselines, Wav2Vec2 experiments, MFCC baselines, and discarded trials |
| Follow the development process | `PROJECT_LOG.md` | Chronological notes on what changed, what worked, and what failed |




The `archive/` folders are not required for the final pipeline, but they are kept to show the model-development history behind the final result.

---

## Repository Layout

```text
project/
+-- models/
|   +-- speech_pipeline/
|   |   +-- train.py              (final Emotion2Vec+ model with speaker-holdout)
|   |   +-- test.py               (evaluation script)
|   |   +-- archive/              (all experimental and baseline scripts)
|   +-- text_pipeline/
|   |   +-- train.py              (final TF-IDF model with speaker-holdout)
|   |   +-- test.py               (evaluation script)
|   |   +-- archive/              (old random-split scripts)
|   +-- fusion_pipeline/
|   |   +-- train.py              (final fusion model with speaker-holdout)
|   |   +-- test.py               (evaluation script)
|   |   +-- archive/              (old random-split scripts)
|   +-- visualize_representations.py
+-- src/
|   +-- speech_emotion/
|       +-- dataset.py
|       +-- audio_features.py
|       +-- evaluation.py
+-- Results/
|   +-- tables/                   (accuracy tables and classification reports)
|   +-- plots/                    (confusion matrices and representation plots)
|   +-- checkpoints/              (generated locally after training)
|   +-- archive/                  (older exploratory outputs)
+-- data/
|   +-- [TESS dataset files]
+-- FINAL_PROJECT_REPORT.md
+-- PROJECT_LOG.md
+-- README.md
+-- requirements.txt
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

**Total: 2,800 unique samples** (7 emotions x 2 speakers x 200 samples per emotion-speaker pair)

### Dataset Note

The dataset loader removes duplicate filenames if the archive was accidentally extracted twice, ensuring exactly 2,800 unique samples.

---

## 2. Python Environment Setup

### Prerequisites

- Python 3.10 or 3.11
- pip
- Internet access for the first run, because `pip`, `transformers`, `funasr`, and `modelscope` may download packages/model weights

Check your Python version from the project root:

```powershell
python --version
```

If `python` is not available on Windows, try:

```powershell
py --version
```

### Create Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Windows blocks activation scripts, run this once in the same PowerShell window and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

Use `python -m pip` so the packages are installed into the active virtual environment:

Windows PowerShell:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Quick dependency check:

```powershell
python -m pip check
```

Core dependencies include:
- `librosa` - audio loading and processing
- `scikit-learn` - ML models and feature extraction
- `transformers` - pretrained speech models (Wav2Vec2, Emotion2Vec+)
- `pandas` / `numpy` - data handling
- `matplotlib` / `seaborn` - visualization

### Verify Dataset Detection

After placing the TESS `.wav` files under `data/`, run:

```powershell
python -c "from src.speech_emotion.dataset import load_tess_dataframe; df = load_tess_dataframe('data'); print(len(df)); print(df[['emotion','transcript']].head())"
```

Expected count: `2800` unique clips. If the count is larger, the loader will deduplicate repeated filenames during training.

### Notes for Other Computers

- Run commands from the repository root, the folder that contains `README.md`.
- Keep the dataset under `data/`; nested folders are fine because the loader searches recursively.
- The first speech/fusion run can take longer because pretrained model weights are downloaded and cached.
- If you see an `ffmpeg is not installed` message, the scripts still work through the torchaudio fallback used in this project.

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

## 4. Official Train/Test Scripts 

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
- **Note:** You may see a "ffmpeg is not installed" notice at startup. This is harmless; the script works fine using torchaudio as a fallback.

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
- **Note:** You may see a "ffmpeg is not installed" notice at startup. This is harmless; the script works fine using torchaudio as a fallback.

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

All test scripts load trained checkpoints from `Results/checkpoints/` and display accuracy summaries. Checkpoints are generated locally and are not committed to GitHub.

---

## 5. Explore Development Stages (Optional)

The final speech model evolved through four stages. If you want to explore how the model improved at each stage, the scripts are available in `models/speech_pipeline/archive/`:

**Stage 1: MFCC Baseline**
```powershell
python models\speech_pipeline\archive\mfcc_holdout_reports.py
```
49.50% speaker-holdout accuracy - shows why handcrafted features don't generalize.

**Stage 2A: Generic Wav2Vec2**
```powershell
python models\speech_pipeline\archive\pretrained_embedding_holdout.py
```
62.89% speaker-holdout accuracy - generic speech embeddings improve over MFCCs.

**Stage 2B: Wav2Vec2 with Adaptation**
```powershell
python models\speech_pipeline\archive\wav2vec2_domain_adaptation.py
```
66.71% speaker-holdout accuracy - speaker normalization reduces cross-speaker shift.

**Stage 3: Emotion-Finetuned Wav2Vec2**
```powershell
python models\speech_pipeline\archive\emotion_ft_best_holdout.py
```
84.89% speaker-holdout accuracy - emotion specialization helps much more than generic features.

**Stage 4: Emotion2Vec+ Champion** (also in archive)
```powershell
python models\speech_pipeline\archive\emotion2vec_holdout.py
```
99.86% speaker-holdout accuracy - state-of-the-art emotion speech model.

These scripts are provided for **educational exploration** only. For the official deliverable, use the `train.py` and `test.py` scripts in the main folder.

---

## 5A. Text and Fusion Pipeline Details

### Text-only pipeline (models/text_pipeline/)

The text-only pipeline treats each clip's transcript (a single word derived from the filename) as the input. It lowercases the text and converts it into TF-IDF features with unigrams and bigrams, then trains a balanced Logistic Regression classifier under the same speaker-holdout setting (OAF -> YAF, then YAF -> OAF). Because TESS words carry minimal emotion content, this pipeline stays near chance at about 14.29% accuracy. Evaluation artifacts are written to the `Results/tables/` and `Results/plots/` folders.

### Fusion pipeline (models/fusion_pipeline/)

The fusion pipeline combines Emotion2Vec+ speech embeddings with TF-IDF text features by concatenating both representations and training a balanced Linear SVM. Speech embeddings are standardized and L2-normalized, while text follows the same TF-IDF setup as the text-only model. It uses speaker-holdout evaluation and reaches the same 99.86% accuracy as speech-only, showing that text adds no measurable signal on TESS. Fusion outputs are also stored under `Results/` for tables and plots.

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

3. **Speaker-holdout is essential**: Random-split gave misleading 99.82% early on, but speaker-holdout revealed true cross-speaker generalization (49.50% initially -> 99.86% finally).

4. **Fusion with weak modalities doesn't help**: Text adds zero value to the fusion (99.86% identical to speech-only), confirming that weak features act as noise, not signal.

---

## 8. Notes For Reviewers

- The primary evaluation uses **speaker holdout** (OAF train / YAF test and vice versa), not random splitting.
- TESS text consists of isolated words, so intentionally weak text-only performance validates the acoustic nature of emotion.
- The 99.86% result is excellent on controlled TESS speaker-holdout, but should not be treated as guaranteed real-world performance on unrelated datasets.
- All code, data, and results are self-contained for reproducibility.

---

## 9. Google Drive ZIP Evaluation Guide

This section is for evaluators who download the submitted Google Drive ZIP file:

```text
IIITH speech analysis.zip
```

### Step 1: Download and Extract

Download `IIITH speech analysis.zip` from Google Drive, then extract it. The ZIP contains one top-level folder named `IIITH speech analysis`.

Windows PowerShell:

```powershell
Expand-Archive -Path "IIITH speech analysis.zip" -DestinationPath "."
cd "IIITH speech analysis"
```

macOS / Linux:

```bash
unzip "IIITH speech analysis.zip"
cd "IIITH speech analysis"
```

If using the file explorer instead of the terminal, right-click the ZIP, choose extract/unzip, then open the extracted `IIITH speech analysis` folder.

Confirm that the extracted folder contains the expected project files:

```powershell
Get-ChildItem README.md
Get-ChildItem models
Get-ChildItem Results
```

On macOS / Linux:

```bash
ls README.md
ls models
ls Results
```

### Step 2: Create the Environment

Create a fresh virtual environment inside the extracted project folder. Do not rely on a copied `.venv` from another computer.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### Step 3: Check the Dataset

If the ZIP includes `data/`, no additional dataset step is needed. If it does not include `data/`, download TESS from Kaggle and place the extracted `.wav` files under:

```text
data/
```

Verify dataset loading:

```powershell
python -c "from src.speech_emotion.dataset import load_tess_dataframe; df = load_tess_dataframe('data'); print(len(df)); print(df[['emotion','transcript']].head())"
```

Expected count: `2800` unique clips.

### Step 4: Test Existing Checkpoints

If `Results/checkpoints/` contains `.joblib` checkpoint files, run:

```powershell
python models\speech_pipeline\test.py
python models\text_pipeline\test.py
python models\fusion_pipeline\test.py
```

These scripts reload saved checkpoints and write refreshed reports/plots under `Results/`.

### Step 5: Retrain If Needed

If checkpoints are missing, or if full reproduction from training is required, run:

```powershell
python models\speech_pipeline\train.py
python models\text_pipeline\train.py
python models\fusion_pipeline\train.py
```

Then run the test commands from Step 4.

### Step 6: Optional Plot Regeneration

To regenerate representation plots after checkpoints/embedding caches are available:

```powershell
python models\visualize_representations.py
```

Final tables are in `Results/tables/`, and final figures are in `Results/plots/`.

---

## Questions? 

Refer to:
- `FINAL_PROJECT_REPORT.md` for detailed analysis
- `PROJECT_LOG.md` for the full development journey
- `Results/tables/README.md` for result tables
- `Results/plots/README.md` for plot descriptions
