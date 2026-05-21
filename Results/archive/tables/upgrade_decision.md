# Upgrade Decision (Archive)

This note records a historical decision point after the initial speaker-holdout baseline dropped to ~48-51% accuracy. The goal was to improve **speaker robustness** without overhauling the entire pipeline.

## Options Considered

1. Better handcrafted features
2. Data augmentation
3. CNN on spectrograms
4. Pretrained speech embeddings (HuBERT, wav2vec 2.0, WavLM)
5. More diverse data (additional speakers or datasets)

## Decision

**Chosen path:** better handcrafted features + data augmentation.

**Why this choice:**

- Fits the existing MFCC-based codebase with minimal refactoring
- Directly targets speaker-style variance through augmentation
- Lower risk than jumping immediately to large pretrained models

## What Was Implemented

**Enhanced feature set** (in addition to MFCCs):

- Delta and delta-delta MFCCs
- RMS energy and zero-crossing rate
- Spectral centroid, bandwidth, rolloff, flatness
- Chroma and spectral contrast
- Optional pitch features

**Augmentations applied:**

- Noise injection
- Volume scaling
- Pitch shifting
- Time stretching

**Classifier:** Linear SVM

## Results

Random-split accuracy remained extremely high (easy setting):

| Model | Accuracy |
| --- | ---: |
| Enhanced + Augmentation + Linear SVM | 99.82% |

Speaker-holdout accuracy improved:

| Train Speaker | Test Speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 57.71% |
| YAF | OAF | 58.64% |

Average speaker-holdout accuracy:

```text
58.18%
```

This is a meaningful gain over the original 48-51% holdout performance, but still far below random-split accuracy.

## Interpretation

Feature enrichment and augmentation reduce speaker dependence but do not eliminate it. The core limitation is the dataset itself (two speakers), which makes speaker-independent emotion recognition inherently hard at this stage.

## Recommended Next Step

If further improvement is required, the next logical upgrade is to use **pretrained speech embeddings** trained on large corpora:

- wav2vec 2.0
- HuBERT
- WavLM

These models tend to produce more speaker-robust representations than handcrafted features and were the basis for later stages in the project.

