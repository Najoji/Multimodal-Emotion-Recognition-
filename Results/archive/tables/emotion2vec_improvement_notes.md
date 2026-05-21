# Emotion2Vec+ Improvement Notes (Archive)

This note documents the jump from the emotion-finetuned Wav2Vec2 stage (~mid-80s) to an **emotion-specialized speech model** that is explicitly trained for affective cues.

## Why It Was Tested

After reaching the mid-80% range with an emotion-finetuned Wav2Vec2 model, the next logical step was to try a model **purpose-built for speech emotion recognition** instead of continuing with augmentation on generic speech embeddings.

## Setup

- **Model:** `iic/emotion2vec_plus_base`
- **Evaluation:** strict speaker-holdout (OAF -> YAF, then YAF -> OAF)
- **Classifier:** `StandardScaler` -> L2 normalization -> `LinearSVC(C=0.1, balanced)`

## Results

| Train Speaker | Test Speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 99.79% |
| YAF | OAF | 99.93% |

Average speaker-holdout accuracy:

```text
99.86%
```

## Sanity Check (Label Shuffle)

To verify that the score is not due to leakage or a broken split, the same pipeline was trained with **shuffled training labels**. Performance dropped to near chance:

| Train Speaker | Test Speaker | Accuracy (Shuffled Labels) |
| --- | --- | ---: |
| OAF | YAF | 15.36% |
| YAF | OAF | 9.14% |

For a 7-class task, chance is ~14.29%, so these values confirm that the real gains come from meaningful emotion structure in the embeddings.

## Decision

Emotion2Vec+ clearly dominates earlier approaches on TESS speaker-holdout:

- Handcrafted features
- Generic Wav2Vec2
- Emotion-finetuned Wav2Vec2

Because the result is already near-perfect, augmentation is no longer the first priority for improving headline accuracy. Any further augmentation is best treated as a **robustness study**, not the main optimization path.
