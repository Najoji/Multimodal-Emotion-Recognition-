# Wav2Vec2 Improvement Notes (Archive)

This note tracks incremental improvements starting from a **wav2vec2-base + Linear SVM** baseline on speaker-holdout evaluation.

## Baseline (No Adaptation)

This is the strictest setting: the classifier is trained on one speaker and evaluated on the other **without using any target-speaker statistics**.

| Method | Average Accuracy |
| --- | ---: |
| Wav2Vec2-base + Linear SVM | 62.90% |
| Wav2Vec2-base + L2 normalization + Linear SVM | 63.75% |

L2 normalization provides a small but consistent improvement while keeping the evaluation strictly zero-shot.

## Unsupervised Speaker Adaptation

This setting uses **unlabeled target-speaker statistics** (mean/variance) to reduce distribution shift, but does **not** use any target emotion labels.

| Method | OAF -> YAF | YAF -> OAF | Average |
| --- | ---: | ---: | ---: |
| Speaker-wise z-score + L2 normalization | 70.57% | 62.86% | 66.71% |
| Speaker-wise centering + L2 normalization | 70.71% | 62.64% | 66.68% |

## Interpretation

The gap between 63.75% and ~66.7% indicates that a meaningful portion of the error is due to **speaker distribution shift** rather than model capacity. Normalizing per-speaker embeddings mitigates speaker-specific offsets before classification.

## Reporting 

- Report **63.75%** as the strict zero-shot speaker-holdout result.
- Report **66.71%** only with a clear note that it uses **unlabeled target-speaker adaptation**.

Both results are valid, but they represent different evaluation assumptions.

