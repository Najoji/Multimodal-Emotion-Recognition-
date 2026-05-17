# Wav2Vec2 Improvement Notes

After `wav2vec2-base + Linear SVM` reached **62.90%** average speaker-holdout accuracy, additional experiments were run on the cached Wav2Vec2 embeddings.

## Strict Zero-Shot Improvement

This setting does **not** use any unlabeled statistics from the target speaker.

| Method | Average Accuracy |
| --- | ---: |
| Original Wav2Vec2 + Linear SVM | 62.90% |
| Wav2Vec2 + L2 normalization + Linear SVM | 63.75% |

This is a small but honest gain for the strictest setting.

## Unsupervised Speaker Adaptation Improvement

This setting uses the unlabeled batch distribution of the target speaker, but **does not use target emotion labels**.

| Method | OAF -> YAF | YAF -> OAF | Average |
| --- | ---: | ---: | ---: |
| Speaker-wise z-score + L2 normalization | 70.57% | 62.86% | 66.71% |
| Speaker-wise centering + L2 normalization | 70.71% | 62.64% | 66.68% |

## Interpretation

The gain suggests that part of the remaining error comes from a distribution shift between the two speakers. Normalizing each speaker's embedding distribution reduces speaker-specific offsets before emotion classification.

## Important Reporting Caveat

- Use **63.75%** if reporting the strictest zero-shot unseen-speaker result.
- Use **66.71%** only if you clearly state that the method uses **unlabeled target-speaker adaptation** over a batch of target clips.

The second setting is still valid, but it is a different evaluation assumption from pure one-shot zero-shot generalization.

