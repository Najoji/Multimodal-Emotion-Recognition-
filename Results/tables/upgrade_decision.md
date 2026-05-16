# Upgrade Decision

Five possible improvement paths were considered:

1. Better handcrafted features
2. Data augmentation
3. CNN on spectrograms
4. Pretrained speech embeddings such as HuBERT, wav2vec 2.0, or WavLM
5. More diverse data

## Decision

The selected immediate path is:

```text
Better handcrafted features + data augmentation
```

This was chosen because it fits the current codebase, is easy to explain in the report, does not require GPU access, and directly targets the speaker-overfitting issue.

## What Was Added

Enhanced speech features:

- MFCC
- Delta MFCC
- Delta-delta MFCC
- RMS energy
- Zero-crossing rate
- Spectral centroid
- Spectral bandwidth
- Spectral rolloff
- Spectral flatness
- Chroma
- Spectral contrast
- Optional pitch mode

Data augmentation:

- Noise injection
- Volume increase/decrease
- Pitch shifting
- Time stretching

Classifier:

- Linear SVM

## Results

Random-split performance stayed very high:

| Model | Accuracy |
| --- | ---: |
| Enhanced + Augmentation + Linear SVM | 99.82% |

Speaker-holdout performance improved:

| Train Speaker | Test Speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 57.71% |
| YAF | OAF | 58.64% |

Average speaker-holdout accuracy:

```text
58.18%
```

This is better than the original speaker-holdout result of about 48-51%, but it is still far below the random-split result.

## Interpretation

The upgrade reduced some speaker dependence but did not fully solve speaker-independent emotion recognition. This supports the conclusion that TESS is limited for speaker-independent modelling because it has only two speakers.

## Next Best Path

If more improvement is required, the next recommended step is pretrained speech embeddings:

- wav2vec 2.0
- HuBERT
- WavLM

These models are trained on large speech corpora and are more likely to produce speaker-robust features than handcrafted audio features.

