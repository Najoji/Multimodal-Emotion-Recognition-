# Emotion2Vec+ Follow-Up

## Why I Tested It

After the emotion-specialized Wav2Vec2 model reached the mid-80s, the next sensible step was to try a model built more directly for speech emotion recognition instead of spending more time on augmentation first.

## Setup

- model: `iic/emotion2vec_plus_base`
- evaluation: strict speaker holdout
- classifier: `StandardScaler -> L2 normalization -> Linear SVM (C=0.1)`

## Result

| Train Speaker | Test Speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 99.79% |
| YAF | OAF | 99.93% |

Average speaker-holdout accuracy:

```text
99.86%
```

## Sanity Check

To check that the score was not coming from a broken split or accidental label leakage, I trained the same classifier after shuffling the training labels.

| Train Speaker | Test Speaker | Accuracy With Shuffled Labels |
| --- | --- | ---: |
| OAF | YAF | 15.36% |
| YAF | OAF | 9.14% |

These values are near chance for a 7-class task, so the real high score depends on meaningful emotion structure in the embeddings.

## Decision

The main model-selection question is now answered. `Emotion2Vec+ base` is far stronger on this dataset than:

- handcrafted features
- generic Wav2Vec2
- the earlier emotion-specialized Wav2Vec2 model

Because the result is already almost perfect on TESS speaker holdout, augmentation is no longer the first priority. It would make more sense as a robustness experiment later, not as the main path for improving the headline result.
