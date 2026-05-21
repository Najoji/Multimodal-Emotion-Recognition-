# Overfitting and Generalization Assessment (Archive)

This note documents why early random-split results looked extremely strong, and why speaker-holdout evaluation is the more honest test of generalization on TESS.

## 1) Random-Split Results (easy setting)

The speech-only baseline performed almost perfectly on a standard random split:

| Evaluation | Accuracy |
| --- | ---: |
| Train split | 100.00% |
| Random held-out test split | 99.82% |

The tiny train/test gap suggests no obvious overfitting **within the same distribution**. However, random split is easy on TESS because both speakers and the same spoken words appear in both train and test.

## 2) Speaker-Holdout Results (harder, realistic setting)

When the model is forced to generalize to an unseen speaker, performance drops sharply:

| Train Speaker | Test Speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 48.07% |
| YAF | OAF | 50.93% |

This indicates strong **speaker/style dependence**. The 99.82% random-split result should therefore be treated as a baseline for a controlled setting, not as evidence of robust generalization.

## 3) Interpretation

The model performs very well when train and test come from the same speaker distribution, but struggles on speaker-independent evaluation. This limitation should be explicitly noted in any discussion of results.

## 4) Simple Follow-Up Experiments

Several classical models were tested under speaker-holdout. The best simple variant was:

| Model | Feature Set | Average Speaker-Holdout Accuracy |
| --- | --- | ---: |
| Linear SVM | MFCC + energy/spectral prosody | 53.93% |

This is only a small improvement over the initial speaker-holdout baseline. The core issue is not just classifier choice; the dataset contains only two speakers, so the style shift is large.

The full comparison is saved in `Results/archive/tables/speaker_holdout_experiments_summary.csv`.

## 5) Selected Upgrade Path

The practical upgrade combined richer handcrafted features (MFCCs, energy, spectral features, chroma, spectral contrast) with data augmentation (noise, volume, pitch shift, time stretching). Speaker-holdout accuracy improved to:

| Train Speaker | Test Speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 57.71% |
| YAF | OAF | 58.64% |

Average accuracy:

```text
58.18%
```

This is a meaningful gain over 48–51%, but still highlights how difficult speaker-independent emotion recognition is when only two speakers are available.
