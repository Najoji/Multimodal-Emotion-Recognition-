# Emotion-Specialized Pretrained Model Results

The strongest speaker-holdout result found so far uses:

```text
audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
```

This is an emotion-specialized pretrained speech model, unlike generic `wav2vec2-base`, which is mainly pretrained for general speech representation / ASR transfer.

## Main Results

| Method | OAF -> YAF | YAF -> OAF | Average |
| --- | ---: | ---: | ---: |
| Wav2Vec2-base + Linear SVM | 64.79% | 61.00% | 62.90% |
| Emotion-specialized Wav2Vec2 + Linear SVM | 85.14% | 82.21% | 83.68% |
| Emotion-specialized Wav2Vec2 + L2 normalization + Linear SVM | 85.71% | 83.93% | 84.82% |
| Emotion-specialized Wav2Vec2 + L2 normalization + Linear SVM (`C=0.1`) | 85.86% | 83.93% | 84.89% |

## Pooling Comparison

| Pooling | Average Accuracy |
| --- | ---: |
| Mean + standard deviation | 84.89% |
| Mean only | 84.36% |
| Standard deviation only | 79.79% |

## Interpretation

The large improvement shows that the choice of pretrained representation matters more than small classifier changes. A model already trained for emotion-related speech structure transfers much better to speaker-independent TESS evaluation than a generic speech representation model.

## Current Best Strict Speaker-Holdout Result

```text
84.89%
```

Files:

```text
Results/tables/wav2vec2-large-robust-12-ft-emotion-msp-dim_speaker_holdout.csv
Results/tables/emotion_ft_wav2vec2_domain_adaptation.csv
Results/tables/emotion_ft_wav2vec2_domain_adaptation_summary.csv
Results/tables/emotion_ft_svm_sweep.csv
Results/tables/emotion_ft_svm_sweep_summary.csv
Results/tables/emotion_ft_pooling_compare.csv
Results/tables/emotion_ft_pooling_compare_summary.csv
```

