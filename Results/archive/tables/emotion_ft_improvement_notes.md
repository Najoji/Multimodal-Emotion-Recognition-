# Emotion-Finetuned Wav2Vec2 Improvement Notes (Archive)

This note summarizes the gains from switching to an **emotion-specialized pretrained speech model** after generic wav2vec2 baselines plateaued.

## Model Used

```
audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
```

Unlike `wav2vec2-base` (generic speech representation), this model is finetuned specifically for emotion-related speech cues.

## Main Results (Speaker-Holdout)

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

Mean + standard deviation pooling provides the strongest overall performance.

## Interpretation

The large jump from ~63% to ~85% shows that **representation quality dominates** small classifier changes. An emotion-specialized pretrained model transfers far better to speaker-independent TESS evaluation than a generic speech model.

## Best Strict Speaker-Holdout Result

```
84.89%
```

## Related Files (Archive Tables)

These files capture the detailed sweeps and comparisons for this stage:

```
Results/archive/tables/wav2vec2-large-robust-12-ft-emotion-msp-dim_speaker_holdout_predictions.csv
Results/archive/tables/emotion_ft_wav2vec2_domain_adaptation.csv
Results/archive/tables/emotion_ft_wav2vec2_domain_adaptation_summary.csv
Results/archive/tables/emotion_ft_svm_sweep.csv
Results/archive/tables/emotion_ft_svm_sweep_summary.csv
Results/archive/tables/emotion_ft_pooling_compare.csv
Results/archive/tables/emotion_ft_pooling_compare_summary.csv
```

