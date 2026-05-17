# Overfitting And Generalization Assessment

The speech-only model achieved high accuracy on the original random stratified split:

| Evaluation | Accuracy |
| --- | ---: |
| Train split | 100.00% |
| Random held-out test split | 99.82% |

This does not show a large train/test gap, so the model is not simply memorizing the exact training rows under the same random-split setting.

However, the random split is easy for TESS because both speakers and the same spoken words appear in both train and test. To check generalization more strictly, a speaker-holdout evaluation was run:

| Train Speaker | Test Speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 48.07% |
| YAF | OAF | 50.93% |

This shows that the model generalizes much less well to an unseen speaker. Therefore, the 99.82% result should be reported as a clean TESS random-split baseline, not as evidence that the system is robust to new speakers or real-world speech.

## Report Interpretation

The model performs extremely well when train and test samples come from the same controlled dataset distribution. Performance drops under speaker-independent evaluation, suggesting speaker/style dependence. This is an important limitation and should be mentioned honestly in the analysis section.

## How To Improve

- Use speaker-independent training and testing.
- Add more speakers and more diverse datasets.
- Use data augmentation such as noise, pitch shift, and speed perturbation.
- Try stronger speech representations such as wav2vec 2.0 embeddings.
- Add regularization or simpler feature selection if the model remains too dataset-specific.

## Simple Follow-Up Experiments

Several classical models were tested under speaker-holdout evaluation. The best simple variant was:

| Model | Feature Set | Average Speaker-Holdout Accuracy |
| --- | --- | ---: |
| Linear SVM | MFCC + energy/spectral prosody | 53.93% |

This is only a small improvement over the original logistic regression speaker-holdout result. The main issue is therefore not just classifier choice; the dataset has only two speakers, so training on one speaker and testing on the other creates a large speaker/style shift.

The full comparison is saved in `Results/tables/speaker_holdout_experiments_summary.csv`.

## Selected Upgrade Path

The selected practical upgrade was enhanced handcrafted features plus data augmentation. This used MFCCs, energy features, spectral features, chroma, spectral contrast, and augmentation with noise, volume changes, pitch shift, and time stretching.

Speaker-holdout result after this upgrade:

| Train Speaker | Test Speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 57.71% |
| YAF | OAF | 58.64% |

Average accuracy:

```text
58.18%
```

This is a meaningful improvement over the original 48-51% speaker-holdout result, but it still shows that speaker-independent emotion recognition remains difficult with only two speakers.
