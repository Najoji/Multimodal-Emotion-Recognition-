# Project Log

This is my clean record of how the project developed, what I changed, what worked, what failed, and what I should remember when explaining the work later.

## 1. Project Goal

The project is about multimodal emotion recognition using:

1. speech only
2. text only
3. speech and text together

The required functional blocks are:

- preprocessing
- feature extraction
- temporal/contextual modelling
- fusion
- classifier

The dataset used throughout the project is TESS.

## 2. Project Structure

I organized the repository around the deliverables from the project brief:

```text
models/
  speech_pipeline/
  text_pipeline/
  fusion_pipeline/
src/
  speech_emotion/
Results/
  tables/
  plots/
  archive/
  checkpoints/
  embedding_cache/
README.md
FINAL_PROJECT_REPORT.md
requirements.txt
```

The final reviewer-facing outputs are kept in `Results/tables/` and `Results/plots/`. Older exploratory outputs are preserved in `Results/archive/` so the main folders stay readable.

## 3. Dataset Loading

I created:

```text
src/speech_emotion/dataset.py
```

The dataset loader:

- searches recursively for `.wav` files
- extracts the transcript word from each filename
- extracts the emotion label from the filename/folder
- normalizes labels such as `pleasant_surprise`
- removes duplicate filenames

I discovered that the dataset had been extracted twice inside itself. There were 5600 audio files physically present, but only 2800 unique TESS samples. The loader now deduplicates by filename.

Final clean dataset:

```text
2800 unique clips
7 emotions
400 clips per emotion
```

## 4. Environment Setup

I used Python 3.11 in a virtual environment and installed the main packages listed in `requirements.txt`.

Core packages used during the project include:

- librosa
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- joblib
- torch
- transformers
- funasr
- modelscope

The complete setup commands are documented in `README.md`.

## 5. First Speech Baseline

Main files:

```text
models/speech_pipeline/train.py
models/speech_pipeline/test.py
src/speech_emotion/audio_features.py
```

Speech preprocessing:

- convert audio to mono
- resample to 16 kHz
- trim silence

Initial speech features:

- 40 MFCCs
- delta MFCCs
- delta-delta MFCCs

Each track was summarized with:

- mean
- standard deviation
- minimum
- maximum

That created:

```text
120 feature tracks x 4 summary values = 480 features
```

Classifier:

```text
StandardScaler + LogisticRegression
```

Random split result:

```text
99.82% accuracy
```

At first this looked excellent, but later evaluation showed that it was too easy because both speakers were present in both training and testing.

## 6. Text Baseline

Main files:

```text
models/text_pipeline/train.py
models/text_pipeline/test.py
models/text_pipeline/baseline_compare.py
```

Text representation:

```text
TF-IDF unigrams and bigrams
```

Classifier:

```text
LogisticRegression
```

The text-only branch performed poorly because TESS transcripts are isolated words such as:

```text
back
dog
road
young
```

The same words appear across all emotions, so the transcript itself carries almost no emotion signal.

This became an important project finding rather than just a bad result.

Random split result:

```text
0.00% accuracy
```

## 7. First Fusion Baseline

Main files:

```text
models/fusion_pipeline/train.py
models/fusion_pipeline/test.py
```

Fusion method:

```text
speech features + TF-IDF text features
```

Classifier:

```text
LogisticRegression
```

Random split result:

```text
99.82% accuracy
```

The fusion result matched speech-only because the text branch added almost no useful information on TESS.

## 8. Why Random Splits Were Not Enough

The original speech result was suspiciously high, so I tested a stricter evaluation.

In the random split:

- both speakers appear in train and test
- the model can partially benefit from speaker-specific information

I then used strict speaker holdout:

| Train speaker | Test speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 48.07% |
| YAF | OAF | 50.93% |

Average:

```text
49.50%
```

This was the first major turning point in the project. The model was not useless, but it did not generalize nearly as well as the random split suggested.

## 9. Handcrafted Feature Improvements

I next tried to improve the classical speech pipeline before moving to pretrained models.

Additional features tested:

- RMS energy
- zero-crossing rate
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- chroma
- spectral contrast
- optional pitch statistics

Augmentations tested:

- additive noise
- louder/quieter variants
- pitch shift
- time stretch

I also compared several classifiers:

- Logistic Regression
- Linear SVM
- RBF SVM
- Random Forest
- Extra Trees

Best handcrafted progression:

| Setup | Average speaker-holdout accuracy |
| --- | ---: |
| Original MFCC baseline | 49.50% |
| Best simple sweep | 53.93% |
| Enhanced features + augmentation + Linear SVM | 58.18% |

The handcrafted route helped, but it clearly had a ceiling.

## 10. Generic Pretrained Wav2Vec2

Main file:

```text
models/speech_pipeline/pretrained_embedding_holdout.py
```

Model:

```text
facebook/wav2vec2-base
```

I extracted frame-level hidden representations and pooled them with:

```text
mean + standard deviation over time
```

The final vector size was:

```text
1536 features
```

Speaker-holdout result:

| Train speaker | Test speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 64.79% |
| YAF | OAF | 61.00% |

Average:

```text
62.90%
```

This was a real improvement over handcrafted features and showed that learned speech representations were more useful than manual descriptors alone.

## 11. Normalization And Speaker Adaptation

Main file:

```text
models/speech_pipeline/wav2vec2_domain_adaptation.py
```

I tested whether the speaker shift could be reduced after embedding extraction.

Strict zero-shot setting:

| Setup | Average accuracy |
| --- | ---: |
| Wav2Vec2 baseline | 62.90% |
| Wav2Vec2 + L2 normalization | 63.75% |

Unsupervised target-statistics setting:

| Setup | Average accuracy |
| --- | ---: |
| speaker-wise z-score + L2 normalization | 66.71% |

The main idea was simple: the two speakers occupy slightly different regions of the feature space, and normalization helps the classifier focus more on emotion than speaker identity.

## 12. Emotion-Finetuned Wav2Vec2

I then tested an emotion-specialized pretrained representation:

```text
audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
```

Initial result:

| Train speaker | Test speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 85.14% |
| YAF | OAF | 82.21% |

Average:

```text
83.68%
```

After normalization and classifier tuning:

```text
84.89%
```

This was the first major jump. It showed that emotion-specific representations were much more important than endlessly tuning the classifier.

## 13. Extra Experiments Around The Emotion-Finetuned Model

I tested several follow-up ideas after reaching the mid-80s.

### Feature Fusion

I combined the pretrained embedding with handcrafted features.

| Setup | Average accuracy |
| --- | ---: |
| Embedding only | 84.89% |
| Handcrafted only | 55.93% |
| Embedding + handcrafted features | 80.11% |

Adding weaker handcrafted features made the stronger representation worse.

### Pooling Comparison

| Pooling | Average accuracy |
| --- | ---: |
| Mean + standard deviation | 84.89% |
| Mean only | 84.36% |
| Standard deviation only | 79.79% |

Mean plus standard deviation was the best pooling choice.

### Pseudo-Label Adaptation

Using confident pseudo-labels from the unseen speaker produced a small improvement:

```text
84.89% -> 85.46%
```

This showed that there was still some room for adaptation, but the main gain still came from the representation itself.

## 14. Emotion2Vec+ Champion

Main file:

```text
models/speech_pipeline/emotion2vec_holdout.py
```

Model:

```text
iic/emotion2vec_plus_base
```

Pipeline:

```text
Emotion2Vec+ utterance embedding
+ StandardScaler
+ L2 normalization
+ Linear SVM with C=0.1
```

Speaker-holdout result:

| Train speaker | Test speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 99.79% |
| YAF | OAF | 99.93% |

Average:

```text
99.86%
```

Because this result was unusually high, I checked for obvious evaluation mistakes:

- the split still used different speakers for train and test
- each speaker contributed 1400 clips
- the confusion matrices were almost perfectly diagonal
- a shuffled-label sanity check collapsed near chance

Shuffled-label control:

| Train speaker | Test speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 15.36% |
| YAF | OAF | 9.14% |

The result is therefore valid for the controlled TESS speaker-holdout setup. I still need to describe it carefully: it does **not** prove that arbitrary outside audio will also reach 99% accuracy.

## 15. Final Text And Fusion Speaker-Holdout Results

### Text only

Main file:

```text
models/text_pipeline/speaker_holdout.py
```

Result:

| Train speaker | Test speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 14.29% |
| YAF | OAF | 14.29% |

Average:

```text
14.29%
```

This is exactly chance level for seven classes, which confirms that the transcript words themselves are not useful for emotion recognition on TESS.

### Fusion

Main file:

```text
models/fusion_pipeline/speaker_holdout.py
```

Result:

| Train speaker | Test speaker | Accuracy |
| --- | --- | ---: |
| OAF | YAF | 99.79% |
| YAF | OAF | 99.93% |

Average:

```text
99.86%
```

Fusion matched speech-only instead of improving it because the text branch contributes almost no additional emotion signal.

## 16. Visualizations

Main file:

```text
models/visualize_representations.py
```

Final reviewer-facing plots:

- `Results/plots/speech_model_evolution.png`
- `Results/plots/speech_representation_pca.png`
- `Results/plots/text_representation_svd.png`
- `Results/plots/fusion_representation_pca.png`

The most useful plot is the speech evolution figure:

1. MFCC space is mixed and overlapping
2. generic Wav2Vec2 is still quite overlapped
3. Emotion2Vec+ forms clean emotion clusters

The visual progression matches the accuracy progression from about `49%` to almost `99%`.

## 17. Final Results To Remember

| Approach | Historical random-split accuracy | Average speaker-holdout accuracy |
| --- | ---: | ---: |
| MFCC baseline | 99.82% | 49.50% |
| Enhanced handcrafted features + augmentation | Not used as the main metric | 58.18% |
| Generic Wav2Vec2 baseline | Not used as the main metric | 62.90% |
| Generic Wav2Vec2 + adaptation | Not used as the main metric | 66.71% |
| Emotion-finetuned Wav2Vec2 | Not used as the main metric | 84.89% |
| Emotion2Vec+ champion | Not used as the main metric | 99.86% |
| Text only | 0.00% | 14.29% |
| Fusion | 99.82% | 99.86% |

## 18. Main Lessons

1. Random splits can make a weak model look excellent when speaker identity leaks across train and test.
2. Speaker holdout is the more honest evaluation for TESS.
3. Better handcrafted features helped, but only modestly.
4. Generic pretrained speech features helped more than handcrafted features.
5. Emotion-specialized representations produced the largest gains.
6. Text-only emotion recognition is ineffective on TESS because the text is almost emotion-neutral.
7. Fusion only helps when both modalities carry useful information.
8. Representation quality mattered more than classifier complexity throughout the project.

## 19. Important Final Files

### Documentation

```text
README.md
FINAL_PROJECT_REPORT.md
PROJECT_LOG.md
```

### Final tables

```text
Results/archive/tables/speech_stage1_mfcc_baseline.csv
Results/archive/tables/speech_stage2_wav2vec2_adaptation.csv
Results/archive/tables/speech_stage3_emotion_finetuned.csv
Results/archive/tables/speech_stage4_emotion2vec_champion.csv
Results/archive/tables/text_speaker_holdout_accuracy.csv
Results/tables/fusion_speaker_holdout_accuracy.csv
```

### Final plots

```text
Results/plots/speech_model_evolution.png
Results/plots/speech_representation_pca.png
Results/plots/text_representation_svd.png
Results/plots/fusion_representation_pca.png
```
