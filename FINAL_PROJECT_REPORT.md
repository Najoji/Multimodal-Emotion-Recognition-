# Multimodal Emotion Recognition on TESS

> This report is best viewed in Markdown Preview so the figures render inline. In VS Code, use `Ctrl+Shift+V`.

## Project Summary

This project builds and compares three emotion-recognition systems on the Toronto Emotional Speech Set (TESS):

1. speech only
2. text only
3. speech + text fusion

The final submission is not just a single model. It is a staged investigation of how speech representations improve from handcrafted acoustic features to emotion-specialized pretrained embeddings. The main evaluation uses **speaker holdout**, where one TESS speaker is used for training and the other is kept completely unseen for testing. This makes the result more meaningful than a random split that mixes both speakers into both sides.

The most important result is:

```text
MFCC baseline:             49.50%
Generic Wav2Vec2:          66.71%
Emotion-finetuned Wav2Vec2:84.89%
Emotion2Vec+ champion:     99.86%
```

For historical context, the original random-split baselines were:

| Early baseline | Random-split accuracy |
| --- | ---: |
| Speech-only MFCC | 99.82% |
| Text-only TF-IDF | 0.00% |
| Fusion MFCC + TF-IDF | 99.82% |

These early results are useful because they explain why speaker holdout became necessary.

### Executive Summary

The project began with the simple interpretation of the task: build one speech model, one text model, one fusion model, and compare their accuracies. During development, a much more important issue emerged. A random split produced almost perfect speech accuracy, but this was misleading because both TESS speakers were present in both the training and test sets. Once a stricter speaker-holdout evaluation was introduced, the initial MFCC model fell to roughly `49%`, revealing that the harder and more meaningful problem was **generalization to an unseen speaker**.

From there, the work became a controlled progression:

1. establish an interpretable handcrafted baseline
2. improve beyond handcrafted features using generic pretrained speech representations
3. test whether emotion-specific pretraining matters
4. evaluate a model designed directly for speech emotion representation

That progression led to the final conclusion of the project: **the quality of the learned representation mattered much more than the complexity of the classifier**.

### Final Visual Summary

![Speech Evolution](Results/plots/speech_model_evolution.png)

*Figure 1. The central result of the project: speech representations become progressively more separable from MFCCs to Emotion2Vec+.*

## A. Architecture Decisions

### 1. Preprocessing

**Speech:** Each audio file is loaded as mono audio, resampled to `16 kHz`, and trimmed for leading/trailing silence. This gives every model a consistent input format while keeping the emotionally useful parts of the waveform.

**Text:** The transcript is recovered from the TESS filename and normalized into a clean text field. TESS contains isolated spoken words rather than full emotional sentences, so the text branch is intentionally simple.

**Why this mattered:** Audio samples naturally vary in duration, leading silence, and waveform scale. Standardizing the signal before feature extraction reduces irrelevant variation and makes later comparisons between model stages fairer. For text, the goal was not to invent information that the dataset does not contain, but to represent the available transcript honestly.

### 2. Feature Extraction

**Speech stage 1:** The first baseline uses handcrafted acoustic descriptors:

- 40 MFCCs
- delta MFCCs
- delta-delta MFCCs
- fixed-length summary statistics over time

This is a standard interpretable baseline for speech emotion work.

MFCCs were chosen first because they are compact, explainable, and closely tied to the spectral envelope of speech. Delta and delta-delta features add short-term motion information, which helps capture how the voice changes over time rather than treating every utterance as static.

**Speech stage 2:** Generic `Wav2Vec2-base` embeddings are extracted and pooled over time. These representations are richer than MFCCs because they are learned from large-scale speech data.

**Speech stage 3:** An emotion-finetuned Wav2Vec2 representation is used to inject task-specific emotional knowledge before classification.

**Speech stage 4:** `Emotion2Vec+ base` utterance embeddings are used as the final speech representation. This model is specialized for speech emotion information and produced the strongest speaker-independent result in this project.

**Text:** TF-IDF unigrams and bigrams are used. This is appropriate for a lightweight text baseline, although the TESS transcripts themselves contain very little emotion information.

The feature-extraction design intentionally evolves from **human-designed acoustic cues** to **learned general speech cues** to **learned emotion-specific cues**. This made it possible to test not only which model was best, but also *why* the score improved.

### 3. Temporal / Contextual Modelling

**Speech:** The pretrained speech models perform the main temporal modelling internally before the utterance embedding is extracted. For the MFCC baseline, time-varying acoustic tracks are summarized into fixed-length statistics.

**Text:** TF-IDF provides a bag-of-words contextual baseline. Since TESS uses single words, richer contextual models would not add meaningful linguistic context.

This block is one of the strongest contrasts between the modalities. Speech contains emotion in timing, energy, pitch movement, and spectral shape. Text in TESS contains almost none of that because the same isolated words are spoken in every emotion. Therefore, the speech branch benefits enormously from stronger temporal representations, while the text branch is structurally limited by the dataset itself.

### 4. Fusion

The fusion pipeline concatenates the final speech representation with the TF-IDF text representation before classification. This keeps the multimodal design transparent and makes it easy to compare against the unimodal branches.

This choice was deliberately conservative. A simple fusion rule prevents the experiment from hiding weak text features behind a complicated architecture and makes it easier to answer the actual research question: **does text add useful information beyond speech on TESS?**

### 5. Classifier

The early baseline uses Logistic Regression. The stronger speech models use a Linear SVM, with normalization where useful. A linear classifier was kept on purpose so that gains would come mainly from better representations rather than from hiding weak features behind a complex classifier.

This was an important design discipline. If the final score improved while the classifier stayed simple, then the improvement could be attributed primarily to the representation rather than to a heavily tuned downstream model.

## B. Experiments

### Evaluation Design

The main comparison uses **speaker holdout** rather than a random split:

- train on `OAF`, test on `YAF`
- train on `YAF`, test on `OAF`

This is stricter than random splitting because the test speaker is never seen during training.

### Why Speaker Holdout Became Necessary

The project first produced these random-split baseline results:

| Early baseline | Random-split accuracy |
| --- | ---: |
| Speech-only MFCC | 99.82% |
| Text-only TF-IDF | 0.00% |
| Fusion MFCC + TF-IDF | 99.82% |

The speech and fusion results looked impressive, but they were not the right answers to trust. In a random split, recordings from both TESS speakers appear in both training and testing, so the model can quietly exploit speaker-specific patterns that do not generalize.

Speaker holdout forces a more realistic test:

- the model must learn emotion cues that transfer from one voice to another
- every test utterance comes from a speaker the classifier has never seen
- a large drop from random-split accuracy becomes evidence of overfitting to speaker identity

This decision changed the project from a superficial accuracy exercise into a genuine generalization study.

### Development Progression

The project moved through four clear stages:

| Stage | What changed | What was learned |
| --- | --- | --- |
| 1 | Built a simple MFCC baseline | A working model is not the same as a generalizing model; speaker holdout exposed a weak `49.50%` result. |
| 2 | Replaced handcrafted features with generic Wav2Vec2 embeddings | Pretraining helped, but generic speech knowledge alone still left a large speaker gap. |
| 3 | Switched to an emotion-finetuned speech model | Task-specific emotional representations mattered far more than classifier tricks. |
| 4 | Tested Emotion2Vec+ | A model designed specifically for speech emotion produced near-perfect separation on TESS. |

### Stage-By-Stage Experiment Notes

#### Stage 1: MFCC Baseline

The first serious holdout result used MFCC statistics with Logistic Regression. It achieved an average accuracy of `49.50%`. This was valuable precisely because it was not flattering: it established the starting point and showed that handcrafted features alone were insufficient for robust cross-speaker emotion recognition.

#### Stage 2: Generic Wav2Vec2

Generic `Wav2Vec2-base` embeddings improved the result to `62.90%` before adaptation. After testing speaker-wise normalization methods, the best unsupervised adaptation setup reached `66.71%`. This confirmed that pretrained speech knowledge helps, but also showed that a model trained for general speech is not automatically ideal for emotion recognition.

#### Stage 3: Emotion-Finetuned Wav2Vec2

The emotion-finetuned representation produced the first major leap, reaching `84.89%` after normalization and classifier tuning. This was the point where the project clearly demonstrated that **task-specific pretraining beats generic pretraining** for this problem.

#### Stage 4: Emotion2Vec+ Champion

`Emotion2Vec+ base` produced `99.86%` average speaker-holdout accuracy. The jump was so large that an additional sanity check was performed: training labels were intentionally shuffled, and accuracy collapsed near chance. That confirmed the score was not simply an artifact of a broken split.

### Speech Model Progression

| Stage | Representation | Main setup | Historical random-split accuracy | Average speaker-holdout accuracy |
| --- | --- | --- | ---: | ---: |
| 1 | MFCC baseline | MFCC statistics + Logistic Regression | 99.82% | 49.50% |
| 2 | Generic Wav2Vec2 | Wav2Vec2-base + unsupervised speaker adaptation | Not used as the main metric | 66.71% |
| 3 | Emotion-finetuned Wav2Vec2 | emotion-finetuned embedding + Linear SVM | Not used as the main metric | 84.89% |
| 4 | Emotion2Vec+ champion | Emotion2Vec+ base + L2 normalization + Linear SVM | Not used as the main metric | 99.86% |

### Final Modality Comparison

| Model | Historical random-split baseline | Final speaker-holdout accuracy |
| --- | ---: | ---: |
| Speech | 99.82% | 99.86% |
| Text | 0.00% | 14.29% |
| Fusion | 99.82% | 99.86% |

The random-split and speaker-holdout columns should not be read as the same experiment. The random-split column records the original early baselines, while the speaker-holdout column records the final honest comparison. The text branch remains weak because the same isolated words appear across all emotion classes. Fusion matches speech-only because the text features add almost no extra discriminative information on TESS.

### Additional Experiment Outcomes

Several side experiments were also useful even though they did not become the final model:

- Enhanced handcrafted features and augmentation improved the classical pipeline from roughly `49.50%` to `58.18%`, but the gain was still much smaller than the gain from pretrained emotion representations.
- Adding handcrafted features back on top of the strong emotion-finetuned embedding made performance worse, showing that more features are not automatically better.
- Pseudo-label adaptation slightly improved the earlier emotion-finetuned model from `84.89%` to `85.46%`, but once Emotion2Vec+ was introduced, adaptation was no longer the main bottleneck.
- Fusion failed to improve over the final speech model because the text branch had almost no independent emotional signal to contribute.

### Important Findings

1. The first random-split speech score looked unrealistically high, so speaker holdout was introduced to measure actual cross-speaker generalization.
2. Better handcrafted features and augmentation improved the classical pipeline only modestly.
3. Generic pretrained embeddings helped more than handcrafted features.
4. The largest jump came from **choosing a better representation**, not from using a more complicated classifier.
5. The text branch is not truly informative on TESS because the spoken words themselves are almost emotion-neutral.
6. Fusion only helps when both modalities add useful information; here, speech carries almost everything.

### Clean Master Results Table

| Family | Model / method | Historical random-split accuracy | Average speaker-holdout accuracy | Interpretation |
| --- | --- | ---: | ---: | --- |
| Classical speech | MFCC + Logistic Regression | 99.82% | 49.50% | Random split looked excellent, holdout exposed weak unseen-speaker generalization |
| Classical speech | Enhanced handcrafted features + augmentation | Not used as the main metric | 58.18% | Helpful, but still limited |
| Generic pretrained speech | Wav2Vec2-base | Not used as the main metric | 62.90% | Learned speech features beat handcrafted features |
| Generic pretrained speech | Wav2Vec2-base + adaptation | Not used as the main metric | 66.71% | Speaker shift can be reduced, not fully solved |
| Emotion-specialized speech | Emotion-finetuned Wav2Vec2 | Not used as the main metric | 84.89% | Task-specific representations matter greatly |
| Emotion-specialized speech | Emotion2Vec+ base | Not used as the main metric | 99.86% | Best model on TESS speaker holdout |
| Text only | TF-IDF + Logistic Regression | 0.00% | 14.29% | Near chance because transcripts lack emotional content |
| Fusion | MFCC + TF-IDF baseline / Emotion2Vec+ + TF-IDF final | 99.82% | 99.86% | Text adds no measurable value on TESS |

## C. Analysis

### Which Emotions Were Easiest And Hardest?

With the final Emotion2Vec+ model, `angry`, `fear`, `happy`, `neutral`, and `sad` were classified perfectly in both speaker-holdout directions. The only remaining confusion came from a few `disgust` and `pleasant_surprise` samples, which were the hardest classes in the final system.

Earlier models struggled much more unevenly. In the MFCC baseline, performance depended heavily on the direction of the speaker transfer, and several emotions had very low recall. This supports the interpretation that the earlier representation was not stable enough across voices.

### When Did Fusion Help Most?

Fusion did not meaningfully improve the result on this dataset. The speech model already captured nearly all useful information, while the text branch was limited by the nature of TESS transcripts. In a dataset with full emotional sentences, longer conversations, or sentiment-bearing words, fusion would be more likely to help.

This is still an important finding, not a failure of the project. A multimodal system should be tested against the possibility that one modality is redundant. On TESS, the fusion experiment shows that adding a second modality is only useful if that modality contains genuinely discriminative information.

### Error Analysis

The final model made only four errors across the two speaker-holdout tests:

| File | True label | Predicted label |
| --- | --- | --- |
| `YAF_puff_disgust.wav` | disgust | pleasant_surprise |
| `YAF_yes_disgust.wav` | disgust | pleasant_surprise |
| `YAF_doll_ps.wav` | pleasant_surprise | neutral |
| `OAF_pad_ps.wav` | pleasant_surprise | disgust |

These failures are plausible because both `disgust` and `pleasant_surprise` can involve sharp expressive changes in pitch and energy, especially in acted speech.

Even though only four final mistakes remain, retaining them in the report is useful. They prove that the evaluation was performed at the per-file level and provide a concrete starting point for qualitative listening-based analysis.

### Representation Visualizations

#### Speech Representation Evolution

![Speech Evolution](Results/plots/speech_model_evolution.png)

*Figure 2. Evolution of the speech representation space from handcrafted MFCCs to Emotion2Vec+ embeddings.*

#### Final Temporal Modelling Representation

![Final Speech Representation](Results/plots/speech_representation_pca.png)

*Figure 3. Final speech representation from the temporal modelling block.*

#### Contextual Modelling Representation

![Text Overlap](Results/plots/text_representation_svd.png)

*Figure 4. Text representation space. The classes overlap because the transcripts reuse the same words across emotions.*

#### Fusion Representation

![Fusion Clusters](Results/plots/fusion_representation_pca.png)

*Figure 5. Final fused representation space. The separation mostly comes from the speech branch.*

The speech evolution figure matches the numerical results very closely. The visual story moves from a chaotic MFCC cloud to clean Emotion2Vec+ islands: the MFCC baseline forms a crowded, partially overlapping cloud, which is consistent with its roughly `49%` speaker-holdout accuracy. Standard Wav2Vec2 changes the representation space but still leaves substantial overlap between emotion classes. By the final Emotion2Vec+ stage, the points form clean, compact islands with very little class mixing, which explains the jump to almost `99%` accuracy on the TESS speaker-holdout test.

The text visualization stays heavily overlapped because the same words occur in multiple emotion classes. The fusion visualization therefore largely inherits the strong structure of the speech branch rather than creating a new separation pattern from text.

### Why The Visualization Matters

The plots are not only decorative. They are a geometric explanation of the results:

- the MFCC space is visibly mixed, so a simple classifier has difficulty drawing reliable emotion boundaries
- the generic Wav2Vec2 space contains some structure, but the clusters still overlap
- the Emotion2Vec+ space has compact class regions, so even a linear classifier can separate the emotions cleanly

The visual and numerical evidence therefore tell the same story from two different angles.

### Final Interpretation

The most important lesson from the experiments is that **representation quality mattered more than classifier complexity**. Better handcrafted features gave modest gains, generic pretrained speech features helped more, and emotion-specialized embeddings produced the largest improvement by far.

The final `99.86%` score should be interpreted carefully: it is an excellent result on the controlled TESS speaker-holdout setting, not proof of `99%` accuracy on arbitrary real-world audio. A stronger generalization claim would require evaluation on additional speakers, recording conditions, and external datasets.

### Limitations And Future Work

This project reaches a very strong result on TESS, but TESS itself is a controlled dataset:

- only two speakers
- acted rather than spontaneous emotion
- clean studio recordings
- short isolated utterances

For a stronger real-world claim, the next step would be external evaluation on datasets with more speakers, more natural speech, more recording conditions, and cross-corpus testing. Data augmentation could also be revisited as a robustness study rather than as the main path to higher TESS accuracy.

### Deliverable Check

| PDF requirement | How this project satisfies it |
| --- | --- |
| Speech-only model | `models/speech_pipeline/` |
| Text-only model | `models/text_pipeline/` |
| Multimodal fusion model | `models/fusion_pipeline/` |
| Accuracy tables | `Results/tables/` |
| Plots | `Results/plots/` |
| Architecture decisions | Section A |
| Speech/text/fusion comparison | Section B |
| Easiest/hardest emotions | Section C |
| Fusion analysis | Section C |
| 3-5 failure cases | Section C |
| Temporal, contextual, and fusion visualizations | Figures 2-5 |
