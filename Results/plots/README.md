# Main Plots Guide 

Everything in this folder is part of the **final results**. No archive access is required for review.

## 1) Final Confusion Matrices (test-time evaluation)

These plots show which emotions are confused with each other for the held-out speaker.

- `speech_only_OAF_test_confusion_matrix.png` (trained on OAF, tested on YAF)
- `speech_only_YAF_test_confusion_matrix.png` (trained on YAF, tested on OAF)
- `text_only_OAF_test_confusion_matrix.png` (trained on OAF, tested on YAF)
- `text_only_YAF_test_confusion_matrix.png` (trained on YAF, tested on OAF)
- `fusion_OAF_test_confusion_matrix.png` (trained on OAF, tested on YAF)
- `fusion_YAF_test_confusion_matrix.png` (trained on YAF, tested on OAF)

## 2) Representation Visualization Plots

These plots visualize how well each representation separates emotions.

- `speech_model_evolution.png` — progression across the four speech stages (MFCC -> generic Wav2Vec2 -> emotion-finetuned Wav2Vec2 -> Emotion2Vec+)
- `speech_representation_pca.png` — PCA of final speech embeddings (Emotion2Vec+)
- `text_representation_svd.png` — SVD of TF-IDF text features (classes overlap strongly)
- `fusion_representation_pca.png` — PCA of fused speech + text features

## 3) Summary Plot (Historical Baseline)

- `model_comparison_bar.png` — random-split baseline comparison (historical reference, not the final speaker-holdout result)

## Reading Representation Plots 

Each point is one audio sample. Points with the same color share the same emotion label.

- Mixed colors in the same area mean the representation is weakly separated.
- Compact same-color clusters mean the representation is easier for a classifier to separate.

The most important plot for the project story is `speech_model_evolution.png`.
