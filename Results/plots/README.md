# Main Plots Guide

| File | What it shows |
| --- | --- |
| `speech_model_evolution.png` | Side-by-side speech representation progression: MFCC, generic Wav2Vec2, and final Emotion2Vec+. |
| `speech_representation_pca.png` | Final speech embedding space from the temporal modelling block. |
| `text_representation_svd.png` | Text representation space from the contextual modelling block; classes overlap strongly. |
| `fusion_representation_pca.png` | Fused speech + text representation space from the fusion block. |
| `model_comparison_bar.png` | Original baseline comparison chart. |

## How To Read The Representation Plots

Each point is one audio sample. Points with the same color share the same emotion label.

- mixed colors in the same area mean the representation is weakly separated
- compact same-color clusters mean the representation is easier for a classifier to separate

The most important plot for the project story is `speech_model_evolution.png`.

Supporting historical plots are kept in `../archive/plots/`.
