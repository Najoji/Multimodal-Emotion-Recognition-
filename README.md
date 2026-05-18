# Multimodal Emotion Recognition

This project studies emotion recognition on the Toronto Emotional Speech Set (TESS) using:

1. speech-only input
2. text-only input
3. speech + text fusion

The main project result is a staged improvement in speaker-holdout speech accuracy:

| Stage | Model | Average accuracy |
| --- | --- | ---: |
| 1 | MFCC baseline | 49.50% |
| 2 | Generic Wav2Vec2 + adaptation | 66.71% |
| 3 | Emotion-finetuned Wav2Vec2 | 84.89% |
| 4 | Emotion2Vec+ champion | 99.86% |

For context, the original easy random-split baselines were:

| Early baseline | Random-split accuracy |
| --- | ---: |
| Speech-only MFCC | 99.82% |
| Text-only TF-IDF | 0.00% |
| Fusion MFCC + TF-IDF | 99.82% |

Those random-split values are kept as historical context; the final project comparison uses speaker holdout because it is a more honest cross-speaker test.

## Read First

- Full setup and reproduction guide: `SETUP_INSTRUCTIONS.md`
- Final report: `FINAL_PROJECT_REPORT.md`
- Development history: `PROJECT_LOG.md`
- Final tables guide: `Results/tables/README.md`
- Final plots guide: `Results/plots/README.md`

## Repository Layout

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
FINAL_PROJECT_REPORT.md
PROJECT_LOG.md
SETUP_INSTRUCTIONS.md
requirements.txt
```

## Dataset

Download TESS from Kaggle and place the extracted audio files inside:

```text
data/
```

The loader searches recursively for `.wav` files and removes duplicate filenames if the dataset was accidentally extracted twice.

## Final Outputs

Reviewer-facing results are stored in:

- `Results/tables/`
- `Results/plots/`

Older exploratory outputs are preserved in:

- `Results/archive/`

For installation steps, run commands, dataset setup, and public-repository submission notes, see `SETUP_INSTRUCTIONS.md`.
