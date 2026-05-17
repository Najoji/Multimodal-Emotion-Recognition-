# Reviewer Guide

## Recommended Reading Order

1. `Results/FINAL_PROJECT_REPORT.md`
2. `Results/tables/README.md`
3. `Results/plots/README.md`
4. `SUBMISSION_CHECKLIST.md`
5. `PROJECT_LOG.md` only if the full development history is needed

## What The Project Does

The project compares three emotion-recognition settings on TESS:

1. speech only
2. text only
3. speech + text fusion

The final model uses an emotion-specialized speech representation and is evaluated with strict speaker holdout.

## The Main Result

| Model | Average speaker-holdout accuracy |
| --- | ---: |
| Speech-only champion | 99.86% |
| Text-only | 14.29% |
| Fusion | 99.86% |

The key conclusion is that representation quality mattered more than classifier complexity.

## Where To Find The Required Deliverables

| Requirement | Location |
| --- | --- |
| Speech code | `models/speech_pipeline/` |
| Text code | `models/text_pipeline/` |
| Fusion code | `models/fusion_pipeline/` |
| Final report | `Results/FINAL_PROJECT_REPORT.md` |
| Accuracy tables | `Results/tables/` |
| Representation plots | `Results/plots/` |
| Setup instructions | `README.md` |
| Dependency list | `requirements.txt` |

## Quick Reproduction Commands

```powershell
.\.venv\Scripts\python.exe models\text_pipeline\speaker_holdout.py
.\.venv\Scripts\python.exe models\fusion_pipeline\speaker_holdout.py
.\.venv\Scripts\python.exe models\visualize_representations.py
```

The cached embeddings are already stored under `Results/embedding_cache/`, so the visualization script does not need to rebuild them.

## Notes For Evaluation

- The main evaluation uses **speaker holdout**, not a random split.
- TESS text consists of isolated words, so the text branch is intentionally weak.
- The `99.86%` result is excellent for controlled TESS holdout testing, but it should not be interpreted as guaranteed real-world accuracy on unrelated audio.
- Older exploratory files were moved into `Results/archive/` to keep the final submission easier to navigate.
