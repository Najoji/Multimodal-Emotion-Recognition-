# Submission Checklist

This checklist is based directly on the project PDF.

| PDF requirement | Status | Evidence |
| --- | --- | --- |
| Speech-only model code | Pass | `models/speech_pipeline/train.py`, `test.py`, speaker-holdout scripts |
| Text-only model code | Pass | `models/text_pipeline/train.py`, `test.py`, `speaker_holdout.py` |
| Fusion model code | Pass | `models/fusion_pipeline/train.py`, `test.py`, `speaker_holdout.py` |
| Results directory with accuracy tables | Pass | `Results/tables/` |
| Plots directory | Pass | `Results/plots/` |
| README | Pass | `README.md` |
| requirements file | Pass | `requirements.txt` |
| Report section A: architecture decisions | Pass | `Results/FINAL_PROJECT_REPORT.md` |
| Report section B: experiments | Pass | `Results/FINAL_PROJECT_REPORT.md` |
| Report section C: analysis | Pass | `Results/FINAL_PROJECT_REPORT.md` |
| Easiest/hardest emotions discussed | Pass | `Results/FINAL_PROJECT_REPORT.md` |
| Fusion analysis included | Pass | `Results/FINAL_PROJECT_REPORT.md` |
| 3-5 failure cases included | Pass | `Results/FINAL_PROJECT_REPORT.md` |
| Temporal modelling visualization | Pass | `Results/plots/speech_representation_pca.png` and `speech_model_evolution.png` |
| Contextual modelling visualization | Pass | `Results/plots/text_representation_svd.png` |
| Fusion block visualization | Pass | `Results/plots/fusion_representation_pca.png` |
| Reviewer-friendly navigation | Pass | `REVIEWER_GUIDE.md`, `Results/README.md`, folder-level README files |

## Remaining External Step

The PDF also asks for a public GitHub repository with public links. That cannot be verified from the local workspace alone; it still needs to be done when the final repository is published.
