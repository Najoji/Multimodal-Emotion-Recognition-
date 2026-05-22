# Archived Results Guide

This folder contains **supplementary experimental outputs** that are not required for the final review but are preserved for traceability and deeper inspection. Use this section if you want to verify intermediate stages, compare historical baselines, or inspect discarded experiments.

## Folder Layout

- `tables/`
  - CSVs and notes for historical baselines, intermediate stages, and detailed sweeps.
  - This is the **primary archive** for reproducible numbers and reports.
  - See `tables/notes.md` for a complete, file-by-file index.

- `plots/`
  - Confusion matrices from older experiments and tests.
  - Useful for diagnosing failure modes in earlier baselines.
  - See `plots/README.md` for a full explanation of each plot.

## Quick Navigation

If you are looking for a specific kind of artifact:

- **Random-split baselines:** `Results/archive/tables/random_split_baselines.csv`
- **Stage-by-stage speech progression:** `Results/archive/tables/speech_stage1_mfcc_baseline.csv` through `Results/archive/tables/speech_stage4_emotion2vec_champion.csv`
- **Wav2Vec2 adaptation details:** `Results/archive/tables/speech_stage2_wav2vec2_details.csv` and `Results/archive/tables/speech_stage2_wav2vec2_adaptation.csv`
- **Emotion-finetuned Wav2Vec2:** `Results/archive/tables/speech_stage3_emotion_finetuned.csv`
- **Emotion2Vec+ champion (historical copy):** `Results/archive/tables/speech_stage4_emotion2vec_champion.csv`
- **Text-only assessments:** `Results/archive/tables/text_only_assessment.md` and `Results/archive/tables/text_speaker_holdout_accuracy.csv`
- **Fusion baselines:** `Results/archive/tables/fusion_accuracy.csv` and `Results/archive/tables/fusion_classification_report.csv`
- **Ensemble experiments:** `Results/archive/tables/ensemble_wav2vec2_*`
- **Wav2Vec2-large variants:** `Results/archive/tables/wav2vec2-large-960h_*` and `Results/archive/tables/wav2vec2-large-robust-12-ft-emotion-msp-dim_*`

## How to Use This Archive

1. Start with `tables/notes.md` for a complete map of the archive.
2. Use the CSVs for quantitative comparisons and the MD notes for narrative context.
3. Use `plots/README.md` to interpret older confusion matrices.

## Note

The reviewer-facing results are in `Results/tables/` and `Results/plots/`. This archive exists only for deeper context and reproducibility.
