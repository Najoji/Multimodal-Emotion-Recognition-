# Results — Guide for Reviewers

Purpose
-------
This directory consolidates the final outputs produced by the project and provides a concise navigation path for reviewers. Files here are organized to make the principal evaluation artifacts immediately discoverable while preserving a full experiment archive for reproducibility and traceability.

Quick Start (for reviewers)
---------------------------
- Read the project synopsis and methodology in `../FINAL_PROJECT_REPORT.md` for experimental context.
- Consult `tables/README.md` for definitions and schema of the result tables.
- Consult `plots/README.md` for descriptions of the figures and visualization conventions.

Primary reviewer-facing outputs
-------------------------------
These items are the authoritative artifacts for assessment and comparison:

- Tables: `Results/tables/`
  - Final accuracy summaries and per-class classification reports (train and test splits reported).
  - CSV files in this folder contain the numeric results used for plots and the summary tables in the report.

- Plots: `Results/plots/`
  - Confusion matrices for speech-only, text-only, and multimodal (fusion) models.
  - Representation visualizations that illustrate embedding spaces and class separability.

Headline result and evaluation protocol
--------------------------------------
The principal cross-speaker evaluation uses a speaker-holdout protocol (train on OAF, test on YAF, then the reciprocal split). The headline metric reported for this protocol is:

```
Emotion2Vec+ (speech-only) — speaker-holdout accuracy: 99.86%
```

Note: speaker-holdout results are the preferred measure of cross-speaker generalization. Random-split baselines are retained for historical comparison but should not be directly compared to speaker-holdout numbers.

Historical baselines and archive
--------------------------------
- Historical/random-split baselines: `Results/archive/tables/random_split_baselines.csv` (context only).
- Full experiment archive (sweeps, intermediate checkpoints, discarded runs): see `Results/archive/README.md`, `Results/archive/tables/notes.md`, and `Results/archive/plots/README.md`.

Reproducibility pointers
------------------------
- Checkpoints and trained models: see `Results/checkpoints/`.
- Cached embeddings and precomputed features: see `Results/embedding_cache/`.
- For reproduction scripts and training details, consult the notebooks and scripts in the top-level `models/` and `src/` folders described in the main project README.

If you need a specific table, plot, or experiment trace, indicate the artifact name and I will point you to the exact file and relevant commit/checkpoint.
