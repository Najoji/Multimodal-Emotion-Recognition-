# Models Folder Guide

This folder contains the three official pipelines used in the project and a small visualization helper.

## What is here

- `speech_pipeline/` - speech-only model (Emotion2Vec+ embeddings)
- `text_pipeline/` - text-only model (TF-IDF baseline)
- `fusion_pipeline/` - speech + text fusion model
- `visualize_representations.py` - plots representation spaces used in the report

Each pipeline folder has its own README with model details and expected outputs.

## Common evaluation setup

All pipelines use **speaker-holdout** evaluation on TESS:

- Train on OAF, test on YAF
- Train on YAF, test on OAF

This is the primary evaluation reported in the project.

## Where outputs go

All pipelines write results into the main `Results/` folder:

- `Results/checkpoints/` - saved trained models
- `Results/tables/` - accuracy tables and classification reports
- `Results/plots/` - confusion matrices and representation plots

`Results/checkpoints/` is generated locally and is not committed to GitHub.

## Archive scripts

Each pipeline has an `archive/` subfolder with older experiments and baseline scripts. These scripts are kept for traceability.

## Quick links

- Speech pipeline details: `models/speech_pipeline/README.md`
- Text pipeline details: `models/text_pipeline/README.md`
- Fusion pipeline details: `models/fusion_pipeline/README.md`
