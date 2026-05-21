# Results Guide (Reviewer Navigation)

This folder is the **single place to find all results**. It is organized into two layers so the final outcomes are easy to locate while still keeping full experimental traceability.

## 1) What You Should Read First

- `../FINAL_PROJECT_REPORT.md` — full report with methodology and analysis.
- `tables/README.md` — explains every final results table.
- `plots/README.md` — explains every final results plot.

## 2) Final Reviewer-Facing Results

These are the files you should use for grading and comparison:

- **Tables:** `Results/tables/`
	- Final accuracies and per-emotion classification reports
	- Includes both train-time and test-time reports (they match because the split is the same)

- **Plots:** `Results/plots/`
	- Final confusion matrices for speech, text, and fusion
	- Representation visualizations (speech/text/fusion)
	- Historical random-split bar chart clearly labeled as historical

## 3) The One Key Result

```
Emotion2Vec+ speech-only speaker-holdout accuracy = 99.86%
```

This value is the headline result under **speaker-holdout evaluation** (train on OAF, test on YAF, then vice versa). It is the most reliable indicator of cross-speaker generalization on TESS.

## 4) Historical Baselines (Context Only)

Random-split baselines are kept only for historical context:

```
Results/archive/tables/random_split_baselines.csv
```

These should **not** be compared directly to speaker-holdout results.

## 5) Full Experiment Archive (Optional)

If you want to trace earlier experiments, sweeps, and discarded baselines, see:

- `Results/archive/README.md`
- `Results/archive/tables/notes.md`
- `Results/archive/plots/README.md`

The archive is **supplementary** and not required for the final review.
