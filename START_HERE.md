# Start Here

If you are reviewing the finished submission rather than rebuilding it, begin with:

```text
REVIEWER_GUIDE.md
```

## What This Project Is

You are building an emotion classifier. Given an audio sample, its text, or both, the system predicts the speaker's emotion.

TESS has audio files where the emotion is encoded in the filename/folder name. Example:

```text
OAF_back_angry.wav
```

This means:

- speaker/style: `OAF`
- spoken word/text: `back`
- emotion label: `angry`

## First Milestone

Get a complete baseline working:

1. Download and extract the TESS dataset into `data/`.
2. Train the speech-only model.
3. Train the text-only model.
4. Train the fusion model.
5. Compare the three accuracies.
6. Generate confusion matrices and write a short analysis.
7. Generate 2D representation plots to show whether emotion clusters separate.

## Why We Start Simple

A simple baseline gives you:

- a working project quickly
- numbers for the report
- a clean explanation for interviews
- a foundation for better models later

Once this works, we can improve:

- speech: CNN/LSTM over spectrograms, wav2vec embeddings
- text: BERT embeddings
- fusion: learned neural fusion instead of simple concatenation

## Your Main Interview Story

"I first built explainable unimodal baselines for speech and text, then combined both modalities through feature-level fusion. I compared all three settings and analyzed where fusion helped or failed."
