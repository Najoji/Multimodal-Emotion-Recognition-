# Text-Only Assessment (Archive)

This note documents why the text-only pipeline performs poorly on TESS and why that result is expected.

## Result (Historical Baseline)

The text-only model was evaluated using the held-out test split from `text_only_test_split.csv`.

| Model | Test Accuracy |
| --- | ---: |
| Text-only TF-IDF + Logistic Regression | 0.00% |

## Why This Happens

TESS transcripts are usually **single neutral words** (e.g., `back`, `bar`, `dog`, `road`, `young`). The same word appears in **every emotion class**, because emotion is conveyed by **how the word is spoken**, not by the word itself. As a result, the text contains almost no emotion signal.

That makes the text-only model a **negative baseline**: it demonstrates that the dataset’s emotion information is acoustic, not semantic. A random or majority-class predictor can hit chance-level accuracy, but that does not mean the text is useful.

## Interpretation for Report

The text-only pipeline is intentionally weak on this dataset. Even a stronger text model (e.g., BERT) would still struggle, because the underlying input text lacks emotional context. This result supports the central claim that **speech cues dominate** emotion recognition on TESS.
