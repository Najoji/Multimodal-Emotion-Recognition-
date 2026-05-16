# Text-Only Assessment

The text-only model was tested using the held-out test split from `Results/tables/text_only_test_split.csv`.

## Result

| Model | Test Accuracy |
| --- | ---: |
| Text-only TF-IDF + Logistic Regression | 0.00% |

## Why This Happens

The TESS dataset is not a strong dataset for text-only emotion recognition. Each transcript is usually a single neutral word, such as:

- `back`
- `bar`
- `dog`
- `road`
- `young`

The same words appear across all emotion classes. For example, the word `back` can be spoken as angry, happy, sad, fearful, neutral, and so on. Therefore, the word itself does not contain reliable emotion information.

In this project, the text-only model is still useful as a required baseline. It shows that emotion in TESS is carried mainly by acoustic speech cues rather than the transcript content.

A dummy classifier can score around chance level by ignoring the text and always or randomly guessing labels. That does not mean the transcript has useful emotion information. It only gives a lower reference point for comparison.

## Interpretation For Report

The text-only pipeline was evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. Its poor performance indicates that isolated word transcripts are insufficient for emotion prediction in TESS. A stronger text-only model such as BERT would not solve this specific limitation because the input text itself lacks emotional context.
