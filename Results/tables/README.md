# Main Tables Guide

These are the reviewer-facing tables that support the final report.

| File | What it means |
| --- | --- |
| `random_split_baselines.csv` | Early baseline results under the original easy random split: speech `99.82%`, text `0.00%`, fusion `99.82%`. |
| `speech_stage1_mfcc_baseline.csv` | First honest speech baseline: MFCC features with strict speaker holdout. |
| `speech_stage1_mfcc_baseline_*_classification_report.csv` | Per-emotion precision, recall, and F1 for the MFCC baseline in each holdout direction. |
| `speech_stage2_wav2vec2_baseline.csv` | Generic Wav2Vec2 speaker-holdout result before adaptation. |
| `speech_stage2_wav2vec2_adaptation.csv` | Domain-adaptation comparison for Wav2Vec2; the best row reaches `66.71%`. |
| `speech_stage2_wav2vec2_*_classification_report.csv` | Per-emotion reports for the generic Wav2Vec2 stage. |
| `speech_stage3_emotion_finetuned.csv` | Comparison showing why the emotion-finetuned embedding became the stage-3 choice. |
| `speech_stage3_emotion_finetuned_holdout.csv` | Speaker-holdout result for the emotion-finetuned Wav2Vec2 model. |
| `speech_stage3_emotion_finetuned_*_classification_report.csv` | Per-emotion reports for the emotion-finetuned stage. |
| `speech_stage4_emotion2vec_champion.csv` | Final champion result: Emotion2Vec+ under strict speaker holdout. |
| `speech_stage4_emotion2vec_champion_*_classification_report.csv` | Per-emotion reports for the final champion model. |
| `speech_stage4_emotion2vec_champion_*_confusion_matrix.csv` | Confusion matrices for the final champion model. |
| `speech_stage4_emotion2vec_champion_predictions.csv` | Per-file predictions from the final champion; useful for failure-case analysis. |
| `text_speaker_holdout_accuracy.csv` | Final text-only result under speaker holdout. |
| `fusion_speaker_holdout_accuracy.csv` | Final fusion result under speaker holdout. |
| `fusion_speaker_holdout_*_classification_report.csv` | Per-emotion reports for the fusion model. |

## How To Read The Common Files

- `accuracy` tables answer: **how often was the model correct overall?**
- `classification_report` tables answer: **which emotions were easy or hard?**
- `confusion_matrix` tables answer: **which labels were mistaken for which other labels?**
- `predictions` tables answer: **which exact files were right or wrong?**

Everything else from intermediate experiments is kept in `../archive/tables/`.
