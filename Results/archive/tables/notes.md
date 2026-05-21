# Archive Tables Index (Supplementary Outputs)

This folder contains historical and supplementary experiment outputs that are not part of the final reviewer-facing Results. The list below explains every file in this folder.

## 1) Overall Comparisons and Tracking

- `model_comparison.csv` — consolidated comparison table used during early model selection.
- `text_baseline_comparison.csv` — side-by-side baseline summary focused on text-only variants.
- `random_split_baselines.csv` — historical random-split baseline summary (speech/text/fusion).
- `speaker_holdout_progress.csv` — chronological log of speaker-holdout accuracy across iterations.
- `speaker_holdout_experiments.csv` — raw results of multiple speaker-holdout trials.
- `speaker_holdout_experiments_summary.csv` — condensed statistics from the above experiments.
- `speaker_holdout_augmented.csv` — results after applying augmentation or extra features.
- `overfitting_assessment.md` — narrative note about random-split vs speaker-holdout generalization.
- `upgrade_decision.md` — rationale for choosing feature + augmentation upgrades.
- `eval_random_and_groupkfold.log` — log output from random split and GroupKFold evaluation runs.
- `notes.md` — this index file.

## 2) Stage 1: MFCC Baseline (Speaker-Holdout)

- `speech_stage1_mfcc_baseline.csv` — OAF->YAF and YAF->OAF accuracies for MFCC baseline.
- `speech_stage1_mfcc_baseline_OAF_to_YAF_classification_report.csv` — per-emotion report (OAF->YAF).
- `speech_stage1_mfcc_baseline_YAF_to_OAF_classification_report.csv` — per-emotion report (YAF->OAF).

## 3) Stage 2: Wav2Vec2 Baselines and Adaptation

- `speech_stage2_wav2vec2_baseline.csv` — speaker-holdout accuracy using wav2vec2-base embeddings.
- `speech_stage2_wav2vec2_OAF_to_YAF_classification_report.csv` — per-emotion report (OAF->YAF).
- `speech_stage2_wav2vec2_YAF_to_OAF_classification_report.csv` — per-emotion report (YAF->OAF).
- `speech_stage2_wav2vec2_details.csv` — detailed results across normalization/adaptation variants.
- `speech_stage2_wav2vec2_adaptation.csv` — summary table of the best adaptation variants.
- `speech_stage2_wav2vec2_domain_adaptation.csv` — older domain-adaptation summary table.
- `speech_stage2_wav2vec2_domain_adaptation_summary.csv` — compact summary of the above file.
- `pretrained_speaker_holdout_predictions.csv` — per-file predictions for the wav2vec2-base holdout run.
- `pretrained_OAF_to_YAF_confusion_matrix.csv` — confusion matrix (OAF->YAF).
- `pretrained_YAF_to_OAF_confusion_matrix.csv` — confusion matrix (YAF->OAF).

## 4) Stage 3: Emotion-Finetuned Wav2Vec2

- `speech_stage3_emotion_finetuned.csv` — summary table of emotion-finetuned speaker-holdout results.
- `speech_stage3_emotion_finetuned_holdout.csv` — the underlying holdout table (same rows as above).
- `speech_stage3_emotion_finetuned_OAF_to_YAF_classification_report.csv` — per-emotion report (OAF->YAF).
- `speech_stage3_emotion_finetuned_YAF_to_OAF_classification_report.csv` — per-emotion report (YAF->OAF).

## 5) Stage 4: Emotion2Vec+ Champion (Historical Copy)

- `speech_stage4_emotion2vec_champion.csv` — speaker-holdout accuracy for Emotion2Vec+.
- `speech_stage4_emotion2vec_champion_OAF_to_YAF_classification_report.csv` — per-emotion report (OAF->YAF).
- `speech_stage4_emotion2vec_champion_YAF_to_OAF_classification_report.csv` — per-emotion report (YAF->OAF).
- `speech_stage4_emotion2vec_champion_OAF_to_YAF_confusion_matrix.csv` — confusion matrix (OAF->YAF).
- `speech_stage4_emotion2vec_champion_YAF_to_OAF_confusion_matrix.csv` — confusion matrix (YAF->OAF).
- `speech_stage4_emotion2vec_champion_predictions.csv` — per-file predictions for error analysis.

## 6) Legacy Speech-Only Random-Split Baselines

- `speech_only_accuracy.csv` — overall accuracy for early random-split speech-only baseline.
- `speech_only_classification_report.csv` — per-emotion report for that baseline.
- `speech_only_test_accuracy.csv` — accuracy from a separate test run.
- `speech_only_test_classification_report.csv` — per-emotion report from the test run.
- `speech_only_test_split.csv` — test split metadata.

## 7) Legacy Text-Only Baselines

- `text_only_accuracy.csv` — overall accuracy for early random-split text-only baseline.
- `text_only_classification_report.csv` — per-emotion report for that baseline.
- `text_only_test_accuracy.csv` — accuracy from a separate test run.
- `text_only_test_classification_report.csv` — per-emotion report from the test run.
- `text_only_test_split.csv` — test split metadata.
- `text_speaker_holdout_accuracy.csv` — speaker-holdout accuracy for the text-only model.
- `text_only_assessment.md` — narrative note explaining why text-only underperforms on TESS.

## 8) Legacy Fusion Baselines

- `fusion_accuracy.csv` — overall accuracy for early random-split fusion baseline.
- `fusion_classification_report.csv` — per-emotion report for that baseline.
- `fusion_full_eval_accuracy.csv` — accuracy from a full re-evaluation pass.
- `fusion_full_eval_classification_report.csv` — per-emotion report from that pass.
- `fusion_test_accuracy.csv` — accuracy from a separate test run.
- `fusion_test_classification_report.csv` — per-emotion report from the test run.
- `fusion_test_split.csv` — test split metadata.

## 9) Wav2Vec2 Base Diagnostics

- `wav2vec2_base_groupkfold_metrics.csv` — GroupKFold metrics for wav2vec2-base embeddings.
- `wav2vec2_base_random80_20_metrics.csv` — random 80/20 split metrics for wav2vec2-base.
- `wav2vec2_base_random80_20_confusion_matrix.csv` — confusion matrix for the random 80/20 split.
- `wav2vec2_domain_adaptation.csv` — early domain adaptation results (pre-emotion fine-tuning).
- `wav2vec2_improvement_notes.md` — narrative notes about wav2vec2 improvements.

## 10) Wav2Vec2 Large and Robust Variants

- `wav2vec2-large-960h_speaker_holdout.csv` — speaker-holdout accuracy for wav2vec2-large-960h.
- `wav2vec2-large-960h_speaker_holdout_predictions.csv` — per-file predictions for that model.
- `wav2vec2-large-960h_OAF_to_YAF_classification_report.csv` — per-emotion report (OAF->YAF).
- `wav2vec2-large-960h_YAF_to_OAF_classification_report.csv` — per-emotion report (YAF->OAF).
- `wav2vec2-large-960h_OAF_to_YAF_confusion_matrix.csv` — confusion matrix (OAF->YAF).
- `wav2vec2-large-960h_YAF_to_OAF_confusion_matrix.csv` — confusion matrix (YAF->OAF).

- `wav2vec2-large-robust-12-ft-emotion-msp-dim_speaker_holdout_predictions.csv` — predictions for the robust emotion-tuned variant.
- `wav2vec2-large-robust-12-ft-emotion-msp-dim_OAF_to_YAF_confusion_matrix.csv` — confusion matrix (OAF->YAF).
- `wav2vec2-large-robust-12-ft-emotion-msp-dim_YAF_to_OAF_confusion_matrix.csv` — confusion matrix (YAF->OAF).

## 11) Emotion Fine-Tuning and Feature Experiments

- `emotion_ft_feature_fusion.csv` — results from feature-level fusion experiments.
- `emotion_ft_pooling_compare.csv` — comparison of pooling strategies.
- `emotion_ft_pooling_compare_summary.csv` — compact summary of pooling comparison.
- `emotion_ft_pseudolabel_adaptation.csv` — results from pseudo-label adaptation experiments.
- `emotion_ft_pseudolabel_adaptation_summary.csv` — compact summary of pseudo-label experiments.
- `emotion_ft_svm_sweep.csv` — SVM hyperparameter sweep results.
- `emotion_ft_svm_sweep_summary.csv` — compact summary of the SVM sweep.
- `emotion_ft_wav2vec2_domain_adaptation.csv` — emotion finetuned domain adaptation results.
- `emotion_ft_wav2vec2_domain_adaptation_summary.csv` — compact summary of the above file.
- `emotion_ft_improvement_notes.md` — narrative notes about emotion fine-tuning gains.
- `emotion2vec_improvement_notes.md` — narrative notes about the final Emotion2Vec+ gains.

## 12) Ensemble Experiments

- `ensemble_wav2vec2_speaker_holdout.csv` — speaker-holdout accuracy for wav2vec2 ensembles.
- `ensemble_wav2vec2_speaker_holdout_predictions.csv` — per-file predictions for the ensemble.
- `ensemble_wav2vec2_OAF_to_YAF_classification_report.csv` — per-emotion report (OAF->YAF).
- `ensemble_wav2vec2_YAF_to_OAF_classification_report.csv` — per-emotion report (YAF->OAF).
- `ensemble_wav2vec2_OAF_to_YAF_confusion_matrix.csv` — confusion matrix (OAF->YAF).
- `ensemble_wav2vec2_YAF_to_OAF_confusion_matrix.csv` — confusion matrix (YAF->OAF).

## 13) Augmented SVM Experiments

- `speech_enhanced_aug_svm_accuracy.csv` — accuracy for the augmented SVM baseline.
- `speech_enhanced_aug_svm_classification_report.csv` — per-emotion report for that model.
- `speech_enhanced_aug_svm_test_accuracy.csv` — accuracy from a test-time rerun.
- `speech_enhanced_aug_svm_test_classification_report.csv` — per-emotion report from the test run.
- `speech_enhanced_aug_svm_test_split.csv` — test split metadata for those runs.

## 14) Plot Metadata

- `representation_plots.csv` — metadata used to generate representation visualization plots.
