# Archive Plots Guide (Experimental Outputs)

This folder contains **historical confusion matrices** from experimental runs. These are kept for traceability. Each plot name matches a corresponding table in `Results/archive/tables/`.

## Files In This Folder

| File | What it shows | Paired tables (archive) |
| --- | --- | --- |
| `speech_only_confusion_matrix.png` | Confusion matrix from an early speech-only baseline run. | `speech_only_accuracy.csv`, `speech_only_classification_report.csv` |
| `speech_only_test_confusion_matrix.png` | Confusion matrix from the speech-only test split evaluation. | `speech_only_test_accuracy.csv`, `speech_only_test_classification_report.csv`, `speech_only_test_split.csv` |
| `text_only_confusion_matrix.png` | Confusion matrix from an early text-only baseline run. | `text_only_accuracy.csv`, `text_only_classification_report.csv` |
| `text_only_test_confusion_matrix.png` | Confusion matrix from the text-only test split evaluation. | `text_only_test_accuracy.csv`, `text_only_test_classification_report.csv`, `text_only_test_split.csv` |
| `fusion_confusion_matrix.png` | Confusion matrix from an early fusion baseline run. | `fusion_accuracy.csv`, `fusion_classification_report.csv` |
| `fusion_test_confusion_matrix.png` | Confusion matrix from the fusion test split evaluation. | `fusion_test_accuracy.csv`, `fusion_test_classification_report.csv`, `fusion_test_split.csv` |
| `fusion_full_eval_confusion_matrix.png` | Confusion matrix from a full re-evaluation pass of the fusion baseline. | `fusion_full_eval_accuracy.csv`, `fusion_full_eval_classification_report.csv` |
| `speech_enhanced_aug_svm_confusion_matrix.png` | Confusion matrix for the enhanced feature + augmentation SVM experiment. | `speech_enhanced_aug_svm_accuracy.csv`, `speech_enhanced_aug_svm_classification_report.csv` |
| `speech_enhanced_aug_svm_test_confusion_matrix.png` | Confusion matrix from the test split of the enhanced augmentation experiment. | `speech_enhanced_aug_svm_test_accuracy.csv`, `speech_enhanced_aug_svm_test_classification_report.csv`, `speech_enhanced_aug_svm_test_split.csv` |

## Reading 

- Rows are the **true emotions**, columns are the **predicted emotions**.
- The diagonal shows correct predictions; off-diagonals show confusions.
- These plots are useful for understanding failure patterns in earlier baselines.

If you only need final results, use the plots in `Results/plots/` instead.