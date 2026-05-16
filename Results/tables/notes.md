# Baseline Results Notes

The first baseline run used 2800 unique TESS audio files after ignoring a duplicated nested extraction.

| Model | Accuracy |
| --- | ---: |
| Speech-only | 99.82% |
| Text-only | 0.00% |
| Fusion | 99.82% |

The text-only result is expected to be weak for TESS because the transcript is usually just an isolated word such as `back`, `bar`, or `base`. The same words are spoken in all emotion classes, so the text itself does not contain emotion cues. The speech-only model performs very well because TESS emotions are expressed through acoustic patterns.

Fusion does not improve over speech-only in this baseline because the text feature adds little useful information for this dataset.

The text-only result should be interpreted as a required negative baseline, not as a broken evaluation. The text-only model produces predictions, but the transcript-only input does not contain enough information to identify emotion labels in TESS.

The high speech-only accuracy is valid for the random TESS split, but it should not be oversold. A speaker-holdout check drops to about 48-51% accuracy, meaning the baseline is sensitive to speaker/style differences.

A follow-up speaker-holdout sweep with SVM, tree models, and added energy/spectral prosody features improved the best average speaker-holdout score to about 53.93%, which is still far below the random-split result.
