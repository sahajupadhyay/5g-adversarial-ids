# Baseline Random Forest Results

## Training Summary
- **Date:** 2025-08-05 17:16:52
- **Model:** Random Forest Classifier
- **Dataset:** 5G PFCP SANCUS Dataset

## Cross-Validation Results
- **Cross-validation macro-F1:** 0.643 ± 0.023
- **Individual fold scores:** ['0.621', '0.626', '0.687', '0.639', '0.644']

## Test Set Performance
- **Test set macro-F1:** 0.642
- **Test set accuracy:** 0.650
- **Test set precision (macro):** 0.646
- **Test set recall (macro):** 0.648

## Model Files
- **Model file:** models/rf_baseline.joblib
- **Scaler file:** models/scaler.joblib
- **Metadata file:** models/rf_baseline_metadata.json

## Status
- **Baseline Target (≥94% macro-F1):** ❌ FAILED

## Per-Class Performance
```
              precision    recall  f1-score   support

           0       0.45      0.61      0.52        95
           1       1.00      0.98      0.99        96
           2       0.40      0.40      0.40        95
           3       0.38      0.25      0.30        95
           4       1.00      1.00      1.00        96

    accuracy                           0.65       477
   macro avg       0.65      0.65      0.64       477
weighted avg       0.65      0.65      0.64       477

```

## Confusion Matrix
```
[[58  0 24 13  0]
 [ 1 94  0  1  0]
 [31  0 38 26  0]
 [39  0 32 24  0]
 [ 0  0  0  0 96]]
```
