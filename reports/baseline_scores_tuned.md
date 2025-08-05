# Baseline Random Forest Results

## Training Summary
- **Date:** 2025-08-05 17:18:42
- **Model:** Random Forest Classifier
- **Dataset:** 5G PFCP SANCUS Dataset

## Cross-Validation Results
- **Cross-validation macro-F1:** 0.643 ± 0.022
- **Individual fold scores:** ['0.627', '0.631', '0.680', '0.622', '0.655']

## Test Set Performance
- **Test set macro-F1:** 0.638
- **Test set accuracy:** 0.648
- **Test set precision (macro):** 0.641
- **Test set recall (macro):** 0.646

## Model Files
- **Model file:** models/rf_baseline_tuned.joblib
- **Scaler file:** models/scaler.joblib
- **Metadata file:** models/rf_baseline_metadata.json

## Status
- **Baseline Target (≥94% macro-F1):** ❌ FAILED

## Per-Class Performance
```
              precision    recall  f1-score   support

           0       0.46      0.64      0.53        95
           1       1.00      0.98      0.99        96
           2       0.39      0.38      0.39        95
           3       0.36      0.23      0.28        95
           4       1.00      1.00      1.00        96

    accuracy                           0.65       477
   macro avg       0.64      0.65      0.64       477
weighted avg       0.64      0.65      0.64       477

```

## Confusion Matrix
```
[[61  0 22 12  0]
 [ 1 94  0  1  0]
 [33  0 36 26  0]
 [39  0 34 22  0]
 [ 0  0  0  0 96]]
```
