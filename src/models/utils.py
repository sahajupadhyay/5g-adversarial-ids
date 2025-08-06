"""
Utility functions for baseline model training
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import json
import os
from datetime import datetime

def load_processed_data(data_dir="data/processed/"):
    """
    Load preprocessed 5G PFCP dataset
    
    Returns:
        tuple: X_train, X_test, y_train, y_test as numpy arrays
    """
    print("Loading 5G PFCP dataset...")
    
    # Load data files
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Verify expected shapes
    expected_train_shape = (1113, 43)
    expected_test_shape = (477, 43)
    
    assert X_train.shape == expected_train_shape, f"Expected X_train shape {expected_train_shape}, got {X_train.shape}"
    assert X_test.shape == expected_test_shape, f"Expected X_test shape {expected_test_shape}, got {X_test.shape}"
    
    print(f"✅ Data loaded: X_train{X_train.shape}, X_test{X_test.shape}")
    
    # Print data statistics
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes in training: {len(np.unique(y_train))}")
    print(f"Class distribution (training): {np.bincount(y_train)}")
    print(f"Feature ranges - min: {X_train.min():.3f}, max: {X_train.max():.3f}")
    
    return X_train, X_test, y_train, y_test

def preprocess_features(X_train, X_test, save_scaler=True, scaler_path="models/scaler.joblib"):
    """
    Standardize features using StandardScaler
    
    Args:
        X_train: Training features
        X_test: Test features
        save_scaler: Whether to save the fitted scaler
        scaler_path: Path to save the scaler
        
    Returns:
        tuple: Scaled X_train, X_test
    """
    print("🔄 Preprocessing features...")
    
    # Initialize and fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for future use
    if save_scaler:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")
    
    print(f"✅ Preprocessing complete")
    print(f"Scaled feature ranges - train min: {X_train_scaled.min():.3f}, max: {X_train_scaled.max():.3f}")
    
    return X_train_scaled, X_test_scaled

def save_model_metadata(model, cv_scores, test_score, config, metadata_path="models/rf_baseline_metadata.json"):
    """
    Save model training metadata
    
    Args:
        model: Trained model
        cv_scores: Cross-validation scores
        test_score: Test set performance
        config: Model configuration
        metadata_path: Path to save metadata
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier',
        'model_parameters': config,
        'cross_validation': {
            'n_folds': len(cv_scores),
            'macro_f1_mean': float(np.mean(cv_scores)),
            'macro_f1_std': float(np.std(cv_scores)),
            'individual_scores': [float(score) for score in cv_scores]
        },
        'test_performance': {
            'macro_f1': float(test_score)
        },
        'training_data': {
            'n_samples': 1113,
            'n_features': 43,
            'n_classes': 5
        }
    }
    
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {metadata_path}")

def create_results_report(cv_scores, test_metrics, model_path, report_path="reports/baseline_scores.md"):
    """
    Create markdown report with baseline results
    
    Args:
        cv_scores: Cross-validation scores
        test_metrics: Test set metrics dictionary
        model_path: Path to saved model
        report_path: Path to save report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""# Baseline Random Forest Results

## Training Summary
- **Date:** {timestamp}
- **Model:** Random Forest Classifier
- **Dataset:** 5G PFCP SANCUS Dataset

## Cross-Validation Results
- **Cross-validation macro-F1:** {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}
- **Individual fold scores:** {[f"{score:.3f}" for score in cv_scores]}

## Test Set Performance
- **Test set macro-F1:** {test_metrics['macro_f1']:.3f}
- **Test set accuracy:** {test_metrics['accuracy']:.3f}
- **Test set precision (macro):** {test_metrics['macro_precision']:.3f}
- **Test set recall (macro):** {test_metrics['macro_recall']:.3f}

## Model Files
- **Model file:** {model_path}
- **Scaler file:** models/scaler.joblib
- **Metadata file:** models/rf_baseline_metadata.json

## Status
- **Baseline Target (≥94% macro-F1):** {'✅ ACHIEVED' if test_metrics['macro_f1'] >= 0.94 else '❌ FAILED'}

## Per-Class Performance
```
{test_metrics['classification_report']}
```

## Confusion Matrix
```
{test_metrics['confusion_matrix']}
```
"""
    
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"✅ Results report saved: {report_path}")

def print_performance_summary(cv_scores, test_metrics):
    """
    Print formatted performance summary
    
    Args:
        cv_scores: Cross-validation scores
        test_metrics: Test metrics dictionary
    """
    print("\n" + "="*60)
    print("BASELINE RANDOM FOREST PERFORMANCE SUMMARY")
    print("="*60)
    print(f"✅ 5-fold CV macro-F1: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    print(f"✅ Test macro-F1: {test_metrics['macro_f1']:.3f}")
    print(f"✅ Test accuracy: {test_metrics['accuracy']:.3f}")
    
    # Success validation
    if test_metrics['macro_f1'] >= 0.94:
        print("✅ BASELINE ACHIEVED - Phase 1 complete")
        return True
    else:
        print("❌ BASELINE FAILED - Macro-F1 < 94%")
        return False
