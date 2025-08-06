"""
PHASE 1A: Baseline Random Forest Classifier for 5G PFCP Dataset
Target: Achieve ≥94% macro-F1 score on test set

Author: AI Agent for Adversarial 5G IDS Project
Date: August 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime
import os
import sys

# Add src to path for imports
sys.path.append('src')
from models.utils import (
    load_processed_data, 
    preprocess_features, 
    save_model_metadata, 
    create_results_report,
    print_performance_summary
)

def main():
    """
    Main function to train baseline Random Forest classifier
    """
    print("🚀 PHASE 1A: BASELINE RANDOM FOREST TRAINING")
    print("="*60)
    
    # Step 1: Load processed data
    try:
        X_train, X_test, y_train, y_test = load_processed_data()
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return 1
    
    # Step 2: Preprocess features
    try:
        X_train_scaled, X_test_scaled = preprocess_features(X_train, X_test)
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return 1
    
    # Step 3: Define Random Forest configuration
    rf_config = {
        'n_estimators': 200,        # Higher for stability
        'max_depth': 15,           # Prevent overfitting
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'class_weight': 'balanced', # Handle class imbalance
        'n_jobs': -1
    }
    
    print(f"🔧 Random Forest Configuration:")
    for key, value in rf_config.items():
        print(f"   {key}: {value}")
    
    # Step 4: Initialize Random Forest model
    rf_model = RandomForestClassifier(**rf_config)
    
    # Step 5: 5-Fold Cross-Validation
    print("\n🔄 Performing 5-fold stratified cross-validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        rf_model, X_train_scaled, y_train, 
        cv=cv, scoring='f1_macro', n_jobs=-1
    )
    
    print(f"Cross-validation macro-F1 scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.3f}")
    
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"✅ 5-fold CV macro-F1: {cv_mean:.3f} ± {cv_std:.3f}")
    
    # Step 6: Train final model on full training set
    print("\n🔄 Training final model on full training set...")
    rf_model.fit(X_train_scaled, y_train)
    
    # Step 7: Evaluate on test set
    print("🔄 Evaluating on test set...")
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate comprehensive metrics
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, average='macro')
    test_recall = recall_score(y_test, y_pred, average='macro')
    
    # Generate detailed reports
    class_report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    test_metrics = {
        'macro_f1': test_macro_f1,
        'accuracy': test_accuracy,
        'macro_precision': test_precision,
        'macro_recall': test_recall,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }
    
    # Step 8: Save trained model
    model_path = "models/rf_baseline.joblib"
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, model_path)
    print(f"✅ Model saved: {model_path}")
    
    # Step 9: Save metadata
    save_model_metadata(rf_model, cv_scores, test_macro_f1, rf_config)
    
    # Step 10: Create results report
    create_results_report(cv_scores, test_metrics, model_path)
    
    # Step 11: Print performance summary and validate success
    success = print_performance_summary(cv_scores, test_metrics)
    
    # Additional detailed output
    print(f"\nDetailed Test Set Results:")
    print(f"Macro-F1: {test_macro_f1:.3f}")
    print(f"Accuracy: {test_accuracy:.3f}")
    print(f"Macro-Precision: {test_precision:.3f}")
    print(f"Macro-Recall: {test_recall:.3f}")
    
    print(f"\nClassification Report:")
    print(class_report)
    
    print(f"\nConfusion Matrix:")
    print(conf_matrix)
    
    # Feature importance analysis
    feature_importance = rf_model.feature_importances_
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]
    
    print(f"\nTop 10 Most Important Features:")
    for i, idx in enumerate(top_features_idx, 1):
        print(f"   {i:2d}. Feature {idx:2d}: {feature_importance[idx]:.4f}")
    
    # Final success validation
    if success:
        print(f"\n🎉 SUCCESS: Baseline Random Forest achieved {test_macro_f1:.1%} macro-F1")
        print("✅ BASELINE ACHIEVED - Ready for Phase 1B")
        return 0
    else:
        print(f"\n❌ FAILURE: Baseline Random Forest achieved only {test_macro_f1:.1%} macro-F1")
        print("❌ BASELINE FAILED - Hyperparameter tuning required")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
