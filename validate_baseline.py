"""
Model Validation Script - Verify baseline model is ready for Phase 2
"""
import numpy as np
import joblib
from sklearn.metrics import classification_report, f1_score
import json

def validate_baseline_model():
    """
    Validate that the baseline model is properly saved and achieves expected performance
    """
    print("🔍 PHASE 1A MODEL VALIDATION")
    print("="*50)
    
    # Load test data
    print("Loading test data...")
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Load feature transformers
    print("Loading feature engineering pipeline...")
    feature_transformers = joblib.load('models/feature_transformers.joblib')
    scaler = joblib.load('models/scaler_advanced.joblib')
    
    # Load baseline model
    print("Loading baseline model...")
    baseline_model = joblib.load('models/rf_advanced.joblib')
    
    # Apply feature engineering pipeline
    print("Applying feature engineering...")
    variance_selector, mi_selector, pca = feature_transformers
    
    # Transform test data through the same pipeline
    X_test_var = variance_selector.transform(X_test)
    X_test_mi = mi_selector.transform(X_test_var)
    if pca is not None:
        X_test_pca = pca.transform(X_test_mi)
        X_test_scaled = scaler.transform(X_test_pca)
    else:
        X_test_scaled = scaler.transform(X_test_mi)
    
    # Make predictions
    print("Making predictions...")
    y_pred = baseline_model.predict(X_test_scaled)
    
    # Calculate metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n📊 BASELINE MODEL VALIDATION RESULTS:")
    print(f"✅ Macro-F1: {macro_f1:.3f}")
    print(f"✅ Test samples: {len(y_test)}")
    print(f"✅ Feature dimensions: {X_test_scaled.shape}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Validation checks
    expected_f1 = 0.608
    tolerance = 0.01
    
    if abs(macro_f1 - expected_f1) <= tolerance:
        print(f"✅ VALIDATION PASSED: Model performance matches expected ({expected_f1:.3f})")
        
        # Save validation metadata
        validation_metadata = {
            'validation_date': '2025-08-05',
            'model_path': 'models/rf_advanced.joblib',
            'feature_transformers_path': 'models/feature_transformers.joblib',
            'scaler_path': 'models/scaler_advanced.joblib',
            'validated_macro_f1': float(macro_f1),
            'expected_macro_f1': expected_f1,
            'test_samples': int(len(y_test)),
            'feature_dimensions': X_test_scaled.shape[1],
            'ready_for_phase2': True
        }
        
        with open('models/validation_metadata.json', 'w') as f:
            json.dump(validation_metadata, f, indent=2)
        
        print("✅ Validation metadata saved")
        print("\n🚀 READY FOR PHASE 2: ADVERSARIAL ATTACK IMPLEMENTATION")
        return True
    else:
        print(f"❌ VALIDATION FAILED: Expected {expected_f1:.3f}, got {macro_f1:.3f}")
        return False

if __name__ == "__main__":
    success = validate_baseline_model()
    if success:
        print("\n✅ PHASE 1A COMPLETE - BASELINE MODEL VALIDATED")
    else:
        print("\n❌ VALIDATION FAILED - CHECK MODEL FILES")
