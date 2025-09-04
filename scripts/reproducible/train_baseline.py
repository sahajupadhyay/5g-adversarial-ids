#!/usr/bin/env python3
"""
Reproducible training pipeline for 5G Adversarial IDS System
This script ensures deterministic results across runs.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Set all random seeds for reproducibility
RANDOM_SEED = 42

def set_random_seeds(seed=RANDOM_SEED):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def generate_synthetic_5g_data(n_samples=1000, save_path=None):
    """
    Generate synthetic 5G PFCP network data with realistic patterns
    """
    set_random_seeds()
    
    print(f"Generating {n_samples} synthetic 5G network samples...")
    
    # Generate realistic 5G network features
    data = {
        'message_type': np.random.choice([1, 2, 3, 4, 5, 50, 51, 52], n_samples),
        'source_ip_encoded': np.random.randint(0, 1000, n_samples),
        'dest_ip_encoded': np.random.randint(0, 1000, n_samples),
        'packet_size': np.random.normal(800, 300, n_samples).clip(64, 1500),
        'timestamp_delta': np.random.exponential(0.1, n_samples),
        'sequence_number': np.random.randint(1, 65535, n_samples),
        'flow_label': np.random.randint(0, 1048575, n_samples),
        'teid': np.random.randint(1, 100000, n_samples),  # Reduced range to avoid int32 overflow
        'qfi': np.random.randint(0, 63, n_samples),
        'priority': np.random.randint(0, 15, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic attack patterns (20% attack rate)
    attack_prob = 0.2
    labels = np.random.choice([0, 1], n_samples, p=[1-attack_prob, attack_prob])
    
    # Modify features for attack samples to create detectable patterns
    attack_mask = labels == 1
    
    # Attacks tend to have larger packets and different timing
    df.loc[attack_mask, 'packet_size'] *= 1.5
    df.loc[attack_mask, 'timestamp_delta'] *= 0.5
    df.loc[attack_mask, 'sequence_number'] = np.random.randint(60000, 65535, attack_mask.sum())
    
    df['label'] = labels
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Synthetic data saved to {save_path}")
    
    return df

def preprocess_data(df):
    """
    Preprocess the data with consistent transformations
    """
    print("Preprocessing data...")
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Ensure all features are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    return X, y

def train_baseline_model(X_train, y_train, save_dir):
    """
    Train baseline Random Forest model with fixed parameters
    """
    print("Training baseline Random Forest model...")
    
    # Fixed hyperparameters for reproducibility
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        n_jobs=1  # Single threaded for reproducibility
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Save model
    model_path = Path(save_dir) / "rf_baseline_reproducible.joblib"
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "parameters": model.get_params(),
        "training_samples": len(X_train),
        "features": X_train.columns.tolist() if hasattr(X_train, 'columns') else list(range(X_train.shape[1])),
        "random_seed": RANDOM_SEED,
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = Path(save_dir) / "rf_baseline_reproducible_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model saved to {model_path}")
    print(f"Metadata saved to {metadata_path}")
    
    return model

def evaluate_model(model, X_test, y_test, save_dir):
    """
    Evaluate model with consistent metrics
    """
    print("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save results
    results = {
        "accuracy": float(accuracy),
        "classification_report": report,
        "test_samples": len(X_test),
        "random_seed": RANDOM_SEED,
        "timestamp": datetime.now().isoformat()
    }
    
    results_path = Path(save_dir) / "baseline_evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Results saved to {results_path}")
    
    return results

def main():
    """
    Main reproducible training pipeline
    """
    print("=== 5G Adversarial IDS - Reproducible Training Pipeline ===")
    print(f"Random seed: {RANDOM_SEED}")
    
    # Set random seeds
    set_random_seeds()
    
    # Create directories
    data_dir = Path("data/synthetic")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate synthetic data
    data_path = data_dir / "synthetic_5g_data.csv"
    df = generate_synthetic_5g_data(n_samples=2000, save_path=data_path)
    
    # Step 2: Preprocess data
    X, y = preprocess_data(df)
    
    # Step 3: Split data deterministically
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Attack rate in training: {y_train.mean():.3f}")
    print(f"Attack rate in test: {y_test.mean():.3f}")
    
    # Step 4: Train baseline model
    model = train_baseline_model(X_train, y_train, models_dir)
    
    # Step 5: Evaluate model
    results = evaluate_model(model, X_test, y_test, models_dir)
    
    print("=== Training completed successfully ===")
    print(f"Final accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
