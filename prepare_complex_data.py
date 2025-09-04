#!/usr/bin/env python3
"""
Prepare Complex Dataset for 5G IDS System Testing
Convert our complex dataset into the format expected by the IDS system
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

def prepare_complex_dataset():
    """Prepare complex dataset for IDS system testing"""
    
    print("🔄 Preparing complex dataset for 5G IDS system...")
    
    # Load our complex dataset
    df = pd.read_csv('complex_5g_dataset/full_complex_dataset.csv')
    print(f"📊 Loaded complex dataset: {df.shape}")
    
    # Separate features and labels
    feature_cols = [col for col in df.columns if col not in ['attack_type', 'label']]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"📊 Features shape: {X.shape}")
    print(f"📊 Labels shape: {y.shape}")
    print(f"📊 Unique labels: {np.unique(y)}")
    print(f"📊 Label distribution: {np.bincount(y)}")
    
    # Map labels to attack type names for consistency
    attack_types = df['attack_type'].unique()
    print(f"📊 Attack types: {list(attack_types)}")
    
    # Create label mapping
    label_mapping = {}
    for label_val in np.unique(y):
        # Find the most common attack type for this label
        attack_type = df[df['label'] == label_val]['attack_type'].mode()[0]
        label_mapping[label_val] = attack_type
    
    print(f"📊 Label mapping: {label_mapping}")
    
    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train set: {X_train.shape}, {y_train.shape}")
    print(f"📊 Test set: {X_test.shape}, {y_test.shape}")
    
    # Save to data/processed directory
    np.save('data/processed/X_train.npy', X_train.astype(np.float32))
    np.save('data/processed/X_test.npy', X_test.astype(np.float32))
    np.save('data/processed/y_train.npy', y_train.astype(np.int32))
    np.save('data/processed/y_test.npy', y_test.astype(np.int32))
    
    # Create metadata
    unique_labels = np.unique(y)
    class_names = [label_mapping[label] for label in sorted(unique_labels)]
    
    metadata = {
        "classes": class_names,
        "n_features": X.shape[1],
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "label_mapping": {int(k): v for k, v in label_mapping.items()},
        "attack_distribution": {
            "train": {label_mapping[label]: int(count) for label, count in zip(*np.unique(y_train, return_counts=True))},
            "test": {label_mapping[label]: int(count) for label, count in zip(*np.unique(y_test, return_counts=True))}
        }
    }
    
    # Save metadata
    with open('data/processed/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Complex dataset prepared for IDS system!")
    print(f"✅ Saved X_train.npy: {X_train.shape}")
    print(f"✅ Saved X_test.npy: {X_test.shape}")
    print(f"✅ Saved y_train.npy: {y_train.shape}")
    print(f"✅ Saved y_test.npy: {y_test.shape}")
    print(f"✅ Saved metadata.json with {len(class_names)} classes")
    
    return metadata

if __name__ == "__main__":
    metadata = prepare_complex_dataset()
    print("\n🎯 Ready to test the complete 5G IDS system!")
    print(f"Classes: {metadata['classes']}")
    print(f"Features: {metadata['n_features']}")
    print(f"Train samples: {metadata['train_size']}")
    print(f"Test samples: {metadata['test_size']}")
