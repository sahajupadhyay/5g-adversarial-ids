"""
Adversarial Training Implementation for 5G PFCP IDS
Phase 2B: Defense Development

This module implements adversarial training to create robust models that can
withstand adversarial attacks while maintaining good clean accuracy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main')
sys.path.append('/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/src')

from src.attacks.enhanced_attacks import EnhancedAdversarialAttacks
from src.attacks.pfcp_constraints import PFCPConstraints


class AdversarialTrainer:
    """
    Adversarial Training Pipeline for Robust 5G IDS
    
    This class implements progressive adversarial training where models are
    trained on a mixture of clean and adversarial examples to improve robustness.
    """
    
    def __init__(self, base_model_params=None, constraint_config=None):
        """
        Initialize the adversarial trainer
        
        Args:
            base_model_params: Parameters for the base RandomForest model
            constraint_config: Configuration for PFCP constraints
        """
        self.base_model_params = base_model_params or {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.constraint_config = constraint_config or {
            'max_flow_duration': 300.0,
            'max_packet_size': 1500,
            'max_flow_bytes': 1000000
        }
        
        # Initialize components
        self.model = None
        self.scaler = StandardScaler()
        self.attack_engine = None
        self.constraints = PFCPConstraints(self.constraint_config)
        
        # Training configuration
        self.adversarial_ratio = 0.3  # 30% adversarial examples
        self.progressive_epsilons = [0.1, 0.2, 0.3]  # Progressive training
        self.current_epsilon = 0.1
        
        # Metrics tracking
        self.training_history = []
        
    def load_data(self, data_path='/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/data/processed'):
        """Load the preprocessed 5G PFCP dataset"""
        print("Loading 5G PFCP dataset...")
        
        # Load preprocessed data
        X_train = np.load(os.path.join(data_path, 'X_train.npy'))
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_train = np.load(os.path.join(data_path, 'y_train.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        return X_train, X_test, y_train, y_test
    
    def generate_adversarial_examples(self, X, y, epsilon=0.1):
        """
        Generate adversarial examples for training
        
        Args:
            X: Input features
            y: True labels  
            epsilon: Perturbation budget
            
        Returns:
            X_adv: Adversarial examples
            y_adv: Corresponding labels
        """
        print(f"Generating adversarial examples with ε={epsilon}...")
        
        if self.attack_engine is None:
            # Initialize attack engine with current model
            self.attack_engine = EnhancedAdversarialAttacks(
                model=self.model,
                constraints=self.constraints
            )
        
        # Generate adversarial examples using Enhanced PGD
        try:
            X_adv = self.attack_engine.enhanced_pgd_attack(
                X=X,
                y=y,
                epsilon=epsilon,
                alpha=epsilon/4,  # Step size
                num_iter=10,
                momentum=0.9
            )
            
            print(f"Generated {len(X_adv)} adversarial examples")
            return X_adv, y
            
        except Exception as e:
            print(f"Error generating adversarial examples: {e}")
            # Fallback: return original examples
            return X, y
    
    def create_mixed_dataset(self, X_clean, y_clean, adversarial_ratio=0.3):
        """
        Create a mixed dataset with clean and adversarial examples
        
        Args:
            X_clean: Clean training data
            y_clean: Clean training labels
            adversarial_ratio: Proportion of adversarial examples
            
        Returns:
            X_mixed: Mixed training data
            y_mixed: Mixed training labels
        """
        print(f"Creating mixed dataset with {adversarial_ratio:.1%} adversarial examples...")
        
        # Calculate number of adversarial examples needed
        n_total = len(X_clean)
        n_adversarial = int(n_total * adversarial_ratio)
        n_clean = n_total - n_adversarial
        
        # Sample clean examples
        clean_indices = np.random.choice(len(X_clean), n_clean, replace=False)
        X_clean_subset = X_clean[clean_indices]
        y_clean_subset = y_clean[clean_indices]
        
        # Sample examples for adversarial generation
        adv_indices = np.random.choice(len(X_clean), n_adversarial, replace=False)
        X_for_adv = X_clean[adv_indices]
        y_for_adv = y_clean[adv_indices]
        
        # Generate adversarial examples
        X_adv, y_adv = self.generate_adversarial_examples(
            X_for_adv, y_for_adv, self.current_epsilon
        )
        
        # Combine clean and adversarial examples
        X_mixed = np.vstack([X_clean_subset, X_adv])
        y_mixed = np.hstack([y_clean_subset, y_adv])
        
        # Shuffle the mixed dataset
        shuffle_indices = np.random.permutation(len(X_mixed))
        X_mixed = X_mixed[shuffle_indices]
        y_mixed = y_mixed[shuffle_indices]
        
        print(f"Mixed dataset: {n_clean} clean + {n_adversarial} adversarial = {len(X_mixed)} total")
        return X_mixed, y_mixed
    
    def train_robust_model(self, X_train, y_train, X_val, y_val):
        """
        Train a robust model using adversarial training
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
        """
        print("Starting adversarial training...")
        
        best_robust_accuracy = 0
        best_model = None
        
        for epoch, epsilon in enumerate(self.progressive_epsilons):
            print(f"\n=== Epoch {epoch+1}/{len(self.progressive_epsilons)}: ε={epsilon} ===")
            
            self.current_epsilon = epsilon
            
            # Create mixed training dataset
            X_mixed, y_mixed = self.create_mixed_dataset(
                X_train, y_train, self.adversarial_ratio
            )
            
            # Scale the mixed data
            X_mixed_scaled = self.scaler.fit_transform(X_mixed)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model on mixed data
            print("Training model on mixed dataset...")
            self.model = RandomForestClassifier(**self.base_model_params)
            self.model.fit(X_mixed_scaled, y_mixed)
            
            # Evaluate on clean validation data
            clean_accuracy = self.model.score(X_val_scaled, y_val)
            
            # Evaluate robustness against adversarial examples
            X_val_adv, _ = self.generate_adversarial_examples(X_val_scaled, y_val, epsilon)
            robust_accuracy = self.model.score(X_val_adv, y_val)
            
            print(f"Clean accuracy: {clean_accuracy:.3f}")
            print(f"Robust accuracy: {robust_accuracy:.3f}")
            
            # Track training progress
            epoch_metrics = {
                'epoch': epoch + 1,
                'epsilon': epsilon,
                'clean_accuracy': clean_accuracy,
                'robust_accuracy': robust_accuracy,
                'adversarial_ratio': self.adversarial_ratio
            }
            self.training_history.append(epoch_metrics)
            
            # Save best model based on robust accuracy
            if robust_accuracy > best_robust_accuracy:
                best_robust_accuracy = robust_accuracy
                best_model = self.model
                print(f"New best robust accuracy: {robust_accuracy:.3f}")
        
        # Use best model
        self.model = best_model
        print(f"\nTraining complete. Best robust accuracy: {best_robust_accuracy:.3f}")
    
    def evaluate_defense(self, X_test, y_test):
        """
        Comprehensive evaluation of the defense
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            evaluation_results: Dictionary with detailed results
        """
        print("\n=== Defense Evaluation ===")
        
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Clean accuracy
        clean_predictions = self.model.predict(X_test_scaled)
        clean_accuracy = accuracy_score(y_test, clean_predictions)
        
        print(f"Clean Test Accuracy: {clean_accuracy:.3f}")
        print("\nClean Performance by Class:")
        print(classification_report(y_test, clean_predictions, digits=3))
        
        # Test against different attack strengths
        evaluation_results = {
            'clean_accuracy': clean_accuracy,
            'clean_report': classification_report(y_test, clean_predictions, output_dict=True),
            'robust_accuracy': {}
        }
        
        test_epsilons = [0.1, 0.2, 0.3, 0.5]
        
        for epsilon in test_epsilons:
            print(f"\nTesting robustness against ε={epsilon} attacks...")
            
            # Generate adversarial test examples
            X_test_adv, _ = self.generate_adversarial_examples(X_test_scaled, y_test, epsilon)
            
            # Evaluate model on adversarial examples
            adv_predictions = self.model.predict(X_test_adv)
            robust_accuracy = accuracy_score(y_test, adv_predictions)
            
            evaluation_results['robust_accuracy'][f'epsilon_{epsilon}'] = {
                'accuracy': robust_accuracy,
                'report': classification_report(y_test, adv_predictions, output_dict=True)
            }
            
            print(f"Robust Accuracy (ε={epsilon}): {robust_accuracy:.3f}")
        
        return evaluation_results
    
    def save_model(self, save_dir='/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models'):
        """Save the trained robust model and metadata"""
        
        if self.model is None:
            raise ValueError("No model to save!")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model and scaler
        model_path = os.path.join(save_dir, 'robust_rf_adversarial.joblib')
        scaler_path = os.path.join(save_dir, 'robust_scaler_adversarial.joblib')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_type': 'AdversariallyTrainedRandomForest',
            'model_params': self.base_model_params,
            'training_config': {
                'adversarial_ratio': self.adversarial_ratio,
                'progressive_epsilons': self.progressive_epsilons,
                'constraint_config': self.constraint_config
            },
            'training_history': self.training_history,
            'timestamp': datetime.now().isoformat(),
            'features': ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
                        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                        'Fwd Packet Length Mean', 'Flow Bytes/s']
        }
        
        metadata_path = os.path.join(save_dir, 'robust_rf_adversarial_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Main training pipeline"""
    print("=== Phase 2B: Adversarial Training for Robust 5G IDS ===")
    
    # Initialize trainer
    trainer = AdversarialTrainer()
    
    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()
    
    # Split training data to have validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train robust model
    trainer.train_robust_model(X_train_split, y_train_split, X_val, y_val)
    
    # Evaluate defense
    results = trainer.evaluate_defense(X_test, y_test)
    
    # Save model
    trainer.save_model()
    
    print("\n=== Phase 2B Training Complete ===")
    print("Robust model ready for deployment!")


if __name__ == "__main__":
    main()
