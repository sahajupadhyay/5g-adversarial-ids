"""
Robust Feature Analysis for 5G PFCP IDS
Phase 2B: Defense Development

This module analyzes feature robustness under adversarial conditions and
identifies which features are most/least susceptible to manipulation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import sys
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append('/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main')
sys.path.append('/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/src')

from src.attacks.enhanced_attacks import EnhancedAdversarialAttacks
from src.attacks.pfcp_constraints import PFCPConstraints


class RobustFeatureAnalyzer:
    """
    Analyze feature robustness under adversarial conditions
    
    This class helps identify which features are most important for maintaining
    model performance under adversarial attacks.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the feature analyzer
        
        Args:
            model_path: Path to trained model, or None to use baseline
        """
        self.model = None
        self.scaler = None
        self.feature_names = [
            'Flow Duration',
            'Total Fwd Packets', 
            'Total Backward Packets',
            'Total Length of Fwd Packets',
            'Total Length of Bwd Packets', 
            'Fwd Packet Length Mean',
            'Flow Bytes/s'
        ]
        
        # Load model
        if model_path:
            self.load_model(model_path)
        else:
            self.load_baseline_model()
            
        # Initialize attack components
        self.constraints = PFCPConstraints()
        self.attack_engine = None
        
        # Results storage
        self.robustness_results = {}
        
    def load_baseline_model(self):
        """Load the baseline Random Forest model"""
        model_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/rf_advanced.joblib'
        scaler_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/models/scaler_advanced.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print("Loaded baseline model for feature analysis")
        else:
            raise FileNotFoundError("Baseline model not found!")
    
    def load_model(self, model_path):
        """Load a specific model"""
        scaler_path = model_path.replace('.joblib', '_scaler.joblib')
        
        self.model = joblib.load(model_path)
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        print(f"Loaded model from {model_path}")
    
    def load_data(self):
        """Load test data for analysis"""
        data_path = '/Users/sahajupadhyay/Desktop/Capstone/adversarial-5g-ids-main/data/processed'
        
        X_test = np.load(os.path.join(data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(data_path, 'y_test.npy'))
        
        # The processed data is already scaled and feature-engineered, so don't apply scaler again
        print(f"Loaded processed data: {X_test.shape} (already scaled)")
        return X_test, y_test
    
    def analyze_feature_importance_under_attack(self, X_test, y_test, epsilon=0.3):
        """
        Analyze how feature importance changes under adversarial conditions
        
        Args:
            X_test: Test features
            y_test: Test labels
            epsilon: Attack strength
            
        Returns:
            importance_comparison: Dict comparing clean vs adversarial importance
        """
        print("Analyzing feature importance under adversarial conditions...")
        
        # Get clean feature importance
        clean_importance = self.model.feature_importances_
        
        # Initialize attack engine
        if self.attack_engine is None:
            self.attack_engine = EnhancedAdversarialAttacks(
                model=self.model,
                constraints=self.constraints
            )
        
        # Generate adversarial examples
        X_adv = self.attack_engine.enhanced_pgd_attack(
            X=X_test,
            y=y_test,
            epsilon=epsilon,
            alpha=epsilon/4,
            num_iter=10
        )
        
        # Train new model on adversarial data to see importance shift
        temp_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        temp_model.fit(X_adv, y_test)
        adv_importance = temp_model.feature_importances_
        
        # Calculate importance change
        importance_change = adv_importance - clean_importance
        
        results = {
            'clean_importance': clean_importance,
            'adversarial_importance': adv_importance,
            'importance_change': importance_change,
            'feature_names': self.feature_names
        }
        
        return results
    
    def analyze_feature_perturbation_sensitivity(self, X_test, y_test, epsilon=0.3):
        """
        Analyze how sensitive each feature is to perturbations
        
        Args:
            X_test: Test features  
            y_test: Test labels
            epsilon: Perturbation budget
            
        Returns:
            sensitivity_results: Feature sensitivity analysis
        """
        print("Analyzing feature perturbation sensitivity...")
        
        if self.attack_engine is None:
            self.attack_engine = EnhancedAdversarialAttacks(
                model=self.model,
                constraints=self.constraints
            )
        
        # Baseline clean accuracy
        clean_predictions = self.model.predict(X_test)
        clean_accuracy = accuracy_score(y_test, clean_predictions)
        
        feature_sensitivity = {}
        
        # Test perturbation of each feature individually
        for i, feature_name in enumerate(self.feature_names):
            print(f"Testing sensitivity of {feature_name}...")
            
            # Create perturbation mask (only this feature)
            perturbation_mask = np.zeros(len(self.feature_names))
            perturbation_mask[i] = 1.0
            
            # Generate targeted adversarial examples
            X_adv_single = self.attack_engine.enhanced_pgd_attack(
                X=X_test,
                y=y_test, 
                epsilon=epsilon,
                alpha=epsilon/4,
                num_iter=10
            )
            
            # Apply mask to only perturb this feature
            X_adv_masked = X_test.copy()
            X_adv_masked[:, i] = X_adv_single[:, i]
            
            # Measure accuracy drop
            masked_predictions = self.model.predict(X_adv_masked)
            masked_accuracy = accuracy_score(y_test, masked_predictions)
            
            accuracy_drop = clean_accuracy - masked_accuracy
            sensitivity = accuracy_drop / clean_accuracy  # Relative drop
            
            feature_sensitivity[feature_name] = {
                'accuracy_drop': accuracy_drop,
                'relative_sensitivity': sensitivity,
                'perturbed_accuracy': masked_accuracy
            }
        
        return feature_sensitivity
    
    def analyze_constraint_effectiveness(self, X_test, y_test):
        """
        Analyze how well PFCP constraints protect against feature manipulation
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            constraint_analysis: Effectiveness of constraints per feature
        """
        print("Analyzing PFCP constraint effectiveness...")
        
        constraint_protection = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Get constraint bounds for this feature
            if hasattr(self.constraints, 'get_feature_bounds'):
                bounds = self.constraints.get_feature_bounds(i)
            else:
                # Default bounds based on PFCP protocol
                bounds = self._get_default_bounds(feature_name)
            
            # Calculate how much this feature can be perturbed
            feature_values = X_test[:, i]
            max_positive_perturbation = bounds['max'] - feature_values
            max_negative_perturbation = feature_values - bounds['min']
            
            # Calculate constraint protection metric
            avg_protection = np.mean([
                np.mean(np.clip(max_positive_perturbation, 0, np.inf)),
                np.mean(np.clip(max_negative_perturbation, 0, np.inf))
            ])
            
            constraint_protection[feature_name] = {
                'bounds': bounds,
                'avg_protection_range': avg_protection,
                'feature_index': i
            }
        
        return constraint_protection
    
    def _get_default_bounds(self, feature_name):
        """Get default constraint bounds for features"""
        bounds_map = {
            'Flow Duration': {'min': 0.0, 'max': 300.0},
            'Total Fwd Packets': {'min': 1, 'max': 10000},
            'Total Backward Packets': {'min': 0, 'max': 10000},
            'Total Length of Fwd Packets': {'min': 0, 'max': 1000000},
            'Total Length of Bwd Packets': {'min': 0, 'max': 1000000},
            'Fwd Packet Length Mean': {'min': 20, 'max': 1500},
            'Flow Bytes/s': {'min': 0, 'max': 1000000}
        }
        return bounds_map.get(feature_name, {'min': 0, 'max': 1000})
    
    def generate_robustness_report(self, X_test, y_test):
        """
        Generate comprehensive robustness analysis report
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            complete_analysis: Comprehensive robustness report
        """
        print("Generating comprehensive robustness report...")
        
        # Perform all analyses
        importance_analysis = self.analyze_feature_importance_under_attack(X_test, y_test)
        sensitivity_analysis = self.analyze_feature_perturbation_sensitivity(X_test, y_test)
        constraint_analysis = self.analyze_constraint_effectiveness(X_test, y_test)
        
        # Combine results
        complete_analysis = {
            'feature_importance': importance_analysis,
            'perturbation_sensitivity': sensitivity_analysis,
            'constraint_protection': constraint_analysis,
            'feature_names': self.feature_names
        }
        
        # Generate rankings
        complete_analysis['rankings'] = self._generate_feature_rankings(complete_analysis)
        
        return complete_analysis
    
    def _generate_feature_rankings(self, analysis):
        """Generate feature robustness rankings"""
        
        # Extract sensitivity scores
        sensitivity_scores = {
            name: data['relative_sensitivity'] 
            for name, data in analysis['perturbation_sensitivity'].items()
        }
        
        # Extract constraint protection scores
        protection_scores = {
            name: data['avg_protection_range']
            for name, data in analysis['constraint_protection'].items()
        }
        
        # Extract importance stability (lower change = more stable)
        importance_changes = analysis['feature_importance']['importance_change']
        importance_stability = {
            name: abs(change)
            for name, change in zip(self.feature_names, importance_changes)
        }
        
        # Rank features by robustness (lower sensitivity + higher protection = more robust)
        robustness_scores = {}
        for name in self.feature_names:
            # Combine metrics (normalize and weight)
            sensitivity = sensitivity_scores[name]
            protection = protection_scores[name] / 1000  # Normalize
            stability = importance_stability[name]
            
            # Robustness = high protection, low sensitivity, low importance change
            robustness_score = protection - sensitivity - stability
            robustness_scores[name] = robustness_score
        
        # Sort by robustness (higher = more robust)
        sorted_features = sorted(
            robustness_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        rankings = {
            'most_robust': [name for name, _ in sorted_features[:3]],
            'least_robust': [name for name, _ in sorted_features[-3:]],
            'robustness_scores': robustness_scores,
            'sensitivity_ranking': sorted(sensitivity_scores.items(), key=lambda x: x[1]),
            'protection_ranking': sorted(protection_scores.items(), key=lambda x: x[1], reverse=True)
        }
        
        return rankings
    
    def visualize_robustness_analysis(self, analysis, save_dir='reports'):
        """Create visualizations for robustness analysis"""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Feature importance comparison
        plt.figure(figsize=(12, 8))
        
        x_pos = np.arange(len(self.feature_names))
        width = 0.35
        
        plt.subplot(2, 2, 1)
        plt.bar(x_pos - width/2, analysis['feature_importance']['clean_importance'], 
                width, label='Clean', alpha=0.7)
        plt.bar(x_pos + width/2, analysis['feature_importance']['adversarial_importance'], 
                width, label='Adversarial', alpha=0.7)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance: Clean vs Adversarial')
        plt.xticks(x_pos, self.feature_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Sensitivity analysis
        plt.subplot(2, 2, 2)
        sensitivity_values = [
            analysis['perturbation_sensitivity'][name]['relative_sensitivity']
            for name in self.feature_names
        ]
        plt.bar(self.feature_names, sensitivity_values, alpha=0.7, color='red')
        plt.xlabel('Features') 
        plt.ylabel('Sensitivity Score')
        plt.title('Feature Perturbation Sensitivity')
        plt.xticks(rotation=45, ha='right')
        
        # Constraint protection
        plt.subplot(2, 2, 3)
        protection_values = [
            analysis['constraint_protection'][name]['avg_protection_range']
            for name in self.feature_names
        ]
        plt.bar(self.feature_names, protection_values, alpha=0.7, color='green')
        plt.xlabel('Features')
        plt.ylabel('Protection Range')
        plt.title('PFCP Constraint Protection')
        plt.xticks(rotation=45, ha='right')
        
        # Robustness ranking
        plt.subplot(2, 2, 4)
        robustness_scores = list(analysis['rankings']['robustness_scores'].values())
        colors = ['green' if score > 0 else 'red' for score in robustness_scores]
        plt.bar(self.feature_names, robustness_scores, alpha=0.7, color=colors)
        plt.xlabel('Features')
        plt.ylabel('Robustness Score')
        plt.title('Overall Robustness Ranking')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_robustness_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Robustness analysis visualization saved to {save_dir}/feature_robustness_analysis.png")


def main():
    """Main analysis pipeline"""
    print("=== Feature Robustness Analysis ===")
    
    # Initialize analyzer
    analyzer = RobustFeatureAnalyzer()
    
    # Load test data
    X_test, y_test = analyzer.load_data()
    
    # Generate comprehensive analysis
    analysis = analyzer.generate_robustness_report(X_test, y_test)
    
    # Display results
    print("\n=== Robustness Rankings ===")
    print("Most Robust Features:")
    for i, feature in enumerate(analysis['rankings']['most_robust'], 1):
        score = analysis['rankings']['robustness_scores'][feature]
        print(f"  {i}. {feature} (score: {score:.3f})")
    
    print("\nLeast Robust Features:")
    for i, feature in enumerate(analysis['rankings']['least_robust'], 1):
        score = analysis['rankings']['robustness_scores'][feature]
        print(f"  {i}. {feature} (score: {score:.3f})")
    
    # Create visualizations
    analyzer.visualize_robustness_analysis(analysis)
    
    print("\n=== Feature Analysis Complete ===")


if __name__ == "__main__":
    main()
