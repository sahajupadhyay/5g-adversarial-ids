"""
Industrial-Grade Adversarial Attack Implementation
================================================

This module implements robust gradient-based adversarial attacks (FGSM, PGD)
against the baseline IDS model with comprehensive security, monitoring,
and validation features.

Security Classification: CONFIDENTIAL  
Attack Methods: FGSM, PGD with adaptive epsilon strategies
"""

import logging
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from contextlib import contextmanager
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
import torchattacks

# Import our baseline model components
from src.models.baseline_model import BaselineIDS, IDSDataset


class AttackException(Exception):
    """Custom exception for attack-related issues"""
    pass


class SecurityValidationException(Exception):
    """Custom exception for security validation issues"""
    pass


class ModelWrapperException(Exception):
    """Custom exception for model wrapper issues"""
    pass


class ModelWrapper(nn.Module):
    """
    Industrial-grade model wrapper to adapt binary classification model
    for torchattacks library compatibility.
    
    Converts single-output binary model to two-class format required by
    gradient-based attack libraries.
    """
    
    def __init__(self, model: nn.Module, validate_model: bool = True):
        """
        Initialize model wrapper with validation.
        
        Args:
            model: The baseline binary classification model
            validate_model: Whether to validate model compatibility
            
        Raises:
            ModelWrapperException: If model validation fails
        """
        super(ModelWrapper, self).__init__()
        self.logger = logging.getLogger(f"{__class__.__name__}")
        
        if validate_model:
            self._validate_model(model)
        
        self.model = model
        self.prediction_count = 0
        self.attack_success_count = 0
        
        self.logger.info("ModelWrapper initialized successfully")
    
    def _validate_model(self, model: nn.Module) -> None:
        """Validate model compatibility"""
        if model is None:
            raise ModelWrapperException("Model cannot be None")
        
        if not isinstance(model, nn.Module):
            raise ModelWrapperException("Model must be a PyTorch nn.Module")
        
        # Test with dummy input to verify output shape
        try:
            model.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 80)  # Assuming 80 features
                output = model(dummy_input)
                
                if output.shape[-1] != 1:
                    raise ModelWrapperException(f"Expected single output, got shape {output.shape}")
                    
        except Exception as e:
            raise ModelWrapperException(f"Model validation failed: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with binary-to-multiclass conversion.
        
        Args:
            x: Input tensor [batch_size, features]
            
        Returns:
            Two-class logits [batch_size, 2] where:
            - [:, 0] = logits for class 0 (normal)  
            - [:, 1] = logits for class 1 (attack)
        """
        self.prediction_count += x.shape[0]
        
        # Get single logit output [batch_size, 1]
        logit = self.model(x)
        
        # Convert to two-class format [batch_size, 2]
        # For binary classification: [-logit, logit]
        # Positive logit -> class 1, negative logit -> class 0
        two_class_logits = torch.cat([-logit, logit], dim=1)
        
        return two_class_logits
    
    def get_prediction_stats(self) -> Dict[str, int]:
        """Get prediction statistics"""
        return {
            'total_predictions': self.prediction_count,
            'attack_successes': self.attack_success_count,
            'success_rate': (self.attack_success_count / max(1, self.prediction_count))
        }


class AdversarialAttacker:
    """
    Industrial-grade adversarial attack generator with comprehensive
    monitoring, validation, and security features.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: Optional[str] = None,
                 validate_attacks: bool = True):
        """
        Initialize adversarial attacker.
        
        Args:
            model: The trained baseline model
            device: Computing device ('cuda' or 'cpu')
            validate_attacks: Whether to validate attack outputs
        """
        self.logger = logging.getLogger(f"{__class__.__name__}")
        
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model setup with wrapper
        self.base_model = model.to(self.device)
        self.wrapped_model = ModelWrapper(self.base_model, validate_model=True)
        self.validate_attacks = validate_attacks
        
        # Attack statistics
        self.attack_history = []
        self.performance_metrics = {}
        
        # Security parameters
        self.max_epsilon = 1.0  # Maximum allowed perturbation
        self.max_iterations = 100  # Maximum PGD iterations
        
        self.logger.info(f"AdversarialAttacker initialized on {self.device}")
    
    @contextmanager
    def _attack_context(self, attack_name: str):
        """Context manager for attack execution with monitoring"""
        start_time = time.time()
        try:
            self.logger.info(f"Starting {attack_name} attack")
            yield
        except Exception as e:
            self.logger.error(f"{attack_name} attack failed: {e}")
            raise
        finally:
            duration = time.time() - start_time
            self.logger.info(f"{attack_name} attack completed in {duration:.2f}s")
    
    def _validate_epsilon(self, epsilon: float, attack_type: str) -> None:
        """Validate epsilon parameter for security"""
        if not isinstance(epsilon, (int, float)):
            raise SecurityValidationException("Epsilon must be numeric")
        
        if epsilon < 0:
            raise SecurityValidationException("Epsilon cannot be negative")
        
        if epsilon > self.max_epsilon:
            raise SecurityValidationException(
                f"Epsilon {epsilon} exceeds maximum allowed {self.max_epsilon}"
            )
        
        # Attack-specific validation
        if attack_type == 'fgsm' and epsilon > 0.3:
            self.logger.warning(f"Large FGSM epsilon {epsilon} may cause obvious perturbations")
        
        if attack_type == 'pgd' and epsilon > 0.1:
            self.logger.warning(f"Large PGD epsilon {epsilon} may cause obvious perturbations")
    
    def _validate_attack_params(self, attack_config: Dict[str, Any]) -> None:
        """Validate attack configuration parameters"""
        required_keys = ['epsilon']
        for key in required_keys:
            if key not in attack_config:
                raise AttackException(f"Missing required parameter: {key}")
        
        # PGD-specific validation
        if 'steps' in attack_config:
            steps = attack_config['steps']
            if not isinstance(steps, int) or steps <= 0:
                raise AttackException("PGD steps must be positive integer")
            if steps > self.max_iterations:
                raise SecurityValidationException(
                    f"PGD steps {steps} exceeds maximum {self.max_iterations}"
                )
        
        if 'alpha' in attack_config:
            alpha = attack_config['alpha']
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                raise AttackException("PGD alpha must be positive")
    
    def generate_fgsm_attacks(self,
                             data_loader: DataLoader,
                             epsilon: float = 0.05,
                             targeted: bool = False) -> pd.DataFrame:
        """
        Generate Fast Gradient Sign Method (FGSM) adversarial examples.
        
        Args:
            data_loader: DataLoader with test samples
            epsilon: Perturbation magnitude
            targeted: Whether to use targeted attacks
            
        Returns:
            DataFrame with adversarial examples
        """
        with self._attack_context("FGSM"):
            # Validate parameters
            self._validate_epsilon(epsilon, 'fgsm')
            
            # Configure FGSM attack
            if targeted:
                self.logger.warning("Targeted FGSM attacks not recommended for IDS")
            
            attack = torchattacks.FGSM(self.wrapped_model, eps=epsilon)
            
            # Generate adversarial examples
            return self._execute_attack(attack, data_loader, "FGSM", epsilon)
    
    def generate_pgd_attacks(self,
                            data_loader: DataLoader,
                            epsilon: float = 0.05,
                            alpha: float = 0.01,
                            steps: int = 10,
                            random_start: bool = True) -> pd.DataFrame:
        """
        Generate Projected Gradient Descent (PGD) adversarial examples.
        
        Args:
            data_loader: DataLoader with test samples
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            steps: Number of PGD iterations
            random_start: Whether to use random initialization
            
        Returns:
            DataFrame with adversarial examples
        """
        with self._attack_context("PGD"):
            # Validate parameters
            attack_config = {
                'epsilon': epsilon,
                'alpha': alpha,
                'steps': steps
            }
            self._validate_epsilon(epsilon, 'pgd')
            self._validate_attack_params(attack_config)
            
            # Configure PGD attack
            attack = torchattacks.PGD(
                self.wrapped_model,
                eps=epsilon,
                alpha=alpha,
                steps=steps,
                random_start=random_start
            )
            
            # Generate adversarial examples
            return self._execute_attack(attack, data_loader, "PGD", epsilon)
    
    def _execute_attack(self,
                       attack,
                       data_loader: DataLoader,
                       attack_name: str,
                       epsilon: float) -> pd.DataFrame:
        """
        Execute adversarial attack with comprehensive monitoring.
        
        Args:
            attack: Configured torchattacks object
            data_loader: DataLoader with samples to attack
            attack_name: Name of the attack for logging
            epsilon: Perturbation magnitude
            
        Returns:
            DataFrame containing adversarial examples
        """
        self.wrapped_model.eval()
        
        adversarial_samples = []
        original_samples = []
        original_labels = []
        attack_success = []
        perturbation_norms = []
        
        total_samples = 0
        successful_attacks = 0
        
        # Progress tracking
        pbar = tqdm(data_loader, desc=f"Generating {attack_name} attacks")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Convert targets to long for attack library
            targets_long = targets.long()
            
            # Get original predictions for comparison
            with torch.no_grad():
                original_outputs = self.wrapped_model(data)
                original_preds = torch.argmax(original_outputs, dim=1)
            
            # Generate adversarial examples
            try:
                adversarial_data = attack(data, targets_long)
                
                # Validate adversarial examples
                if self.validate_attacks:
                    self._validate_adversarial_samples(data, adversarial_data, epsilon)
                
                # Check attack success
                with torch.no_grad():
                    adv_outputs = self.wrapped_model(adversarial_data)
                    adv_preds = torch.argmax(adv_outputs, dim=1)
                    
                    # Attack succeeds if prediction changes
                    attack_succeeded = (original_preds != adv_preds).float()
                    successful_attacks += attack_succeeded.sum().item()
                    attack_success.extend(attack_succeeded.cpu().numpy())
                
                # Calculate perturbation norms
                perturbations = adversarial_data - data
                l2_norms = torch.norm(perturbations.view(perturbations.shape[0], -1), 
                                    p=2, dim=1)
                perturbation_norms.extend(l2_norms.cpu().numpy())
                
                # Store results
                adversarial_samples.append(adversarial_data.cpu().numpy())
                original_samples.append(data.cpu().numpy())
                original_labels.append(targets.cpu().numpy())
                
                total_samples += data.shape[0]
                
                # Update progress bar
                success_rate = successful_attacks / total_samples if total_samples > 0 else 0
                pbar.set_postfix({
                    'success_rate': f"{success_rate:.3f}",
                    'avg_l2': f"{np.mean(perturbation_norms):.4f}"
                })
                
            except Exception as e:
                self.logger.error(f"Attack failed on batch {batch_idx}: {e}")
                continue
        
        # Compile results
        adversarial_array = np.concatenate(adversarial_samples, axis=0)
        original_array = np.concatenate(original_samples, axis=0)
        labels_array = np.concatenate(original_labels, axis=0)
        
        # Create DataFrame with feature columns
        feature_columns = getattr(data_loader.dataset, 'feature_columns', 
                                [f'feature_{i}' for i in range(adversarial_array.shape[1])])
        
        adversarial_df = pd.DataFrame(adversarial_array, columns=feature_columns)
        adversarial_df['Label'] = labels_array
        
        # Log attack statistics
        final_success_rate = successful_attacks / total_samples
        avg_perturbation = np.mean(perturbation_norms)
        
        self.logger.info(f"{attack_name} Attack Results:")
        self.logger.info(f"  Total samples: {total_samples}")
        self.logger.info(f"  Successful attacks: {successful_attacks}")
        self.logger.info(f"  Success rate: {final_success_rate:.4f}")
        self.logger.info(f"  Average L2 perturbation: {avg_perturbation:.4f}")
        self.logger.info(f"  Epsilon used: {epsilon}")
        
        # Store attack statistics
        attack_stats = {
            'attack_type': attack_name,
            'epsilon': epsilon,
            'total_samples': total_samples,
            'successful_attacks': successful_attacks,
            'success_rate': final_success_rate,
            'average_l2_perturbation': avg_perturbation,
            'timestamp': time.time()
        }
        self.attack_history.append(attack_stats)
        
        return adversarial_df
    
    def _validate_adversarial_samples(self,
                                    original: torch.Tensor,
                                    adversarial: torch.Tensor,
                                    epsilon: float) -> None:
        """
        Validate generated adversarial samples for security and correctness.
        
        Args:
            original: Original input samples
            adversarial: Generated adversarial samples  
            epsilon: Expected perturbation bound
            
        Raises:
            SecurityValidationException: If validation fails
        """
        # Check shapes match
        if original.shape != adversarial.shape:
            raise SecurityValidationException(
                f"Shape mismatch: original {original.shape}, adversarial {adversarial.shape}"
            )
        
        # Check perturbation bounds (L∞ norm)
        perturbation = adversarial - original
        max_perturbation = torch.max(torch.abs(perturbation)).item()
        
        # Allow small numerical tolerance
        tolerance = 1e-6
        if max_perturbation > epsilon + tolerance:
            self.logger.warning(
                f"Perturbation {max_perturbation:.6f} exceeds epsilon {epsilon:.6f}"
            )
        
        # Check for invalid values
        if torch.isnan(adversarial).any() or torch.isinf(adversarial).any():
            raise SecurityValidationException("Adversarial samples contain NaN or Inf values")
        
        # Check for suspicious large values (potential overflow)
        max_value = torch.max(torch.abs(adversarial)).item()
        if max_value > 1000:  # Reasonable bound for normalized features
            self.logger.warning(f"Large adversarial values detected: {max_value}")
    
    def evaluate_attack_effectiveness(self,
                                    original_df: pd.DataFrame,
                                    adversarial_df: pd.DataFrame,
                                    model: nn.Module) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of adversarial attacks.
        
        Args:
            original_df: Original test samples
            adversarial_df: Generated adversarial samples
            model: Model to evaluate against
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()
        
        # Extract features and labels
        feature_cols = [col for col in original_df.columns if col != 'Label']
        
        # Ensure proper data types for tensor conversion
        original_X_np = original_df[feature_cols].values.astype(np.float32)
        adversarial_X_np = adversarial_df[feature_cols].values.astype(np.float32)
        labels_np = original_df['Label'].values.astype(np.float32)
        
        original_X = torch.tensor(original_X_np, dtype=torch.float32)
        adversarial_X = torch.tensor(adversarial_X_np, dtype=torch.float32)
        labels = torch.tensor(labels_np, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            original_outputs = torch.sigmoid(model(original_X.to(self.device))).squeeze()
            adversarial_outputs = torch.sigmoid(model(adversarial_X.to(self.device))).squeeze()
            
            original_preds = (original_outputs > 0.5).float()
            adversarial_preds = (adversarial_outputs > 0.5).float()
        
        # Calculate metrics
        labels_np = labels.cpu().numpy()
        original_preds_np = original_preds.cpu().numpy()
        adversarial_preds_np = adversarial_preds.cpu().numpy()
        
        # Original performance
        original_accuracy = accuracy_score(labels_np, original_preds_np)
        original_f1 = f1_score(labels_np, original_preds_np, zero_division=0)
        
        # Adversarial performance
        adversarial_accuracy = accuracy_score(labels_np, adversarial_preds_np)
        adversarial_f1 = f1_score(labels_np, adversarial_preds_np, zero_division=0)
        
        # Attack success metrics
        prediction_changes = (original_preds != adversarial_preds).sum().item()
        attack_success_rate = prediction_changes / len(labels)
        
        # Perturbation analysis
        perturbations = adversarial_X - original_X
        l2_norms = torch.norm(perturbations, p=2, dim=1)
        avg_l2_perturbation = l2_norms.mean().item()
        max_l2_perturbation = l2_norms.max().item()
        
        evaluation_results = {
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'accuracy_drop': original_accuracy - adversarial_accuracy,
            'original_f1': original_f1,
            'adversarial_f1': adversarial_f1,
            'f1_drop': original_f1 - adversarial_f1,
            'attack_success_rate': attack_success_rate,
            'prediction_changes': prediction_changes,
            'total_samples': len(labels),
            'average_l2_perturbation': avg_l2_perturbation,
            'max_l2_perturbation': max_l2_perturbation
        }
        
        self.logger.info("Attack Effectiveness Evaluation:")
        self.logger.info(f"  Original Accuracy: {original_accuracy:.4f}")
        self.logger.info(f"  Adversarial Accuracy: {adversarial_accuracy:.4f}")
        self.logger.info(f"  Accuracy Drop: {original_accuracy - adversarial_accuracy:.4f}")
        self.logger.info(f"  Attack Success Rate: {attack_success_rate:.4f}")
        self.logger.info(f"  Average L2 Perturbation: {avg_l2_perturbation:.4f}")
        
        return evaluation_results
    
    def generate_adaptive_attacks(self,
                                 data_loader: DataLoader,
                                 attack_types: List[str] = ['fgsm', 'pgd'],
                                 epsilon_values: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, pd.DataFrame]:
        """
        Generate adaptive adversarial attacks with multiple epsilon values.
        
        Args:
            data_loader: DataLoader with test samples
            attack_types: List of attack types to generate
            epsilon_values: List of epsilon values to test
            
        Returns:
            Dictionary mapping attack configurations to adversarial DataFrames
        """
        self.logger.info("Generating adaptive adversarial attacks...")
        
        attack_results = {}
        
        for attack_type in attack_types:
            for epsilon in epsilon_values:
                config_name = f"{attack_type}_eps_{epsilon}"
                
                try:
                    if attack_type == 'fgsm':
                        adversarial_df = self.generate_fgsm_attacks(data_loader, epsilon)
                    elif attack_type == 'pgd':
                        adversarial_df = self.generate_pgd_attacks(data_loader, epsilon)
                    else:
                        self.logger.warning(f"Unknown attack type: {attack_type}")
                        continue
                    
                    attack_results[config_name] = adversarial_df
                    self.logger.info(f"✅ Generated {config_name} attacks: {len(adversarial_df)} samples")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to generate {config_name}: {e}")
                    continue
        
        self.logger.info(f"Adaptive attack generation complete: {len(attack_results)} configurations")
        return attack_results
    
    def get_attack_summary(self) -> Dict[str, Any]:
        """Get comprehensive attack summary and statistics"""
        if not self.attack_history:
            return {'message': 'No attacks performed yet'}
        
        total_attacks = len(self.attack_history)
        total_samples = sum(attack['total_samples'] for attack in self.attack_history)
        total_successes = sum(attack['successful_attacks'] for attack in self.attack_history)
        
        overall_success_rate = total_successes / total_samples if total_samples > 0 else 0
        
        attack_types = list(set(attack['attack_type'] for attack in self.attack_history))
        epsilon_values = list(set(attack['epsilon'] for attack in self.attack_history))
        
        summary = {
            'total_attacks_performed': total_attacks,
            'attack_types_used': attack_types,
            'epsilon_values_tested': epsilon_values,
            'total_samples_attacked': total_samples,
            'total_successful_attacks': total_successes,
            'overall_success_rate': overall_success_rate,
            'attack_history': self.attack_history,
            'model_prediction_stats': self.wrapped_model.get_prediction_stats()
        }
        
        return summary


if __name__ == "__main__":
    """Direct execution for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🔥 Adversarial Attack Module - Industrial Grade Implementation")
    print("This module provides FGSM and PGD attacks with comprehensive monitoring.")
    print("Use the Phase 3 execution script for complete attack generation.")