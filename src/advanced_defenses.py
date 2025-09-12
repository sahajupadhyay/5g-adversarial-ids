"""
Advanced Adversarial Defense Techniques
=====================================

Implements cutting-edge adversarial defense methods for 5G IDS systems:
- Adversarial Weight Perturbation (AWP)
- TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
- Certified Defenses
- Ensemble Methods
- Input Preprocessing Defenses

Author: AI Assistant
Date: September 12, 2025
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

class AWPDefense(nn.Module):
    """
    Adversarial Weight Perturbation Defense
    
    Adds perturbations to model weights during training to improve robustness.
    """
    
    def __init__(self, model: nn.Module, proxy: nn.Module, gamma: float = 0.01):
        super().__init__()
        self.model = model
        self.proxy = proxy
        self.gamma = gamma
        
    def calc_awp(self, inputs_adv, targets):
        """Calculate adversarial weight perturbation"""
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        loss = F.cross_entropy(self.proxy(inputs_adv), targets)
        
        # Calculate gradients w.r.t. proxy parameters
        proxy_grads = torch.autograd.grad(
            loss, self.proxy.parameters(), create_graph=False
        )
        
        # Apply perturbations to main model
        with torch.no_grad():
            for param, proxy_grad in zip(self.model.parameters(), proxy_grads):
                param.add_(proxy_grad, alpha=-self.gamma)
                
        return loss

class TRADESDefense:
    """
    TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)
    
    Balances natural accuracy and adversarial robustness through a regularization term.
    """
    
    def __init__(self, beta: float = 6.0, distance: str = 'l_inf'):
        self.beta = beta
        self.distance = distance
        
    def trades_loss(self, model, x_natural, y, optimizer, step_size=0.003, 
                   epsilon=0.031, perturb_steps=10):
        """
        TRADES training loss combining natural and adversarial terms
        """
        # Define KL divergence loss
        criterion_kl = nn.KLDivLoss(reduction='sum')
        model.eval()
        
        batch_size = len(x_natural)
        
        # Generate adversarial examples
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                     F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), 
                            x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
        model.train()
        
        x_adv = x_adv.detach()
        
        # Calculate TRADES loss
        logits = model(x_natural)
        loss_natural = F.cross_entropy(logits, y)
        loss_robust = (1.0 / batch_size) * criterion_kl(
            F.log_softmax(model(x_adv), dim=1),
            F.softmax(model(x_natural), dim=1)
        )
        
        loss = loss_natural + self.beta * loss_robust
        return loss

class EnsembleDefense:
    """
    Ensemble-based adversarial defense using multiple diverse models
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
    def ensemble_predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make ensemble prediction"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(weight * pred)
                
        return torch.sum(torch.stack(predictions), dim=0)

class InputPreprocessingDefense:
    """
    Input preprocessing defenses including feature squeezing and noise injection
    """
    
    def __init__(self, defense_type: str = 'feature_squeezing', **kwargs):
        self.defense_type = defense_type
        self.kwargs = kwargs
        
    def feature_squeezing(self, x: torch.Tensor, bit_depth: int = 8) -> torch.Tensor:
        """Reduce precision of input features"""
        x_squeezed = torch.round(x * (2**bit_depth - 1)) / (2**bit_depth - 1)
        return x_squeezed
        
    def gaussian_noise_defense(self, x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise to inputs"""
        noise = torch.randn_like(x) * std
        return x + noise
        
    def apply_defense(self, x: torch.Tensor) -> torch.Tensor:
        """Apply selected preprocessing defense"""
        if self.defense_type == 'feature_squeezing':
            return self.feature_squeezing(x, **self.kwargs)
        elif self.defense_type == 'gaussian_noise':
            return self.gaussian_noise_defense(x, **self.kwargs)
        else:
            return x

class CertifiedDefense:
    """
    Certified adversarial defense using randomized smoothing
    """
    
    def __init__(self, base_classifier: nn.Module, num_classes: int, sigma: float = 0.25):
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        
    def smooth_predict(self, x: torch.Tensor, n: int = 1000, batch_size: int = 100) -> torch.Tensor:
        """Make prediction using randomized smoothing"""
        self.base_classifier.eval()
        
        with torch.no_grad():
            counts = torch.zeros(x.shape[0], self.num_classes)
            
            for i in range(0, n, batch_size):
                current_batch_size = min(batch_size, n - i)
                
                # Add Gaussian noise
                noise = torch.randn(x.shape[0], current_batch_size, *x.shape[1:]) * self.sigma
                noisy_x = x.unsqueeze(1) + noise
                noisy_x = noisy_x.reshape(-1, *x.shape[1:])
                
                # Get predictions
                predictions = self.base_classifier(noisy_x)
                predictions = predictions.reshape(x.shape[0], current_batch_size, self.num_classes)
                
                # Count predictions
                predicted_classes = predictions.argmax(dim=2)
                for j in range(x.shape[0]):
                    for k in range(current_batch_size):
                        counts[j, predicted_classes[j, k]] += 1
                        
            return counts.argmax(dim=1)

class AdvancedDefenseTrainer:
    """
    Trainer for advanced adversarial defense techniques
    """
    
    def __init__(self, model: nn.Module, defense_config: Dict[str, Any]):
        self.model = model
        self.config = defense_config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize defenses based on config
        self.defenses = self._initialize_defenses()
        
    def _initialize_defenses(self) -> Dict[str, Any]:
        """Initialize defense mechanisms"""
        defenses = {}
        
        if self.config.get('use_awp', False):
            proxy_model = type(self.model)(**self.config.get('model_params', {}))
            defenses['awp'] = AWPDefense(
                self.model, proxy_model, 
                gamma=self.config.get('awp_gamma', 0.01)
            )
            
        if self.config.get('use_trades', False):
            defenses['trades'] = TRADESDefense(
                beta=self.config.get('trades_beta', 6.0),
                distance=self.config.get('trades_distance', 'l_inf')
            )
            
        if self.config.get('use_input_preprocessing', False):
            defenses['preprocessing'] = InputPreprocessingDefense(
                defense_type=self.config.get('preprocessing_type', 'feature_squeezing'),
                **self.config.get('preprocessing_params', {})
            )
            
        return defenses
        
    def train_with_defenses(self, train_loader, val_loader, epochs: int = 50):
        """Train model with advanced defense techniques"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Apply input preprocessing if enabled
                if 'preprocessing' in self.defenses:
                    data = self.defenses['preprocessing'].apply_defense(data)
                
                # Calculate loss based on enabled defenses
                if 'trades' in self.defenses:
                    loss = self.defenses['trades'].trades_loss(
                        self.model, data, target, optimizer
                    )
                else:
                    output = self.model(data)
                    loss = F.cross_entropy(output, target)
                
                # Apply AWP if enabled
                if 'awp' in self.defenses:
                    # Generate adversarial examples for AWP
                    data_adv = self._generate_adversarial_examples(data, target)
                    awp_loss = self.defenses['awp'].calc_awp(data_adv, target)
                    loss += 0.1 * awp_loss  # Weighted AWP loss
                
                loss.backward()
                optimizer.step()
                
                # Restore original weights after AWP
                if 'awp' in self.defenses:
                    self.defenses['awp'].proxy.load_state_dict(self.model.state_dict())
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            val_acc = self._validate(val_loader)
            
            self.logger.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
            
    def _generate_adversarial_examples(self, data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Generate adversarial examples using PGD"""
        epsilon = 0.1
        alpha = 0.01
        num_iter = 10
        
        data_adv = data.clone().detach()
        data_adv += torch.empty_like(data_adv).uniform_(-epsilon, epsilon)
        
        for _ in range(num_iter):
            data_adv.requires_grad_()
            
            with torch.enable_grad():
                output = self.model(data_adv)
                loss = F.cross_entropy(output, target)
                
            grad = torch.autograd.grad(loss, [data_adv])[0]
            
            data_adv = data_adv.detach() + alpha * grad.sign()
            data_adv = torch.max(torch.min(data_adv, data + epsilon), data - epsilon)
            
        return data_adv
        
    def _validate(self, val_loader) -> float:
        """Validate model performance"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                if 'preprocessing' in self.defenses:
                    data = self.defenses['preprocessing'].apply_defense(data)
                    
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return correct / total

# Example usage and configuration
def create_defense_config() -> Dict[str, Any]:
    """Create a comprehensive defense configuration"""
    return {
        'use_awp': True,
        'awp_gamma': 0.01,
        'use_trades': True,
        'trades_beta': 6.0,
        'trades_distance': 'l_inf',
        'use_input_preprocessing': True,
        'preprocessing_type': 'feature_squeezing',
        'preprocessing_params': {'bit_depth': 8},
        'model_params': {
            'input_dim': 80,
            'hidden_dims': [256, 128, 64],
            'output_dim': 1,
            'dropout_rate': 0.5
        }
    }

# Export main classes
__all__ = [
    'AWPDefense',
    'TRADESDefense', 
    'EnsembleDefense',
    'InputPreprocessingDefense',
    'CertifiedDefense',
    'AdvancedDefenseTrainer',
    'create_defense_config'
]