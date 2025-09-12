#!/usr/bin/env python3
"""
Adversarial Training Implementation for Enhanced IDS
Combines superior clean performance with robust adversarial defense
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import json
import logging
from pathlib import Path
from tqdm import tqdm
import argparse
from datetime import datetime
import sys
import os

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Add current directory to path for local imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from capstone_presentation.src.baseline import BaselineIDS, PFCPDataset  
import torchattacks

class SimpleConfig:
    """Simple configuration class for loading YAML configs."""
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key, default=None):
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

class AdversarialTrainer:
    """Enhanced adversarial training with multiple attack types and adaptive strategies."""
    
    def __init__(self, config_path: str):
        self.config = SimpleConfig(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Load datasets
        self.load_datasets()
        
        # Initialize model
        self.model = self.load_pretrained_model()
        
        # Setup training components
        self.setup_training()
        
        # Initialize attack methods
        self.setup_attacks()
        
        # Training state
        self.best_val_f1 = 0.0
        self.epochs_without_improvement = 0
        
    def setup_logging(self):
        """Configure comprehensive logging."""
        log_dir = Path(self.config.get('logging.log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"adversarial_training_{timestamp}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging.level', 'INFO')),
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Adversarial training initialized. Logs: {log_file}")
        
    def load_datasets(self):
        """Load and prepare datasets for adversarial training."""
        # Load datasets
        train_dataset = PFCPDataset(
            self.config.get('data.train_file'),
            self.config.get('data.feature_columns_file')
        )
        
        val_dataset = PFCPDataset(
            self.config.get('data.val_file'),
            self.config.get('data.feature_columns_file')
        )
        
        test_dataset = PFCPDataset(
            self.config.get('data.test_file'),
            self.config.get('data.feature_columns_file')
        )
        
        # Create data loaders
        batch_size = self.config.get('training.batch_size', 512)
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        self.num_features = train_dataset.X.shape[1]
        
        # Calculate class weights
        train_labels = train_dataset.y
        pos_count = torch.sum(train_labels).item()
        neg_count = len(train_labels) - pos_count
        self.pos_weight = torch.tensor([neg_count / pos_count]).to(self.device)
        
        self.logger.info(f"Loaded datasets - Train: {len(train_dataset)}, "
                        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        self.logger.info(f"Using {self.num_features} features")
        self.logger.info(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
        self.logger.info(f"Calculated pos_weight: {self.pos_weight.item():.4f}")
        
    def load_pretrained_model(self):
        """Load the pre-trained enhanced baseline model."""
        model = BaselineIDS(
            input_features=self.num_features
        ).to(self.device)
        
        # Load pre-trained weights
        pretrained_path = self.config.get('model.pretrained_model_path')
        if pretrained_path and Path(pretrained_path).exists():
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded pre-trained model from {pretrained_path}")
        else:
            self.logger.warning("No pre-trained model found. Starting from scratch.")
            
        return model
        
    def setup_training(self):
        """Setup training components."""
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # Optimizer with lower learning rate for fine-tuning
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('training.adversarial_lr', 0.0001),  # Lower LR for adversarial training
            weight_decay=self.config.get('training.weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )
        
    def setup_attacks(self):
        """Initialize adversarial attack methods for training."""
        # Wrapper for torchattacks compatibility
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                
            def forward(self, x):
                logits = self.model(x)
                # Convert to probabilities for binary classification
                probs = torch.sigmoid(logits)
                # Stack for binary classification compatibility
                return torch.stack([1-probs.squeeze(), probs.squeeze()], dim=1)
        
        wrapped_model = ModelWrapper(self.model)
        
        # Attack configurations
        attack_config = self.config.get('adversarial_training.attacks', {})
        
        self.attacks = {}
        
        # FGSM attack
        if attack_config.get('fgsm.enabled', True):
            fgsm_eps = attack_config.get('fgsm.epsilon', 0.001)
            self.attacks['fgsm'] = torchattacks.FGSM(wrapped_model, eps=fgsm_eps)
            
        # PGD attack  
        if attack_config.get('pgd.enabled', True):
            pgd_config = attack_config.get('pgd', {})
            self.attacks['pgd'] = torchattacks.PGD(
                wrapped_model,
                eps=pgd_config.get('epsilon', 0.001),
                alpha=pgd_config.get('alpha', 0.0002),
                steps=pgd_config.get('steps', 7)
            )
            
        # BIM attack
        if attack_config.get('bim.enabled', False):
            bim_config = attack_config.get('bim', {})
            self.attacks['bim'] = torchattacks.BIM(
                wrapped_model,
                eps=bim_config.get('epsilon', 0.001),
                alpha=bim_config.get('alpha', 0.0002),
                steps=bim_config.get('steps', 7)
            )
            
        self.logger.info(f"Initialized {len(self.attacks)} attack methods: {list(self.attacks.keys())}")
        
    def generate_adversarial_examples(self, x, y, attack_ratio=0.5):
        """Generate adversarial examples for training."""
        if not self.attacks:
            return x
            
        batch_size = x.shape[0]
        num_adv = int(batch_size * attack_ratio)
        
        if num_adv == 0:
            return x
            
        # Randomly select samples for adversarial attack
        adv_indices = torch.randperm(batch_size)[:num_adv]
        x_adv = x.clone()
        
        # Convert labels for torchattacks (expects class indices)
        y_attack = y[adv_indices].long().squeeze()
        
        # Randomly select attack method
        attack_name = np.random.choice(list(self.attacks.keys()))
        attack = self.attacks[attack_name]
        
        # Generate adversarial examples
        with torch.enable_grad():
            x_adv_batch = attack(x[adv_indices], y_attack)
            x_adv[adv_indices] = x_adv_batch
            
        return x_adv
        
    def train_epoch(self, epoch):
        """Train one epoch with adversarial examples."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Progressive adversarial training strategy
        attack_ratio = min(0.1 + (epoch - 1) * 0.02, 0.5)  # Gradually increase adversarial ratio
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config.get('training.epochs')}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(self.device), y.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # Generate adversarial examples
            x_adv = self.generate_adversarial_examples(x, y, attack_ratio)
            
            # Forward pass
            outputs = self.model(x_adv)
            loss = self.criterion(outputs, y.unsqueeze(1))
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predictions.squeeze() == y).sum().item()
            total_samples += y.size(0)
            
            # Update progress bar
            accuracy = correct_predictions / total_samples
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'adv_ratio': f'{attack_ratio:.2f}'
            })
            
        return total_loss / len(self.train_loader), correct_predictions / total_samples
        
    def evaluate(self, data_loader, description="Validation"):
        """Evaluate model performance."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in tqdm(data_loader, desc=description):
                x, y = x.to(self.device), y.to(self.device).float()
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y.unsqueeze(1))
                total_loss += loss.item()
                
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                all_predictions.extend(predictions.squeeze().cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        # Calculate metrics
        predictions_np = np.array(all_predictions)
        labels_np = np.array(all_labels)
        
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
        
        f1 = f1_score(labels_np, predictions_np)
        precision = precision_score(labels_np, predictions_np, zero_division=0)
        recall = recall_score(labels_np, predictions_np, zero_division=0)
        accuracy = accuracy_score(labels_np, predictions_np)
        avg_loss = total_loss / len(data_loader)
        
        return {
            'loss': avg_loss,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
        
    def evaluate_adversarial_robustness(self, data_loader=None):
        """Evaluate model's adversarial robustness."""
        if data_loader is None:
            data_loader = self.val_loader
            
        results = {}
        
        # Clean evaluation
        clean_metrics = self.evaluate(data_loader, "Clean Evaluation")
        results['clean'] = clean_metrics
        
        # Adversarial evaluation for each attack
        for attack_name, attack in self.attacks.items():
            self.logger.info(f"Evaluating robustness against {attack_name.upper()} attack...")
            
            all_predictions = []
            all_labels = []
            
            self.model.eval()
            for x, y in tqdm(data_loader, desc=f"{attack_name.upper()} Attack"):
                x, y = x.to(self.device), y.to(self.device).float()
                
                # Enable gradients for adversarial attack generation
                x.requires_grad_(True)
                
                # Convert labels for attack
                y_attack = y.long()
                
                # Generate adversarial examples
                with torch.enable_grad():
                    x_adv = attack(x, y_attack)
                
                # Get predictions on adversarial examples (no gradients needed)
                with torch.no_grad():
                    outputs = self.model(x_adv.detach())
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    
                    all_predictions.extend(predictions.squeeze().cpu().numpy())
                    all_labels.extend(y.cpu().numpy())
                    
            # Calculate metrics
            from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
            
            predictions_np = np.array(all_predictions)
            labels_np = np.array(all_labels)
            
            results[attack_name] = {
                'f1': f1_score(labels_np, predictions_np),
                'precision': precision_score(labels_np, predictions_np, zero_division=0),
                'recall': recall_score(labels_np, predictions_np, zero_division=0),
                'accuracy': accuracy_score(labels_np, predictions_np)
            }
            
        return results
        
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.get('model.checkpoint_dir', 'checkpoints'))
        checkpoint_path.mkdir(exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path / 'latest_adversarial_checkpoint.pth')
        
        # Save best model
        if is_best:
            model_path = self.config.get('model.adversarial_model_path', 'adversarial_robust_model.pt')
            torch.save(checkpoint, model_path)
            self.logger.info(f"New best model saved (F1: {metrics['f1']:.4f})")
            
    def train(self):
        """Main adversarial training loop."""
        epochs = self.config.get('training.epochs', 30)
        patience = self.config.get('training.patience', 10)
        
        self.logger.info(f"Starting adversarial training - {epochs} epochs, patience={patience}")
        self.logger.info(f"Using attacks: {list(self.attacks.keys())}")
        
        training_history = []
        
        for epoch in range(1, epochs + 1):
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.evaluate(self.val_loader, f"Epoch {epoch} [Val]")
            
            # Update learning rate
            self.scheduler.step(val_metrics['f1'])
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}, Precision: {val_metrics['precision']:.4f}, "
                f"Recall: {val_metrics['recall']:.4f}"
            )
            
            # Check for improvement
            is_best = val_metrics['f1'] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics['f1']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                
            # Save checkpoint
            epoch_data = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                **val_metrics
            }
            training_history.append(epoch_data)
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                self.logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
                
        # Final evaluation
        self.logger.info("Performing final adversarial robustness evaluation...")
        
        # Load best model
        best_model_path = self.config.get('model.adversarial_model_path', 'adversarial_robust_model.pt')
        if Path(best_model_path).exists():
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Test evaluation
        test_metrics = self.evaluate(self.test_loader, "Final Test")
        robustness_metrics = self.evaluate_adversarial_robustness(self.test_loader)
        
        # Save final results
        results = {
            'training_history': training_history,
            'final_test_metrics': test_metrics,
            'adversarial_robustness': robustness_metrics,
            'config': dict(self.config.config)
        }
        
        results_path = Path(self.config.get('results.save_dir', 'results/adversarial'))
        results_path.mkdir(parents=True, exist_ok=True)
        
        with open(results_path / 'adversarial_training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Adversarial training complete! Results saved to {results_path}")
        self.logger.info(f"Final Test - F1: {test_metrics['f1']:.4f}, "
                        f"Precision: {test_metrics['precision']:.4f}, "
                        f"Recall: {test_metrics['recall']:.4f}")
        
        # Log robustness summary
        self.logger.info("Adversarial Robustness Summary:")
        for attack_name, metrics in robustness_metrics.items():
            if attack_name != 'clean':
                self.logger.info(f"  {attack_name.upper()}: F1 = {metrics['f1']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Adversarial Training for Enhanced IDS")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to adversarial training configuration file')
    
    args = parser.parse_args()
    
    # Initialize and run adversarial training
    trainer = AdversarialTrainer(args.config)
    trainer.train()

if __name__ == "__main__":
    main()