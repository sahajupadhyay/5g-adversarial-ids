"""
Test adversarial attacks
"""
import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestAdversarialAttacks(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.choice([0, 1], 20)
    
    def test_fgsm_attack_generation(self):
        """Test FGSM attack generation"""
        # Simple FGSM implementation for testing
        def simple_fgsm(X, epsilon=0.1):
            """Simple FGSM attack for testing"""
            gradients = np.random.randn(*X.shape)  # Simulated gradients
            perturbation = epsilon * np.sign(gradients)
            return X + perturbation
        
        # Generate adversarial examples
        X_adv = simple_fgsm(self.X_test, epsilon=0.1)
        
        # Test that adversarial examples are different from originals
        self.assertFalse(np.array_equal(X_adv, self.X_test))
        
        # Test that perturbation is bounded
        perturbation = X_adv - self.X_test
        max_perturbation = np.max(np.abs(perturbation))
        self.assertLessEqual(max_perturbation, 0.1 + 1e-6)  # Small tolerance for floating point
    
    def test_pgd_attack_constraints(self):
        """Test PGD attack constraint enforcement"""
        def simple_pgd(X, epsilon=0.1, num_steps=5):
            """Simple PGD attack for testing"""
            X_adv = X.copy()
            
            for _ in range(num_steps):
                # Simulated gradient step
                gradients = np.random.randn(*X.shape)
                X_adv = X_adv + 0.01 * np.sign(gradients)
                
                # Project back to epsilon ball
                perturbation = X_adv - X
                perturbation = np.clip(perturbation, -epsilon, epsilon)
                X_adv = X + perturbation
            
            return X_adv
        
        # Generate PGD adversarial examples
        X_adv = simple_pgd(self.X_test, epsilon=0.1)
        
        # Test constraint satisfaction
        perturbation = X_adv - self.X_test
        max_perturbation = np.max(np.abs(perturbation))
        self.assertLessEqual(max_perturbation, 0.1 + 1e-6)
    
    def test_attack_success_metrics(self):
        """Test attack success rate calculation"""
        # Simulate original and adversarial predictions
        original_preds = np.array([0, 1, 0, 1, 0])
        adversarial_preds = np.array([1, 1, 1, 0, 0])
        
        # Calculate success rate (predictions that changed)
        success_rate = np.mean(original_preds != adversarial_preds)
        
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)
        self.assertEqual(success_rate, 0.6)  # 3 out of 5 changed

if __name__ == '__main__':
    unittest.main()
