"""
Test machine learning models
"""
import unittest
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestModels(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.X_test = np.random.randn(50, 10)
        self.y_test = np.random.choice([0, 1], 50)
    
    def test_random_forest_baseline(self):
        """Test Random Forest baseline model training"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_test, self.y_test, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Test predictions
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        
        # Basic sanity checks
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)
        self.assertEqual(len(predictions), len(y_val))
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        import tempfile
        
        # Train a simple model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(self.X_test, self.y_test)
        
        # Save and load model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name
            
        try:
            joblib.dump(model, temp_path)
            loaded_model = joblib.load(temp_path)
            
            # Test that loaded model works
            predictions = loaded_model.predict(self.X_test)
            self.assertEqual(len(predictions), len(self.y_test))
            
        finally:
            # Cleanup with proper error handling
            try:
                os.unlink(temp_path)
            except (OSError, PermissionError):
                pass  # File cleanup failed, but test still passed

if __name__ == '__main__':
    unittest.main()
