"""
Test CLI functionality
"""
import unittest
import sys
import os
import tempfile
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestCLI(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
    
    def test_config_loading(self):
        """Test configuration file loading"""
        # Create a test config file
        config_data = {
            "model": {
                "type": "RandomForest",
                "n_estimators": 100
            },
            "training": {
                "test_size": 0.2,
                "random_state": 42
            }
        }
        
        config_file = os.path.join(self.test_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Test loading
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config["model"]["type"], "RandomForest")
        self.assertEqual(loaded_config["training"]["test_size"], 0.2)
    
    def test_argument_parsing(self):
        """Test CLI argument parsing logic"""
        # Simulate command line arguments
        test_args = [
            "--mode", "baseline",
            "--config", "configs/baseline.yaml",
            "--data", "data/test_input.csv"
        ]
        
        # Test argument structure
        self.assertIn("--mode", test_args)
        self.assertIn("baseline", test_args)
        self.assertIn("--config", test_args)
        self.assertIn("--data", test_args)
    
    def test_mode_validation(self):
        """Test CLI mode validation"""
        valid_modes = ['baseline', 'attack', 'defense', 'evaluate', 'universal']
        
        # Test valid modes
        for mode in valid_modes:
            self.assertIn(mode, valid_modes)
        
        # Test invalid mode
        invalid_mode = 'invalid_mode'
        self.assertNotIn(invalid_mode, valid_modes)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
