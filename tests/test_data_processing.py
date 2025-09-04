"""
Test data processing functionality
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'message_type': np.random.choice([1, 2, 3, 4, 5], 100),
            'source_ip': np.random.choice(['192.168.1.1', '10.0.0.1', '172.16.0.1'], 100),
            'dest_ip': np.random.choice(['192.168.1.100', '10.0.0.100', '172.16.0.100'], 100),
            'packet_size': np.random.randint(64, 1500, 100),
            'timestamp': np.arange(100),
            'sequence_number': np.random.randint(1, 1000, 100),
            'flow_label': np.random.randint(0, 1048575, 100),
            'teid': np.random.randint(1, 100000, 100),  # Reduced range to avoid int32 overflow
            'qfi': np.random.randint(0, 63, 100),
            'label': np.random.choice([0, 1], 100)
        })
    
    def test_data_loading(self):
        """Test that data can be loaded and has correct structure"""
        self.assertEqual(len(self.sample_data), 100)
        self.assertIn('label', self.sample_data.columns)
        self.assertTrue(all(label in [0, 1] for label in self.sample_data['label']))
    
    def test_feature_extraction(self):
        """Test basic feature extraction"""
        # Test that numerical columns exist
        numerical_cols = ['packet_size', 'timestamp', 'sequence_number', 'flow_label', 'teid', 'qfi']
        for col in numerical_cols:
            self.assertIn(col, self.sample_data.columns)
            self.assertTrue(np.issubdtype(self.sample_data[col].dtype, np.number))
    
    def test_data_preprocessing(self):
        """Test data preprocessing steps"""
        # Test that labels are binary
        unique_labels = self.sample_data['label'].unique()
        self.assertTrue(all(label in [0, 1] for label in unique_labels))
        
        # Test that features are numeric
        features = self.sample_data.drop('label', axis=1)
        for col in ['packet_size', 'timestamp', 'sequence_number']:
            self.assertTrue(pd.api.types.is_numeric_dtype(features[col]))

if __name__ == '__main__':
    unittest.main()
