"""
Test suite for 5G Adversarial IDS System
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_data():
    """Create sample 5G network data for testing"""
    np.random.seed(42)
    return pd.DataFrame({
        'message_type': np.random.choice([1, 2, 3, 4, 5], 100),
        'source_ip': np.random.choice(['192.168.1.1', '10.0.0.1', '172.16.0.1'], 100),
        'dest_ip': np.random.choice(['192.168.1.100', '10.0.0.100', '172.16.0.100'], 100),
        'packet_size': np.random.randint(64, 1500, 100),
        'timestamp': np.arange(100),
        'sequence_number': np.random.randint(1, 1000, 100),
        'flow_label': np.random.randint(0, 1048575, 100),
        'teid': np.random.randint(1, 4294967295, 100),
        'qfi': np.random.randint(0, 63, 100),
        'label': np.random.choice([0, 1], 100)  # 0: normal, 1: attack
    })

@pytest.fixture
def sample_features():
    """Create sample feature matrix"""
    np.random.seed(42)
    return np.random.randn(100, 10)

@pytest.fixture
def sample_labels():
    """Create sample labels"""
    np.random.seed(42)
    return np.random.choice([0, 1], 100)
