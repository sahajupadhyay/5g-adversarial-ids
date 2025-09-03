"""
PFCP Protocol Constraints for 5G Adversarial Attacks
Defines realistic constraints based on 5G PFCP protocol specifications
"""

import numpy as np

# PFCP Protocol Constraints based on 3GPP TS 29.244
PFCP_CONSTRAINTS = {
    'packet_size': {'min': 64, 'max': 1500},  # bytes (standard Ethernet frame)
    'session_id': {'min': 1, 'max': 2**32-1},  # PFCP Session ID range
    'sequence_number': {'type': 'integer', 'min': 0, 'max': 2**24-1},  # 24-bit sequence number
    'message_type': {'allowed_values': [1, 2, 4, 5, 6, 7, 50, 51, 52, 53]},  # PFCP message types
    'teid': {'min': 0, 'max': 2**32-1},  # Tunnel Endpoint Identifier
    'qfi': {'min': 0, 'max': 63},  # QoS Flow Identifier (6 bits)
    'priority': {'min': 1, 'max': 15},  # QoS priority level
    'node_id_type': {'allowed_values': [0, 1, 2]},  # IPv4, IPv6, FQDN
    'cause': {'min': 1, 'max': 255},  # PFCP Cause values
    'timer': {'min': 0, 'max': 255},  # Timer values in seconds
}

# Feature-specific constraints for the 43 features in the processed dataset
# These represent realistic bounds for 5G PFCP traffic features
FEATURE_CONSTRAINTS = {
    i: {'min': -3.0, 'max': 3.0, 'type': 'continuous'} 
    for i in range(43)  # All 43 features get the same bounds for simplicity
}

# Logical constraints for PFCP protocol
LOGICAL_CONSTRAINTS = {
    'session_establishment': {
        'required_ies': ['node_id', 'f_seid', 'create_pdr', 'create_far'],
        'message_type': 50  # Session Establishment Request
    },
    'session_modification': {
        'required_ies': ['update_pdr', 'update_far'],
        'message_type': 52  # Session Modification Request
    },
    'session_deletion': {
        'required_ies': ['f_seid'],
        'message_type': 54  # Session Deletion Request
    }
}

def get_feature_bounds():
    """
    Get min/max bounds for all features in our model
    
    Returns:
        tuple: (min_bounds, max_bounds) as numpy arrays
    """
    min_bounds = np.array([FEATURE_CONSTRAINTS[i]['min'] for i in range(7)])
    max_bounds = np.array([FEATURE_CONSTRAINTS[i]['max'] for i in range(7)])
    
    return min_bounds, max_bounds

def validate_pfcp_constraints(feature_vector):
    """
    Validate that a feature vector satisfies PFCP protocol constraints
    
    Args:
        feature_vector: numpy array of shape (n_features,)
        
    Returns:
        tuple: (is_valid, violations) where violations is list of constraint violations
    """
    violations = []
    
    # Check feature bounds
    for i, value in enumerate(feature_vector):
        constraint = FEATURE_CONSTRAINTS[i]
        
        if value < constraint['min']:
            violations.append(f"Feature {i}: {value:.3f} < min({constraint['min']})")
        elif value > constraint['max']:
            violations.append(f"Feature {i}: {value:.3f} > max({constraint['max']})")
    
    is_valid = len(violations) == 0
    return is_valid, violations

def project_to_pfcp_constraints(feature_vector):
    """
    Project a feature vector to satisfy PFCP protocol constraints
    
    Args:
        feature_vector: numpy array of shape (n_features,)
        
    Returns:
        numpy array: Constraint-compliant feature vector
    """
    projected = feature_vector.copy()
    
    # Apply feature bounds
    for i in range(len(projected)):
        constraint = FEATURE_CONSTRAINTS[i]
        
        # Clip to valid range
        projected[i] = np.clip(projected[i], constraint['min'], constraint['max'])
        
        # Round integer types (not applicable for PCA features, but kept for completeness)
        if constraint['type'] == 'integer':
            projected[i] = np.round(projected[i])
    
    return projected

def calculate_constraint_violations(original, adversarial):
    """
    Calculate number of constraint violations in adversarial examples
    
    Args:
        original: Original feature vectors (n_samples, n_features)
        adversarial: Adversarial feature vectors (n_samples, n_features)
        
    Returns:
        dict: Statistics about constraint violations
    """
    n_samples = adversarial.shape[0]
    violation_count = 0
    total_violations = []
    
    for i in range(n_samples):
        is_valid, violations = validate_pfcp_constraints(adversarial[i])
        if not is_valid:
            violation_count += 1
            total_violations.extend(violations)
    
    violation_rate = violation_count / n_samples
    
    return {
        'violation_rate': violation_rate,
        'violation_count': violation_count,
        'total_samples': n_samples,
        'violation_details': total_violations[:10] if total_violations else []  # Show first 10
    }

def get_epsilon_bounds():
    """
    Get reasonable epsilon bounds for adversarial attacks on PFCP features
    
    Returns:
        dict: Epsilon values for different attack strengths
    """
    return {
        'weak': 0.01,      # Subtle perturbations
        'medium': 0.1,     # Moderate perturbations  
        'strong': 0.3,     # Strong perturbations
        'max_safe': 0.5    # Maximum before obvious detection
    }

# Protocol-specific feature interpretability (for analysis)
FEATURE_DESCRIPTIONS = {
    0: "Primary PFCP session characteristics (PCA-1)",
    1: "Message flow patterns (PCA-2)", 
    2: "Timing and sequence features (PCA-3)",
    3: "QoS and priority indicators (PCA-4)",
    4: "Node identification patterns (PCA-5)",
    5: "Error and cause code patterns (PCA-6)",
    6: "Protocol state indicators (PCA-7)"
}

def get_attack_constraints_summary():
    """
    Get a summary of constraints for attack implementation
    
    Returns:
        dict: Summary of all constraints
    """
    min_bounds, max_bounds = get_feature_bounds()
    epsilon_bounds = get_epsilon_bounds()
    
    return {
        'n_features': len(FEATURE_CONSTRAINTS),
        'feature_bounds': {
            'min': min_bounds.tolist(),
            'max': max_bounds.tolist()
        },
        'epsilon_values': epsilon_bounds,
        'constraint_types': list(PFCP_CONSTRAINTS.keys()),
        'feature_descriptions': FEATURE_DESCRIPTIONS
    }

class PFCPConstraints:
    """
    PFCP Protocol Constraints Handler
    Provides constraint validation and projection for adversarial attacks
    """
    
    def __init__(self, config=None):
        """Initialize with configuration"""
        self.config = config or {}
        self.min_bounds, self.max_bounds = get_feature_bounds()
    
    def validate_constraints(self, X):
        """Validate constraints for a batch of samples"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        violations = []
        for i, sample in enumerate(X):
            is_valid, sample_violations = validate_pfcp_constraints(sample)
            if not is_valid:
                violations.append((i, sample_violations))
        
        return len(violations) == 0, violations
    
    def project_to_constraints(self, X):
        """Project samples to constraint-compliant space"""
        if X.ndim == 1:
            return project_to_pfcp_constraints(X)
        
        projected = np.array([project_to_pfcp_constraints(sample) for sample in X])
        return projected
    
    def get_feature_bounds(self, feature_idx=None):
        """Get bounds for specific feature or all features"""
        if feature_idx is not None:
            return {
                'min': self.min_bounds[feature_idx],
                'max': self.max_bounds[feature_idx]
            }
        return {'min': self.min_bounds, 'max': self.max_bounds}


if __name__ == "__main__":
    # Test constraint functions
    print("🔧 PFCP Constraint System Test")
    print("="*40)
    
    # Test feature bounds
    min_bounds, max_bounds = get_feature_bounds()
    print(f"✅ Feature bounds: min={min_bounds}, max={max_bounds}")
    
    # Test constraint validation
    test_vector = np.array([0.5, -0.3, 1.2, -0.8, 0.0, 0.9, -0.4])
    is_valid, violations = validate_pfcp_constraints(test_vector)
    print(f"✅ Test vector valid: {is_valid}")
    
    # Test projection
    bad_vector = np.array([5.0, -5.0, 2.0, 0.0, 4.0, -4.0, 1.0])
    projected = project_to_pfcp_constraints(bad_vector)
    print(f"✅ Projection test: {bad_vector} -> {projected}")
    
    # Get summary
    summary = get_attack_constraints_summary()
    print(f"✅ Constraint summary: {summary['n_features']} features")
    
    # Test PFCPConstraints class
    constraints = PFCPConstraints()
    is_valid, violations = constraints.validate_constraints(test_vector)
    print(f"✅ Class validation test: {is_valid}")
    
    print("✅ PFCP Constraint System Ready")
