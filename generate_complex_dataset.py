#!/usr/bin/env python3
"""
Advanced 5G Network Traffic Dataset Generator
Simulates realistic PFCP protocol traffic with complex patterns, anomalies, and edge cases
Specifically designed to stress-test the Universal Data Processing System
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import os

class Advanced5GDatasetGenerator:
    def __init__(self, base_samples=5000):
        self.base_samples = base_samples
        self.np_random = np.random.RandomState(42)  # For reproducibility
        
        # Real 5G PFCP protocol parameters
        self.pfcp_message_types = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51, 52, 53, 54, 55, 56, 57]
        self.node_types = ['UPF', 'SMF', 'AMF', 'PCF', 'UDM', 'NRF', 'AUSF']
        self.slice_types = ['eMBB', 'URLLC', 'mMTC', 'Enhanced_Mobile_Broadband', 'Ultra_Reliable_Low_Latency']
        
        # Attack patterns from real 5G threat intelligence
        self.attack_types = {
            'Normal': 0,
            'Malicious_Deletion': 1,
            'Malicious_Establishment': 2, 
            'Malicious_Modification_Type1': 3,
            'Malicious_Modification_Type2': 4,
            'Advanced_Persistent_Threat': 5,
            'Zero_Day_Exploit': 6,
            'Protocol_Fuzzing_Attack': 7,
            'Denial_of_Service': 8,
            'Session_Hijacking': 9
        }
        
    def generate_complex_dataset(self):
        """Generate comprehensive dataset with multiple complexity layers"""
        
        print("🚀 Generating Advanced 5G Network Traffic Dataset...")
        print("=" * 60)
        
        datasets = []
        
        # 1. Normal Traffic Patterns (40%)
        normal_data = self._generate_normal_traffic(int(self.base_samples * 0.4))
        datasets.append(normal_data)
        print(f"✅ Generated {len(normal_data)} normal traffic samples")
        
        # 2. Attack Traffic (35%)
        attack_data = self._generate_attack_traffic(int(self.base_samples * 0.35))
        datasets.append(attack_data)
        print(f"⚔️ Generated {len(attack_data)} attack traffic samples")
        
        # 3. Edge Cases & Anomalies (15%)
        edge_data = self._generate_edge_cases(int(self.base_samples * 0.15))
        datasets.append(edge_data)
        print(f"🔍 Generated {len(edge_data)} edge case samples")
        
        # 4. Stress Test Scenarios (10%)
        stress_data = self._generate_stress_scenarios(int(self.base_samples * 0.10))
        datasets.append(stress_data)
        print(f"💥 Generated {len(stress_data)} stress test samples")
        
        # Combine all datasets
        final_dataset = pd.concat(datasets, ignore_index=True)
        
        # Shuffle the dataset
        final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print("=" * 60)
        print(f"🎯 Total Dataset Size: {len(final_dataset)} samples")
        print(f"📊 Feature Count: {len(final_dataset.columns)} columns")
        
        return final_dataset
    
    def _generate_normal_traffic(self, n_samples):
        """Generate realistic normal 5G traffic patterns"""
        
        data = []
        
        for i in range(n_samples):
            # Simulate time-based patterns (peak hours, off-peak)
            hour = self.np_random.randint(0, 24)
            is_peak_hour = hour in [8, 9, 10, 11, 17, 18, 19, 20]
            
            # Base traffic characteristics
            base_throughput = 50 if is_peak_hour else 20
            base_latency = 5 if is_peak_hour else 2
            
            sample = {
                # Core PFCP metrics
                'session_establishment_rate': self.np_random.normal(base_throughput, 10),
                'session_modification_rate': self.np_random.normal(base_throughput * 0.3, 5),
                'session_deletion_rate': self.np_random.normal(base_throughput * 0.9, 8),
                'pfcp_message_type': self.np_random.choice(self.pfcp_message_types),
                'sequence_number': i + self.np_random.randint(0, 1000),
                
                # Network performance metrics
                'average_latency_ms': self.np_random.normal(base_latency, 1),
                'packet_loss_rate': self.np_random.exponential(0.001),
                'jitter_ms': self.np_random.exponential(0.5),
                'bandwidth_utilization': self.np_random.beta(3, 2) * 100,
                
                # 5G specific features
                'slice_type_encoded': self.np_random.randint(0, len(self.slice_types)),
                'qos_flow_identifier': self.np_random.randint(1, 64),
                'pdu_session_type': self.np_random.choice([0, 1, 2]),  # IPv4, IPv6, Ethernet
                'dnn_encoded': self.np_random.randint(0, 20),
                
                # Protocol compliance features
                'protocol_version': 1,  # PFCP v1
                'message_length': self.np_random.randint(50, 1500),
                'node_type_encoded': self.np_random.randint(0, len(self.node_types)),
                
                # Security features
                'encryption_strength': self.np_random.choice([128, 256]),
                'authentication_method': self.np_random.randint(0, 4),
                'certificate_validity': self.np_random.randint(1, 365),
                
                # Traffic patterns
                'peak_hour_indicator': 1 if is_peak_hour else 0,
                'weekend_indicator': self.np_random.choice([0, 1], p=[5/7, 2/7]),
                'geographic_region': self.np_random.randint(0, 10),
                
                # Performance indicators
                'cpu_utilization': self.np_random.normal(60 if is_peak_hour else 30, 15),
                'memory_utilization': self.np_random.normal(70 if is_peak_hour else 40, 20),
                'network_congestion_level': self.np_random.randint(0, 5),
                
                # Advanced features
                'handover_success_rate': self.np_random.normal(0.98, 0.02),
                'beam_management_efficiency': self.np_random.normal(0.85, 0.1),
                'massive_mimo_gain': self.np_random.normal(15, 3),
                'carrier_aggregation_bands': self.np_random.randint(1, 5),
                
                # Statistical features
                'inter_arrival_time': self.np_random.exponential(2),
                'burst_size': self.np_random.poisson(10),
                'flow_duration': self.np_random.lognormal(2, 1),
                
                # Correlation features (engineered)
                'throughput_latency_ratio': 0,  # Will be calculated
                'efficiency_score': 0,  # Will be calculated
                'anomaly_score': 0,  # Will be calculated
                
                # Complex temporal features
                'time_since_last_session': self.np_random.exponential(1),
                'session_frequency': self.np_random.gamma(2, 2),
                'periodic_pattern_strength': self.np_random.beta(2, 5),
                
                # Multi-dimensional quality features
                'user_experience_quality': self.np_random.normal(4.2, 0.8),  # Out of 5
                'network_reliability_score': self.np_random.normal(0.95, 0.05),
                'service_availability': self.np_random.normal(0.999, 0.001),
                
                # Edge computing features
                'edge_processing_delay': self.np_random.exponential(1),
                'cloud_offload_ratio': self.np_random.beta(2, 3),
                'mec_utilization': self.np_random.normal(0.6, 0.2),
                
                # Label
                'attack_type': 'Normal',
                'label': 0
            }
            
            # Calculate derived features
            sample['throughput_latency_ratio'] = sample['session_establishment_rate'] / max(sample['average_latency_ms'], 0.1)
            sample['efficiency_score'] = (sample['bandwidth_utilization'] * sample['handover_success_rate']) / 100
            sample['anomaly_score'] = abs(sample['cpu_utilization'] - 50) / 50
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def _generate_attack_traffic(self, n_samples):
        """Generate sophisticated attack patterns"""
        
        data = []
        attack_types = list(self.attack_types.keys())[1:]  # Exclude 'Normal'
        
        for i in range(n_samples):
            attack_type = self.np_random.choice(attack_types)
            attack_label = self.attack_types[attack_type]
            
            # Base normal sample
            sample = {
                'session_establishment_rate': self.np_random.normal(30, 15),
                'session_modification_rate': self.np_random.normal(10, 5),
                'session_deletion_rate': self.np_random.normal(25, 10),
                'pfcp_message_type': self.np_random.choice(self.pfcp_message_types),
                'sequence_number': i + self.np_random.randint(0, 1000),
                'average_latency_ms': self.np_random.normal(3, 2),
                'packet_loss_rate': self.np_random.exponential(0.002),
                'jitter_ms': self.np_random.exponential(1),
                'bandwidth_utilization': self.np_random.beta(2, 2) * 100,
                'slice_type_encoded': self.np_random.randint(0, len(self.slice_types)),
                'qos_flow_identifier': self.np_random.randint(1, 64),
                'pdu_session_type': self.np_random.choice([0, 1, 2]),
                'dnn_encoded': self.np_random.randint(0, 20),
                'protocol_version': 1,
                'message_length': self.np_random.randint(50, 1500),
                'node_type_encoded': self.np_random.randint(0, len(self.node_types)),
                'encryption_strength': self.np_random.choice([128, 256]),
                'authentication_method': self.np_random.randint(0, 4),
                'certificate_validity': self.np_random.randint(1, 365),
                'peak_hour_indicator': self.np_random.choice([0, 1]),
                'weekend_indicator': self.np_random.choice([0, 1]),
                'geographic_region': self.np_random.randint(0, 10),
                'cpu_utilization': self.np_random.normal(50, 20),
                'memory_utilization': self.np_random.normal(55, 25),
                'network_congestion_level': self.np_random.randint(0, 5),
                'handover_success_rate': self.np_random.normal(0.95, 0.05),
                'beam_management_efficiency': self.np_random.normal(0.80, 0.15),
                'massive_mimo_gain': self.np_random.normal(12, 5),
                'carrier_aggregation_bands': self.np_random.randint(1, 5),
                'inter_arrival_time': self.np_random.exponential(3),
                'burst_size': self.np_random.poisson(8),
                'flow_duration': self.np_random.lognormal(1, 1.5),
                'time_since_last_session': self.np_random.exponential(2),
                'session_frequency': self.np_random.gamma(1, 3),
                'periodic_pattern_strength': self.np_random.beta(1, 8),
                'user_experience_quality': self.np_random.normal(3.5, 1.2),
                'network_reliability_score': self.np_random.normal(0.90, 0.08),
                'service_availability': self.np_random.normal(0.995, 0.005),
                'edge_processing_delay': self.np_random.exponential(2),
                'cloud_offload_ratio': self.np_random.beta(1, 4),
                'mec_utilization': self.np_random.normal(0.4, 0.3),
            }
            
            # Apply attack-specific modifications
            if 'Deletion' in attack_type:
                sample['session_deletion_rate'] *= self.np_random.uniform(2, 5)  # Abnormally high deletion
                sample['packet_loss_rate'] *= self.np_random.uniform(3, 10)
                sample['session_establishment_rate'] *= self.np_random.uniform(0.1, 0.5)  # Low establishment
                
            elif 'Establishment' in attack_type:
                sample['session_establishment_rate'] *= self.np_random.uniform(3, 8)  # Flood attack
                sample['cpu_utilization'] *= self.np_random.uniform(1.5, 2.5)
                sample['memory_utilization'] *= self.np_random.uniform(1.3, 2.0)
                
            elif 'Modification' in attack_type:
                sample['session_modification_rate'] *= self.np_random.uniform(4, 10)
                sample['jitter_ms'] *= self.np_random.uniform(2, 6)
                sample['average_latency_ms'] *= self.np_random.uniform(1.5, 3)
                
            elif 'Persistent' in attack_type:
                sample['session_frequency'] *= self.np_random.uniform(0.1, 0.3)  # Low frequency, long duration
                sample['flow_duration'] *= self.np_random.uniform(5, 15)
                sample['periodic_pattern_strength'] *= self.np_random.uniform(2, 4)
                
            elif 'Zero_Day' in attack_type:
                # Completely anomalous patterns
                sample['pfcp_message_type'] = self.np_random.randint(100, 255)  # Invalid message type
                sample['protocol_version'] = self.np_random.randint(2, 10)  # Invalid version
                sample['message_length'] = self.np_random.randint(2000, 10000)  # Oversized messages
                
            elif 'Fuzzing' in attack_type:
                # Random field corruption
                corrupt_fields = self.np_random.choice(list(sample.keys()), size=5, replace=False)
                for field in corrupt_fields:
                    if isinstance(sample[field], (int, float)):
                        sample[field] = self.np_random.uniform(-1000, 1000)
                        
            elif 'Denial_of_Service' in attack_type:
                sample['session_establishment_rate'] *= self.np_random.uniform(10, 50)  # Massive flood
                sample['bandwidth_utilization'] = self.np_random.uniform(95, 100)
                sample['cpu_utilization'] = self.np_random.uniform(90, 100)
                sample['network_congestion_level'] = 4
                
            elif 'Hijacking' in attack_type:
                sample['authentication_method'] = self.np_random.randint(10, 99)  # Invalid auth
                sample['encryption_strength'] = self.np_random.choice([0, 64])  # Weak encryption
                sample['certificate_validity'] = 0  # Expired certificate
            
            # Calculate derived features
            sample['throughput_latency_ratio'] = sample['session_establishment_rate'] / max(sample['average_latency_ms'], 0.1)
            sample['efficiency_score'] = (sample['bandwidth_utilization'] * sample['handover_success_rate']) / 100
            sample['anomaly_score'] = abs(sample['cpu_utilization'] - 50) / 50
            
            # Add noise and make some values more realistic
            for key in sample:
                if isinstance(sample[key], float) and key not in ['attack_type', 'label']:
                    sample[key] = max(0, sample[key] + self.np_random.normal(0, abs(sample[key]) * 0.05))
            
            sample['attack_type'] = attack_type
            sample['label'] = attack_label
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def _generate_edge_cases(self, n_samples):
        """Generate challenging edge cases and boundary conditions"""
        
        data = []
        
        for i in range(n_samples):
            # Create various edge case scenarios
            edge_type = self.np_random.choice([
                'extreme_values', 'missing_equivalent', 'boundary_conditions', 
                'temporal_anomalies', 'correlation_breaks', 'multi_modal'
            ])
            
            sample = {
                'session_establishment_rate': 0,
                'session_modification_rate': 0,
                'session_deletion_rate': 0,
                'pfcp_message_type': 1,
                'sequence_number': i,
                'average_latency_ms': 0,
                'packet_loss_rate': 0,
                'jitter_ms': 0,
                'bandwidth_utilization': 0,
                'slice_type_encoded': 0,
                'qos_flow_identifier': 1,
                'pdu_session_type': 0,
                'dnn_encoded': 0,
                'protocol_version': 1,
                'message_length': 50,
                'node_type_encoded': 0,
                'encryption_strength': 128,
                'authentication_method': 0,
                'certificate_validity': 1,
                'peak_hour_indicator': 0,
                'weekend_indicator': 0,
                'geographic_region': 0,
                'cpu_utilization': 0,
                'memory_utilization': 0,
                'network_congestion_level': 0,
                'handover_success_rate': 0,
                'beam_management_efficiency': 0,
                'massive_mimo_gain': 0,
                'carrier_aggregation_bands': 1,
                'inter_arrival_time': 0,
                'burst_size': 0,
                'flow_duration': 0,
                'time_since_last_session': 0,
                'session_frequency': 0,
                'periodic_pattern_strength': 0,
                'user_experience_quality': 0,
                'network_reliability_score': 0,
                'service_availability': 0,
                'edge_processing_delay': 0,
                'cloud_offload_ratio': 0,
                'mec_utilization': 0,
                'throughput_latency_ratio': 0,
                'efficiency_score': 0,
                'anomaly_score': 0,
            }
            
            if edge_type == 'extreme_values':
                # Extremely high or low values
                sample['session_establishment_rate'] = self.np_random.choice([0.001, 10000])
                sample['average_latency_ms'] = self.np_random.choice([0.001, 1000])
                sample['bandwidth_utilization'] = self.np_random.choice([0.001, 99.999])
                sample['cpu_utilization'] = self.np_random.choice([0.1, 99.9])
                
            elif edge_type == 'missing_equivalent':
                # Values that would typically be NaN/missing
                sample['packet_loss_rate'] = 0
                sample['jitter_ms'] = 0
                sample['handover_success_rate'] = 0
                sample['user_experience_quality'] = 0
                
            elif edge_type == 'boundary_conditions':
                # Values at exact boundaries
                sample['bandwidth_utilization'] = self.np_random.choice([0, 100])
                sample['network_reliability_score'] = self.np_random.choice([0, 1])
                sample['service_availability'] = self.np_random.choice([0, 1])
                sample['protocol_version'] = self.np_random.choice([0, 255])
                
            elif edge_type == 'temporal_anomalies':
                # Time-based anomalies
                sample['time_since_last_session'] = self.np_random.choice([0, 86400])  # 0 or 24 hours
                sample['flow_duration'] = self.np_random.choice([0.001, 3600])  # 1ms or 1 hour
                sample['inter_arrival_time'] = self.np_random.choice([0.0001, 1000])
                
            elif edge_type == 'correlation_breaks':
                # Break expected correlations
                sample['session_establishment_rate'] = 1000  # High
                sample['average_latency_ms'] = 0.1  # Low (should be high with high rate)
                sample['cpu_utilization'] = 5  # Low (should be high)
                sample['bandwidth_utilization'] = 1  # Low (should be high)
                
            elif edge_type == 'multi_modal':
                # Multiple peaks/modes in distributions
                mode = self.np_random.choice([0, 1])
                if mode == 0:
                    sample['session_establishment_rate'] = self.np_random.normal(10, 2)
                    sample['average_latency_ms'] = self.np_random.normal(1, 0.2)
                else:
                    sample['session_establishment_rate'] = self.np_random.normal(100, 10)
                    sample['average_latency_ms'] = self.np_random.normal(20, 3)
            
            # Ensure no negative values where inappropriate
            for key in sample:
                if key not in ['attack_type', 'label'] and isinstance(sample[key], (int, float)):
                    if key in ['packet_loss_rate', 'jitter_ms', 'average_latency_ms']:
                        sample[key] = max(0, sample[key])
                    elif key in ['handover_success_rate', 'network_reliability_score', 'service_availability']:
                        sample[key] = max(0, min(1, sample[key]))
                    elif key == 'bandwidth_utilization':
                        sample[key] = max(0, min(100, sample[key]))
            
            # Calculate derived features
            sample['throughput_latency_ratio'] = sample['session_establishment_rate'] / max(sample['average_latency_ms'], 0.1)
            sample['efficiency_score'] = (sample['bandwidth_utilization'] * sample['handover_success_rate']) / 100
            sample['anomaly_score'] = abs(sample['cpu_utilization'] - 50) / 50
            
            sample['attack_type'] = 'Edge_Case_' + edge_type
            sample['label'] = self.np_random.choice([0, 1])  # Random label for edge cases
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def _generate_stress_scenarios(self, n_samples):
        """Generate extreme stress test scenarios"""
        
        data = []
        
        for i in range(n_samples):
            stress_type = self.np_random.choice([
                'memory_exhaustion', 'cpu_overload', 'network_saturation',
                'cascade_failure', 'race_condition', 'buffer_overflow'
            ])
            
            # Start with extreme base values
            sample = {
                'session_establishment_rate': self.np_random.uniform(500, 2000),
                'session_modification_rate': self.np_random.uniform(200, 800),
                'session_deletion_rate': self.np_random.uniform(400, 1500),
                'pfcp_message_type': self.np_random.randint(1, 255),
                'sequence_number': i + self.np_random.randint(0, 100000),
                'average_latency_ms': self.np_random.uniform(50, 500),
                'packet_loss_rate': self.np_random.uniform(0.1, 0.5),
                'jitter_ms': self.np_random.uniform(10, 100),
                'bandwidth_utilization': self.np_random.uniform(85, 100),
                'slice_type_encoded': self.np_random.randint(0, len(self.slice_types)),
                'qos_flow_identifier': self.np_random.randint(1, 64),
                'pdu_session_type': self.np_random.choice([0, 1, 2]),
                'dnn_encoded': self.np_random.randint(0, 20),
                'protocol_version': self.np_random.choice([1, 2, 3]),
                'message_length': self.np_random.randint(1000, 10000),
                'node_type_encoded': self.np_random.randint(0, len(self.node_types)),
                'encryption_strength': self.np_random.choice([128, 256, 512]),
                'authentication_method': self.np_random.randint(0, 10),
                'certificate_validity': self.np_random.randint(0, 365),
                'peak_hour_indicator': 1,
                'weekend_indicator': self.np_random.choice([0, 1]),
                'geographic_region': self.np_random.randint(0, 20),
                'cpu_utilization': self.np_random.uniform(80, 100),
                'memory_utilization': self.np_random.uniform(85, 100),
                'network_congestion_level': self.np_random.randint(3, 5),
                'handover_success_rate': self.np_random.uniform(0.5, 0.8),
                'beam_management_efficiency': self.np_random.uniform(0.3, 0.7),
                'massive_mimo_gain': self.np_random.uniform(5, 20),
                'carrier_aggregation_bands': self.np_random.randint(3, 8),
                'inter_arrival_time': self.np_random.uniform(0.001, 0.1),
                'burst_size': self.np_random.randint(50, 500),
                'flow_duration': self.np_random.uniform(300, 3600),
                'time_since_last_session': self.np_random.uniform(0.001, 1),
                'session_frequency': self.np_random.uniform(50, 200),
                'periodic_pattern_strength': self.np_random.uniform(0.8, 1.0),
                'user_experience_quality': self.np_random.uniform(1, 3),
                'network_reliability_score': self.np_random.uniform(0.5, 0.8),
                'service_availability': self.np_random.uniform(0.9, 0.99),
                'edge_processing_delay': self.np_random.uniform(5, 50),
                'cloud_offload_ratio': self.np_random.uniform(0.8, 1.0),
                'mec_utilization': self.np_random.uniform(0.8, 1.0),
            }
            
            # Apply stress-specific amplifications
            if stress_type == 'memory_exhaustion':
                sample['memory_utilization'] = self.np_random.uniform(95, 100)
                sample['session_establishment_rate'] *= self.np_random.uniform(5, 10)
                sample['message_length'] *= self.np_random.uniform(3, 8)
                
            elif stress_type == 'cpu_overload':
                sample['cpu_utilization'] = self.np_random.uniform(98, 100)
                sample['session_modification_rate'] *= self.np_random.uniform(8, 15)
                sample['burst_size'] *= self.np_random.uniform(4, 10)
                
            elif stress_type == 'network_saturation':
                sample['bandwidth_utilization'] = self.np_random.uniform(98, 100)
                sample['packet_loss_rate'] = self.np_random.uniform(0.3, 0.8)
                sample['average_latency_ms'] *= self.np_random.uniform(5, 20)
                sample['jitter_ms'] *= self.np_random.uniform(10, 50)
                
            elif stress_type == 'cascade_failure':
                # Multiple systems failing simultaneously
                sample['handover_success_rate'] = self.np_random.uniform(0.1, 0.4)
                sample['beam_management_efficiency'] = self.np_random.uniform(0.1, 0.3)
                sample['service_availability'] = self.np_random.uniform(0.8, 0.95)
                sample['user_experience_quality'] = self.np_random.uniform(1, 2)
                
            elif stress_type == 'race_condition':
                # Timing-related stress
                sample['inter_arrival_time'] = self.np_random.uniform(0.0001, 0.001)
                sample['time_since_last_session'] = self.np_random.uniform(0.0001, 0.01)
                sample['session_frequency'] *= self.np_random.uniform(20, 100)
                
            elif stress_type == 'buffer_overflow':
                # Large message/buffer scenarios
                sample['message_length'] = self.np_random.randint(5000, 50000)
                sample['burst_size'] = self.np_random.randint(200, 2000)
                sample['flow_duration'] = self.np_random.uniform(1800, 7200)
            
            # Calculate derived features
            sample['throughput_latency_ratio'] = sample['session_establishment_rate'] / max(sample['average_latency_ms'], 0.1)
            sample['efficiency_score'] = (sample['bandwidth_utilization'] * sample['handover_success_rate']) / 100
            sample['anomaly_score'] = abs(sample['cpu_utilization'] - 50) / 50
            
            sample['attack_type'] = 'Stress_' + stress_type
            sample['label'] = self.np_random.choice([0, 1, 2, 3, 4])  # Various stress-related labels
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def save_datasets(self, dataset, base_path="complex_5g_dataset"):
        """Save dataset in multiple formats for comprehensive testing"""
        
        print(f"\n💾 Saving complex dataset to multiple formats...")
        
        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        # 1. Full CSV dataset
        csv_path = os.path.join(base_path, "full_complex_dataset.csv")
        dataset.to_csv(csv_path, index=False)
        print(f"✅ Saved full dataset: {csv_path} ({len(dataset)} samples)")
        
        # 2. JSON format (subset)
        json_subset = dataset.sample(n=min(1000, len(dataset)), random_state=42)
        json_data = json_subset.to_dict('records')
        json_path = os.path.join(base_path, "complex_dataset_subset.json")
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✅ Saved JSON subset: {json_path} ({len(json_subset)} samples)")
        
        # 3. High-dimensional subset (for feature engineering stress test)
        high_dim_cols = [col for col in dataset.columns if col not in ['attack_type', 'label']]
        high_dim_data = dataset[high_dim_cols + ['label']].copy()
        
        # Add even more synthetic features to stress the system
        for i in range(20):
            high_dim_data[f'synthetic_feature_{i}'] = np.random.normal(0, 1, len(high_dim_data))
            high_dim_data[f'noise_feature_{i}'] = np.random.uniform(-1, 1, len(high_dim_data))
        
        high_dim_path = os.path.join(base_path, "high_dimensional_dataset.csv")
        high_dim_data.to_csv(high_dim_path, index=False)
        print(f"✅ Saved high-dimensional dataset: {high_dim_path} ({high_dim_data.shape[1]} features)")
        
        # 4. Attack-focused subset
        attack_data = dataset[dataset['label'] != 0].copy()
        attack_path = os.path.join(base_path, "attack_focused_dataset.csv")
        attack_data.to_csv(attack_path, index=False)
        print(f"✅ Saved attack-focused dataset: {attack_path} ({len(attack_data)} samples)")
        
        # 5. Edge cases subset
        edge_data = dataset[dataset['attack_type'].str.contains('Edge_Case|Stress_', na=False)].copy()
        edge_path = os.path.join(base_path, "edge_cases_dataset.csv")
        edge_data.to_csv(edge_path, index=False)
        print(f"✅ Saved edge cases dataset: {edge_path} ({len(edge_data)} samples)")
        
        return {
            'full_csv': csv_path,
            'json_subset': json_path,
            'high_dimensional': high_dim_path,
            'attack_focused': attack_path,
            'edge_cases': edge_path
        }

def main():
    """Generate and save complex 5G dataset"""
    
    print("🚀 ADVANCED 5G NETWORK TRAFFIC DATASET GENERATOR")
    print("=" * 60)
    print("Creating comprehensive dataset to stress-test the system...")
    
    # Generate large, complex dataset
    generator = Advanced5GDatasetGenerator(base_samples=10000)
    dataset = generator.generate_complex_dataset()
    
    # Display dataset statistics
    print("\n📊 DATASET STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(dataset)}")
    print(f"Total features: {len(dataset.columns)}")
    print(f"Attack distribution:")
    print(dataset['attack_type'].value_counts().head(10))
    print(f"\nLabel distribution:")
    print(dataset['label'].value_counts())
    
    # Check for various data quality aspects
    print(f"\nMissing values: {dataset.isnull().sum().sum()}")
    print(f"Duplicate rows: {dataset.duplicated().sum()}")
    print(f"Data types: {dataset.dtypes.value_counts().to_dict()}")
    
    # Save in multiple formats
    saved_files = generator.save_datasets(dataset)
    
    print("\n🎯 DATASET COMPLEXITY FEATURES")
    print("=" * 60)
    print("✅ Multi-modal distributions")
    print("✅ Extreme values and outliers")
    print("✅ Edge cases and boundary conditions")
    print("✅ Temporal anomalies")
    print("✅ Broken correlations")
    print("✅ Stress test scenarios")
    print("✅ High-dimensional feature space")
    print("✅ Multiple data formats")
    print("✅ Real-world 5G protocol characteristics")
    print("✅ Sophisticated attack patterns")
    
    print(f"\n🏁 Complex dataset generation complete!")
    print(f"Ready for comprehensive system testing...")
    
    return saved_files

if __name__ == "__main__":
    saved_files = main()
