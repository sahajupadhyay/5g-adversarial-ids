"""
Detect command for the Adversarial 5G IDS CLI.

Provides threat detection capabilities using baseline and robust models.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Dict, List, Any
import argparse

from src.cli.utils.output import CLIOutput, Icons, Colors, ProgressBar

class DetectCommand:
    """Command for threat detection using trained models."""
    
    def __init__(self):
        self.output = CLIOutput()
    
    def add_parser(self, subparsers):
        """Add detect command parser."""
        parser = subparsers.add_parser(
            'detect',
            help='Detect threats in 5G PFCP traffic',
            description='Analyze network traffic for potential security threats using trained ML models.'
        )
        
        parser.add_argument(
            '--data', '-d',
            type=str,
            required=True,
            help='Path to input data file (CSV, NPY, or use "sample" for test data)'
        )
        
        parser.add_argument(
            '--model', '-m',
            choices=['baseline', 'robust', 'both'],
            default='robust',
            help='Model to use for detection (default: robust)'
        )
        
        parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output file for results (JSON format)'
        )
        
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.5,
            help='Detection threshold for probability scores (default: 0.5)'
        )
        
        parser.add_argument(
            '--detailed',
            action='store_true',
            help='Show detailed analysis for each sample'
        )
        
        parser.add_argument(
            '--batch-size',
            type=int,
            default=100,
            help='Batch size for processing (default: 100)'
        )
        
        return parser
    
    def execute(self, args, config, output):
        """Execute the detect command."""
        self.output = output
        self.config = config
        
        try:
            # Load data
            self.output.header("🔍 5G PFCP Threat Detection", "Analyzing network traffic for security threats")
            X, true_labels = self._load_data(args.data)
            
            # Load models
            models = self._load_models(args.model)
            
            # Run detection
            results = self._run_detection(X, models, args.threshold, args.batch_size, true_labels)
            
            # Display results
            self._display_results(results, args.detailed)
            
            # Save results if requested
            if args.output:
                self._save_results(results, args.output)
            
            return 0
            
        except Exception as e:
            self.output.error(f"Detection failed: {str(e)}")
            return 1
    
    def _load_data(self, data_path: str) -> tuple:
        """Load data from various sources."""
        self.output.info(f"Loading data from: {data_path}")
        
        if data_path.lower() == 'sample':
            # Load test data
            test_data_path = self.config.get('paths.test_data')
            test_labels_path = self.config.get('paths.test_labels')
            
            if not Path(test_data_path).exists() or not Path(test_labels_path).exists():
                raise FileNotFoundError("Sample test data not found. Run baseline model training first.")
            
            X = np.load(test_data_path)
            y = np.load(test_labels_path)
            
            self.output.success(f"Loaded {len(X)} samples from test dataset")
            return X, y
        
        elif data_path.endswith('.csv'):
            # Load CSV file
            df = pd.read_csv(data_path)
            
            # Assume last column is labels if present, otherwise None
            if 'label' in df.columns or 'class' in df.columns:
                label_col = 'label' if 'label' in df.columns else 'class'
                y = df[label_col].values
                X = df.drop(columns=[label_col]).values
            else:
                X = df.values
                y = None
            
            self.output.success(f"Loaded {len(X)} samples from CSV file")
            return X, y
        
        elif data_path.endswith('.npy'):
            # Load NumPy array
            X = np.load(data_path)
            y = None
            
            self.output.success(f"Loaded {len(X)} samples from NumPy file")
            return X, y
        
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    def _load_models(self, model_choice: str) -> Dict[str, Any]:
        """Load the specified models."""
        models = {}
        
        if model_choice in ['baseline', 'both']:
            baseline_path = self.config.get('paths.baseline_model')
            if Path(baseline_path).exists():
                models['baseline'] = {
                    'model': joblib.load(baseline_path),
                    'info': self.config.get_model_info('baseline'),
                    'transformers': self._load_transformers()
                }
                self.output.success(f"Loaded baseline model from {baseline_path}")
            else:
                self.output.warning(f"Baseline model not found: {baseline_path}")
        
        if model_choice in ['robust', 'both']:
            robust_path = self.config.get('paths.robust_model')
            if Path(robust_path).exists():
                models['robust'] = {
                    'model': joblib.load(robust_path),
                    'info': self.config.get_model_info('robust'),
                    'transformers': None  # Robust model uses raw features
                }
                self.output.success(f"Loaded robust model from {robust_path}")
            else:
                self.output.warning(f"Robust model not found: {robust_path}")
        
        if not models:
            raise ValueError("No models could be loaded")
        
        return models
    
    def _load_transformers(self) -> Optional[Any]:
        """Load feature transformers for baseline model."""
        transformers_path = self.config.get('paths.transformers')
        if Path(transformers_path).exists():
            return joblib.load(transformers_path)
        return None
    
    def _run_detection(self, X: np.ndarray, models: Dict, threshold: float, 
                      batch_size: int, true_labels: Optional[np.ndarray]) -> Dict[str, Any]:
        """Run threat detection using loaded models."""
        results = {
            'total_samples': len(X),
            'models': {},
            'summary': {},
            'detailed_results': []
        }
        
        for model_name, model_data in models.items():
            self.output.subheader(f"Running {model_name.title()} Model Detection")
            
            model = model_data['model']
            transformers = model_data['transformers']
            
            # Transform features if needed
            if transformers is not None:
                self.output.verbose("Applying feature transformations...")
                X_transformed = transformers.transform(X)
            else:
                X_transformed = X
            
            # Progress bar for detection
            progress = ProgressBar(len(X), f"Detecting threats with {model_name} model")
            
            predictions = []
            probabilities = []
            
            # Process in batches
            for i in range(0, len(X_transformed), batch_size):
                batch = X_transformed[i:i+batch_size]
                
                # Get predictions and probabilities
                batch_pred = model.predict(batch)
                batch_prob = model.predict_proba(batch)
                
                predictions.extend(batch_pred)
                probabilities.extend(batch_prob)
                
                progress.update(len(batch))
            
            progress.finish(f"{model_name.title()} detection complete")
            
            # Analyze results
            model_results = self._analyze_predictions(
                predictions, probabilities, threshold, true_labels, model_name
            )
            
            results['models'][model_name] = model_results
        
        # Generate summary
        results['summary'] = self._generate_summary(results['models'])
        
        return results
    
    def _analyze_predictions(self, predictions: List, probabilities: List, 
                           threshold: float, true_labels: Optional[np.ndarray],
                           model_name: str) -> Dict[str, Any]:
        """Analyze model predictions."""
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate threat detection metrics
        threat_count = np.sum(predictions != 4)  # Class 4 is 'Normal'
        normal_count = np.sum(predictions == 4)
        
        # Get class distribution
        unique, counts = np.unique(predictions, return_counts=True)
        class_distribution = {}
        for class_id, count in zip(unique, counts):
            class_name = self.config.get_class_name(int(class_id))
            class_distribution[class_name] = int(count)
        
        # Calculate confidence statistics
        max_probs = np.max(probabilities, axis=1)
        avg_confidence = float(np.mean(max_probs))
        min_confidence = float(np.min(max_probs))
        max_confidence = float(np.max(max_probs))
        
        # High-confidence threats (above threshold)
        high_conf_threats = np.sum((predictions != 4) & (max_probs >= threshold))
        
        results = {
            'model_name': model_name,
            'total_samples': len(predictions),
            'threats_detected': int(threat_count),
            'normal_traffic': int(normal_count),
            'threat_rate': float(threat_count / len(predictions)),
            'high_confidence_threats': int(high_conf_threats),
            'class_distribution': class_distribution,
            'confidence_stats': {
                'average': avg_confidence,
                'minimum': min_confidence,
                'maximum': max_confidence
            },
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        
        # Add accuracy if true labels are available
        if true_labels is not None:
            accuracy = float(np.mean(predictions == true_labels))
            results['accuracy'] = accuracy
            
            # Confusion matrix-like statistics
            tp = np.sum((predictions != 4) & (true_labels != 4))  # True threats
            fp = np.sum((predictions != 4) & (true_labels == 4))  # False alarms
            tn = np.sum((predictions == 4) & (true_labels == 4))  # True normal
            fn = np.sum((predictions == 4) & (true_labels != 4))  # Missed threats
            
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            
            results['classification_metrics'] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        
        return results
    
    def _generate_summary(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary of detection results."""
        summary = {
            'models_used': list(model_results.keys()),
            'total_samples': 0,
            'consensus': {}
        }
        
        if model_results:
            # Get sample count from first model
            first_model = list(model_results.values())[0]
            summary['total_samples'] = first_model['total_samples']
            
            # If multiple models, calculate consensus
            if len(model_results) > 1:
                baseline_threats = set()
                robust_threats = set()
                
                for i, pred in enumerate(model_results.get('baseline', {}).get('predictions', [])):
                    if pred != 4:  # Not normal
                        baseline_threats.add(i)
                
                for i, pred in enumerate(model_results.get('robust', {}).get('predictions', [])):
                    if pred != 4:  # Not normal
                        robust_threats.add(i)
                
                if baseline_threats and robust_threats:
                    consensus_threats = baseline_threats.intersection(robust_threats)
                    disagreement = baseline_threats.symmetric_difference(robust_threats)
                    
                    summary['consensus'] = {
                        'agreed_threats': len(consensus_threats),
                        'disagreed_samples': len(disagreement),
                        'consensus_rate': len(consensus_threats) / max(len(baseline_threats), len(robust_threats))
                    }
        
        return summary
    
    def _display_results(self, results: Dict[str, Any], detailed: bool):
        """Display detection results."""
        
        # Overall summary
        self.output.header("🎯 Detection Results Summary")
        
        total_samples = results['total_samples']
        self.output.metric("Total Samples Analyzed", str(total_samples))
        self.output.metric("Models Used", ", ".join(results['models'].keys()))
        
        # Model-specific results
        for model_name, model_results in results['models'].items():
            self.output.subheader(f"{model_name.title()} Model Results")
            
            threats = model_results['threats_detected']
            threat_rate = model_results['threat_rate'] * 100
            high_conf = model_results['high_confidence_threats']
            avg_conf = model_results['confidence_stats']['average'] * 100
            
            self.output.bullet(f"Threats Detected: {Colors.RED}{threats}{Colors.RESET} ({threat_rate:.1f}%)")
            self.output.bullet(f"High-Confidence Threats: {Colors.YELLOW}{high_conf}{Colors.RESET}")
            self.output.bullet(f"Average Confidence: {Colors.CYAN}{avg_conf:.1f}%{Colors.RESET}")
            
            # Accuracy if available
            if 'accuracy' in model_results:
                accuracy = model_results['accuracy'] * 100
                self.output.bullet(f"Accuracy: {Colors.GREEN}{accuracy:.1f}%{Colors.RESET}")
            
            # Class distribution
            self.output.blank_line()
            self.output.info("Threat Type Distribution:")
            for class_name, count in model_results['class_distribution'].items():
                percentage = (count / total_samples) * 100
                color = Colors.RED if class_name != 'Normal' else Colors.GREEN
                self.output.bullet(f"{class_name}: {color}{count}{Colors.RESET} ({percentage:.1f}%)")
        
        # Consensus results if multiple models
        if len(results['models']) > 1 and results['summary'].get('consensus'):
            self.output.subheader("Model Consensus Analysis")
            consensus = results['summary']['consensus']
            
            agreed = consensus['agreed_threats']
            disagreed = consensus['disagreed_samples']
            consensus_rate = consensus['consensus_rate'] * 100
            
            self.output.bullet(f"Agreed Threats: {Colors.GREEN}{agreed}{Colors.RESET}")
            self.output.bullet(f"Disagreed Samples: {Colors.YELLOW}{disagreed}{Colors.RESET}")
            self.output.bullet(f"Consensus Rate: {Colors.CYAN}{consensus_rate:.1f}%{Colors.RESET}")
        
        # Detailed results if requested
        if detailed:
            self._display_detailed_results(results)
    
    def _display_detailed_results(self, results: Dict[str, Any]):
        """Display detailed per-sample results."""
        self.output.subheader("Detailed Sample Analysis")
        
        # Show first 10 samples as example
        sample_count = min(10, results['total_samples'])
        
        for model_name, model_results in results['models'].items():
            self.output.info(f"{model_name.title()} Model - First {sample_count} Samples:")
            
            # Table header
            widths = [8, 15, 12, 12]
            self.output.table_row(['Sample', 'Prediction', 'Confidence', 'Threat'], widths)
            self.output.table_separator(widths)
            
            predictions = model_results['predictions']
            probabilities = model_results['probabilities']
            
            for i in range(sample_count):
                pred_id = predictions[i]
                pred_name = self.config.get_class_name(pred_id)
                confidence = max(probabilities[i]) * 100
                is_threat = "Yes" if pred_id != 4 else "No"
                
                threat_color = Colors.RED if is_threat == "Yes" else Colors.GREEN
                
                self.output.table_row([
                    str(i),
                    pred_name,
                    f"{confidence:.1f}%",
                    f"{threat_color}{is_threat}{Colors.RESET}"
                ], widths)
            
            self.output.blank_line()
    
    def _save_results(self, results: Dict[str, Any], output_path: str):
        """Save results to file."""
        import json
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.output.success(f"Results saved to: {output_path}")
            
        except Exception as e:
            self.output.error(f"Failed to save results: {str(e)}")
