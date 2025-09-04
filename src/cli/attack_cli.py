"""
Attack CLI Module

Handles adversarial attack generation, execution, and evaluation
for the 5G adversarial IDS system.

Author: Capstone Team
Date: September 3, 2025
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from src.cli.utils import (
    load_processed_data, load_model, save_results, 
    create_output_directory, print_experiment_header, 
    print_experiment_footer, format_performance_metrics,
    get_class_names, validate_model_file
)
from src.attacks.enhanced_attacks import EnhancedConstraintFGSM, EnhancedConstraintPGD, EnhancedAdversarialAttacks
from src.attacks.constraint_fgsm import ConstraintAwareFGSM
from src.attacks.attack_utils import evaluate_attack_success

class AttackCLI:
    """CLI interface for adversarial attack generation and evaluation."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.attack_config = config.get('attacks', {})
        self.data_config = config.get('data', {})
        self.models_config = config.get('models', {})
        
    def validate_attack_config(self) -> bool:
        """Validate attack-specific configuration."""
        # Check if target model is specified
        target_model = self.config.get('target_model') or self.attack_config.get('target_model')
        if not target_model:
            self.logger.error("❌ No target model specified")
            return False
        
        # Validate target model exists
        if not validate_model_file(target_model, self.logger):
            return False
        
        # Check attack types
        attack_types = self.attack_config.get('attack_types', [])
        if not attack_types:
            self.logger.warning("⚠️  No attack types specified, using default: enhanced_pgd")
            self.attack_config['attack_types'] = ['enhanced_pgd']
        
        return True
    
    def load_target_model_and_data(self) -> tuple:
        """Load target model, scaler, and test data."""
        # Load target model
        target_model_path = self.config.get('target_model') or self.attack_config.get('target_model')
        model = load_model(target_model_path, self.logger)
        
        # Load scaler
        scaler_path = self.attack_config.get('target_scaler', 'models/scaler.joblib')
        scaler = load_model(scaler_path, self.logger)
        
        # Load test data
        data_dir = self.data_config.get('processed_dir', 'data/processed')
        data = load_processed_data(data_dir, self.logger)
        
        return model, scaler, data
    
    def setup_attacks(self) -> Dict[str, Any]:
        """Initialize attack objects based on configuration."""
        attacks = {}
        
        attack_types = self.attack_config.get('attack_types', ['enhanced_pgd'])
        
        for attack_type in attack_types:
            if attack_type == 'enhanced_pgd':
                params = self.attack_config.get('enhanced_pgd', {})
                attacks[attack_type] = EnhancedConstraintPGD(model=self.model, **params)
                
            elif attack_type == 'enhanced_fgsm':
                params = self.attack_config.get('enhanced_fgsm', {})
                attacks[attack_type] = EnhancedConstraintFGSM(model=self.model, **params)
                
            elif attack_type == 'constraint_fgsm':
                params = self.attack_config.get('constraint_fgsm', {})
                attacks[attack_type] = ConstraintAwareFGSM(model=self.model, **params)
                
            else:
                self.logger.warning(f"⚠️  Unknown attack type: {attack_type}")
        
        self.logger.info(f"✅ Initialized {len(attacks)} attack types: {list(attacks.keys())}")
        return attacks
    
    def execute_attack(self, attack_name: str, attack_obj: Any, model: Any, 
                      scaler: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Execute a single adversarial attack."""
        self.logger.info(f"⚔️  Executing {attack_name} attack...")
        
        try:
            # Scale test data
            X_test_scaled = scaler.transform(X_test)
            
            # Generate adversarial examples
            start_time = time.time()
            attack_result = attack_obj.generate_adversarial_samples(X_test_scaled, y_test)
            attack_time = time.time() - start_time
            
            # Handle different return formats (some attacks return tuples)
            if isinstance(attack_result, tuple):
                X_adv, attack_info = attack_result
            else:
                X_adv = attack_result
                attack_info = {}
            
            # Evaluate attack success
            attack_results = evaluate_attack_success(
                model, X_test_scaled, X_adv, y_test, attack_name
            )
            
            # Add timing and configuration info
            attack_results.update({
                'attack_name': attack_name,
                'attack_time': attack_time,
                'samples_attacked': len(X_test),
                'adversarial_examples': X_adv,
                'original_samples': X_test_scaled
            })
            
            # Check PFCP constraint compliance if enabled
            if self.attack_config.get('pfcp_constraints', {}).get('enforce_protocol_compliance', False):
                compliance_rate = self.check_pfcp_compliance(X_test_scaled, X_adv)
                attack_results['pfcp_compliance_rate'] = compliance_rate
                
                self.logger.info(f"🔒 PFCP Compliance: {compliance_rate:.2%}")
            
            # Log attack summary
            evasion_rate = attack_results.get('overall_success_rate', 0)
            self.logger.info(f"✅ {attack_name} completed:")
            self.logger.info(f"   Evasion rate: {evasion_rate:.2%}")
            self.logger.info(f"   Attack time: {attack_time:.2f}s")
            self.logger.info(f"   Samples processed: {len(X_test)}")
            
            return attack_results
            
        except Exception as e:
            self.logger.error(f"❌ {attack_name} attack failed: {e}")
            raise
    
    def check_pfcp_compliance(self, X_original: np.ndarray, X_adv: np.ndarray) -> float:
        """Check PFCP protocol compliance for adversarial examples."""
        try:
            # Import PFCP constraints checker
            from src.attacks.pfcp_constraints import check_pfcp_compliance
            
            # Check compliance for all adversarial examples
            compliance_results = []
            for orig, adv in zip(X_original, X_adv):
                is_compliant = check_pfcp_compliance(orig, adv)
                compliance_results.append(is_compliant)
            
            compliance_rate = np.mean(compliance_results)
            return compliance_rate
            
        except Exception as e:
            self.logger.warning(f"⚠️  PFCP compliance check failed: {e}")
            return 1.0  # Assume compliant if check fails
    
    def analyze_attack_results(self, all_attack_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze and compare results across all attacks."""
        self.logger.info("📊 Analyzing attack results...")
        
        analysis = {
            'summary': {},
            'per_class_analysis': {},
            'attack_comparison': {},
            'vulnerability_analysis': {}
        }
        
        # Overall summary
        for attack_name, results in all_attack_results.items():
            analysis['summary'][attack_name] = {
                'evasion_rate': results.get('overall_success_rate', 0),
                'attack_time': results.get('attack_time', 0),
                'samples_attacked': results.get('samples_attacked', 0),
                'pfcp_compliance': results.get('pfcp_compliance_rate', 1.0)
            }
        
        # Per-class analysis
        class_names = get_class_names()
        for class_idx, class_name in enumerate(class_names):
            analysis['per_class_analysis'][class_name] = {}
            for attack_name, results in all_attack_results.items():
                per_class_results = results.get('per_class_results', {})
                if class_idx in per_class_results:
                    analysis['per_class_analysis'][class_name][attack_name] = \
                        per_class_results[class_idx].get('success_rate', 0)
        
        # Find most vulnerable classes
        vulnerability_scores = {}
        for class_name in class_names:
            class_vulnerabilities = []
            for attack_name in all_attack_results.keys():
                vuln_score = analysis['per_class_analysis'][class_name].get(attack_name, 0)
                class_vulnerabilities.append(vuln_score)
            
            vulnerability_scores[class_name] = np.mean(class_vulnerabilities) if class_vulnerabilities else 0
        
        analysis['vulnerability_analysis'] = {
            'most_vulnerable': max(vulnerability_scores, key=vulnerability_scores.get),
            'least_vulnerable': min(vulnerability_scores, key=vulnerability_scores.get),
            'vulnerability_scores': vulnerability_scores
        }
        
        # Best attack overall
        best_attack = max(analysis['summary'], key=lambda x: analysis['summary'][x]['evasion_rate'])
        analysis['attack_comparison']['best_attack'] = best_attack
        analysis['attack_comparison']['best_evasion_rate'] = analysis['summary'][best_attack]['evasion_rate']
        
        return analysis
    
    def save_attack_artifacts(self, all_attack_results: Dict[str, Dict], 
                            analysis: Dict[str, Any]) -> bool:
        """Save attack results, adversarial examples, and analysis."""
        try:
            # Create output directory
            output_dir = create_output_directory(
                self.config.get('output_dir', 'results'),
                'adversarial_attacks',
                self.logger
            )
            
            # Save complete attack results
            results_path = output_dir / 'attack_results.json'
            save_results(all_attack_results, str(results_path), self.logger)
            
            # Save analysis
            analysis_path = output_dir / 'attack_analysis.json'
            save_results(analysis, str(analysis_path), self.logger)
            
            # Save adversarial examples if enabled
            if self.attack_config.get('evaluation', {}).get('save_attack_samples', True):
                samples_dir = output_dir / 'adversarial_samples'
                samples_dir.mkdir(exist_ok=True)
                
                for attack_name, results in all_attack_results.items():
                    if 'adversarial_examples' in results:
                        adv_path = samples_dir / f'{attack_name}_adversarial.npy'
                        np.save(adv_path, results['adversarial_examples'])
                        
                        orig_path = samples_dir / f'{attack_name}_original.npy'
                        np.save(orig_path, results['original_samples'])
                
                self.logger.info(f"✅ Adversarial samples saved to: {samples_dir}")
            
            # Generate attack report
            if self.attack_config.get('evaluation', {}).get('generate_attack_reports', True):
                self.generate_attack_report(analysis, output_dir)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save attack artifacts: {e}")
            return False
    
    def generate_attack_report(self, analysis: Dict[str, Any], output_dir: Path):
        """Generate a comprehensive attack report."""
        report_path = output_dir / 'attack_report.md'
        
        try:
            with open(report_path, 'w') as f:
                f.write("# Adversarial Attack Evaluation Report\n\n")
                f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary table
                f.write("## Attack Summary\n\n")
                f.write("| Attack Type | Evasion Rate | Attack Time | PFCP Compliance |\n")
                f.write("|-------------|-------------|-------------|------------------|\n")
                
                for attack_name, summary in analysis['summary'].items():
                    evasion = summary['evasion_rate']
                    time_taken = summary['attack_time']
                    compliance = summary['pfcp_compliance']
                    f.write(f"| {attack_name} | {evasion:.2%} | {time_taken:.2f}s | {compliance:.2%} |\n")
                
                # Vulnerability analysis
                f.write("\n## Class Vulnerability Analysis\n\n")
                vuln_scores = analysis['vulnerability_analysis']['vulnerability_scores']
                f.write("| Class | Average Vulnerability |\n")
                f.write("|-------|---------------------|\n")
                for class_name, score in vuln_scores.items():
                    f.write(f"| {class_name} | {score:.2%} |\n")
                
                # Best attack
                best_attack = analysis['attack_comparison']['best_attack']
                best_rate = analysis['attack_comparison']['best_evasion_rate']
                f.write(f"\n## Best Attack: {best_attack}\n")
                f.write(f"**Evasion Rate**: {best_rate:.2%}\n\n")
                
                # Most vulnerable class
                most_vuln = analysis['vulnerability_analysis']['most_vulnerable']
                f.write(f"**Most Vulnerable Class**: {most_vuln}\n")
                
            self.logger.info(f"✅ Attack report generated: {report_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to generate attack report: {e}")
    
    def execute(self) -> bool:
        """Execute complete adversarial attack pipeline."""
        print_experiment_header("ADVERSARIAL ATTACK EVALUATION", self.config, self.logger)
        
        try:
            # Validate configuration
            if not self.validate_attack_config():
                return False
            
            # Load target model and data
            self.logger.info("📂 Loading target model and data...")
            model, scaler, data = self.load_target_model_and_data()
            
            # Set model for attack initialization
            self.model = model
            
            # Setup attacks
            attacks = self.setup_attacks()
            if not attacks:
                self.logger.error("❌ No valid attacks configured")
                return False
            
            # Execute all attacks
            all_attack_results = {}
            X_test = data['X_test']
            y_test = data['y_test']
            
            for attack_name, attack_obj in attacks.items():
                attack_results = self.execute_attack(
                    attack_name, attack_obj, model, scaler, X_test, y_test
                )
                all_attack_results[attack_name] = attack_results
            
            # Analyze results
            analysis = self.analyze_attack_results(all_attack_results)
            
            # Save artifacts
            if not self.save_attack_artifacts(all_attack_results, analysis):
                return False
            
            # Log final summary
            self.logger.info("⚔️  ATTACK EVALUATION SUMMARY:")
            for attack_name, summary in analysis['summary'].items():
                evasion = summary['evasion_rate']
                compliance = summary['pfcp_compliance']
                self.logger.info(f"   {attack_name}: {evasion:.2%} evasion, {compliance:.2%} compliant")
            
            best_attack = analysis['attack_comparison']['best_attack']
            best_rate = analysis['attack_comparison']['best_evasion_rate']
            self.logger.info(f"   Best attack: {best_attack} ({best_rate:.2%})")
            
            print_experiment_footer("ADVERSARIAL ATTACK EVALUATION", True, self.logger)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Attack evaluation pipeline failed: {e}")
            print_experiment_footer("ADVERSARIAL ATTACK EVALUATION", False, self.logger)
            return False
