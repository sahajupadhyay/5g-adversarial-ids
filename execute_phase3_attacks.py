#!/usr/bin/env python3
"""
Phase 3: Adversarial Attack Generation
====================================

Industrial-grade execution script for generating comprehensive adversarial
attacks against the trained baseline IDS model.

Security Classification: CONFIDENTIAL
Attack Methods: FGSM, PGD with adaptive epsilon strategies
"""

import os
import sys
import logging
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.attacks.adversarial_attacks import AdversarialAttacker
    from src.models.baseline_model import BaselineIDS, IDSDataset, BaselineTrainer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure you're running from the project root directory")
    sys.exit(1)


def setup_logging() -> logging.Logger:
    """Setup industrial-grade logging"""
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Setup logging configuration
    log_filename = logs_dir / f"phase3_attacks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("Phase3Execution")
    logger.info(f"Logging initialized - Log file: {log_filename}")
    return logger


def validate_environment() -> bool:
    """Validate execution environment and prerequisites"""
    logger = logging.getLogger("Phase3Execution")
    
    try:
        # Check critical dependencies
        required_packages = ['torch', 'torchattacks', 'pandas', 'numpy', 'sklearn', 'tqdm']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            return False
        
        # Check baseline model exists
        model_path = project_root / "models" / "baseline" / "baseline_model.pt"
        if not model_path.exists():
            logger.error(f"Baseline model not found: {model_path}")
            return False
        
        # Check processed data exists
        data_dir = project_root / "data" / "processed"
        required_files = ["test.csv", "feature_columns.json"]
        
        for file in required_files:
            if not (data_dir / file).exists():
                logger.error(f"Required file not found: {data_dir / file}")
                return False
        
        logger.info("✅ Environment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def load_baseline_model() -> tuple:
    """Load the trained baseline model"""
    import torch
    logger = logging.getLogger("Phase3Execution")
    
    try:
        # Model paths
        model_path = project_root / "models" / "baseline" / "baseline_model.pt"
        data_dir = project_root / "data" / "processed"
        
        # Load feature schema to determine input size
        feature_columns_path = data_dir / "feature_columns.json"
        with open(feature_columns_path, 'r') as f:
            feature_columns = json.load(f)
        
        input_features = len(feature_columns)
        
        # Initialize model architecture
        model = BaselineIDS(
            input_features=input_features,
            hidden_layers=[256, 128, 64],
            dropout_rate=0.5,
            use_batch_norm=True,
            activation='relu'
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"✅ Baseline model loaded: {input_features} features")
        logger.info(f"Model metrics: {checkpoint.get('metrics', 'Not available')}")
        
        return model, input_features
        
    except Exception as e:
        logger.error(f"Failed to load baseline model: {e}")
        raise


def prepare_test_data() -> object:
    """Prepare test data loader"""
    logger = logging.getLogger("Phase3Execution")
    
    try:
        data_dir = str(project_root / "data" / "processed")
        test_path = os.path.join(data_dir, "test.csv")
        columns_path = os.path.join(data_dir, "feature_columns.json")
        
        # Create test dataset
        test_dataset = IDSDataset(test_path, columns_path, validate_data=True)
        
        # Create data loader with industrial-grade configuration
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,  # Smaller batch size for attack generation
            shuffle=False,
            num_workers=2,
            pin_memory=False  # CPU-friendly
        )
        
        logger.info(f"✅ Test data prepared: {len(test_dataset)} samples")
        return test_loader
        
    except Exception as e:
        logger.error(f"Failed to prepare test data: {e}")
        raise


def execute_adversarial_attacks(model, test_loader) -> dict:
    """Execute comprehensive adversarial attacks"""
    logger = logging.getLogger("Phase3Execution")
    
    try:
        logger.info("🔥 Initializing adversarial attacker...")
        
        # Initialize attacker
        attacker = AdversarialAttacker(
            model=model,
            device='cpu',  # Use CPU for compatibility
            validate_attacks=True
        )
        
        # Attack configurations
        attack_configs = [
            {'type': 'fgsm', 'epsilon': 0.01, 'name': 'FGSM_Conservative'},
            {'type': 'fgsm', 'epsilon': 0.05, 'name': 'FGSM_Standard'},
            {'type': 'fgsm', 'epsilon': 0.1, 'name': 'FGSM_Aggressive'},
            {'type': 'pgd', 'epsilon': 0.01, 'name': 'PGD_Conservative'},
            {'type': 'pgd', 'epsilon': 0.05, 'name': 'PGD_Standard'},
            {'type': 'pgd', 'epsilon': 0.1, 'name': 'PGD_Aggressive'}
        ]
        
        attack_results = {}
        
        # Execute each attack configuration
        for config in attack_configs:
            logger.info(f"\n🚀 Executing {config['name']} (ε={config['epsilon']})...")
            
            try:
                if config['type'] == 'fgsm':
                    adversarial_df = attacker.generate_fgsm_attacks(
                        test_loader, 
                        epsilon=config['epsilon']
                    )
                elif config['type'] == 'pgd':
                    adversarial_df = attacker.generate_pgd_attacks(
                        test_loader,
                        epsilon=config['epsilon'],
                        alpha=config['epsilon']/10,  # alpha = epsilon/10
                        steps=10
                    )
                
                attack_results[config['name']] = {
                    'dataframe': adversarial_df,
                    'config': config,
                    'sample_count': len(adversarial_df)
                }
                
                logger.info(f"✅ {config['name']} completed: {len(adversarial_df)} adversarial samples")
                
            except Exception as e:
                logger.error(f"❌ {config['name']} failed: {e}")
                continue
        
        logger.info(f"\n🎉 Attack generation completed: {len(attack_results)} successful configurations")
        return attack_results, attacker
        
    except Exception as e:
        logger.error(f"Attack execution failed: {e}")
        raise


def evaluate_attack_effectiveness(attack_results, attacker, original_test_data, model):
    """Evaluate the effectiveness of generated attacks"""
    import pandas as pd
    logger = logging.getLogger("Phase3Execution")
    
    logger.info("\n📊 Evaluating attack effectiveness...")
    
    evaluation_results = {}
    
    # Load original test data for comparison
    test_path = project_root / "data" / "processed" / "test.csv"
    original_df = pd.read_csv(test_path)
    
    for attack_name, attack_data in attack_results.items():
        logger.info(f"\n🔍 Evaluating {attack_name}...")
        
        try:
            adversarial_df = attack_data['dataframe']
            
            # Evaluate attack effectiveness
            effectiveness = attacker.evaluate_attack_effectiveness(
                original_df=original_df,
                adversarial_df=adversarial_df,
                model=model
            )
            
            evaluation_results[attack_name] = {
                'config': attack_data['config'],
                'effectiveness': effectiveness,
                'sample_count': attack_data['sample_count']
            }
            
            # Log key metrics
            logger.info(f"  📉 Accuracy Drop: {effectiveness['accuracy_drop']:.4f}")
            logger.info(f"  🎯 Attack Success Rate: {effectiveness['attack_success_rate']:.4f}")
            logger.info(f"  📏 Avg L2 Perturbation: {effectiveness['average_l2_perturbation']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for {attack_name}: {e}")
            continue
    
    return evaluation_results


def save_attack_results(attack_results, evaluation_results, attacker):
    """Save all attack results and evaluations"""
    logger = logging.getLogger("Phase3Execution")
    
    try:
        # Create results directories
        results_dir = project_root / "results" / "adversarial_attacks"
        data_dir = results_dir / "datasets"
        analysis_dir = results_dir / "analysis"
        
        results_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(exist_ok=True)
        analysis_dir.mkdir(exist_ok=True)
        
        # Save adversarial datasets
        for attack_name, attack_data in attack_results.items():
            dataset_path = data_dir / f"{attack_name.lower()}_adversarial_samples.csv"
            attack_data['dataframe'].to_csv(dataset_path, index=False)
            logger.info(f"💾 Saved {attack_name} dataset: {dataset_path}")
        
        # Save evaluation results
        evaluation_path = analysis_dir / "attack_effectiveness_evaluation.json"
        
        # Prepare serializable evaluation data
        serializable_results = {}
        for attack_name, results in evaluation_results.items():
            serializable_results[attack_name] = {
                'config': results['config'],
                'sample_count': results['sample_count'],
                'effectiveness_metrics': results['effectiveness']
            }
        
        with open(evaluation_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📊 Saved evaluation results: {evaluation_path}")
        
        # Save attack summary with proper serialization
        attack_summary = attacker.get_attack_summary()
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        serializable_attack_summary = convert_numpy_types(attack_summary)
        summary_path = analysis_dir / "attack_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(serializable_attack_summary, f, indent=2)
        
        logger.info(f"📋 Saved attack summary: {summary_path}")
        
        # Save comprehensive report
        report = {
            'phase': 3,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'attack_configurations': len(attack_results),
            'total_adversarial_samples': sum(r['sample_count'] for r in attack_results.values()),
            'attack_summary': serializable_attack_summary,
            'evaluation_results': serializable_results,
            'next_phase': 'Phase 4: Adversarial Defense Implementation'
        }
        
        report_path = results_dir / "phase3_comprehensive_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Saved comprehensive report: {report_path}")
        
        return report_path
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise


def main():
    """Main execution function"""
    # Setup logging
    logger = setup_logging()
    
    try:
        logger.info("=" * 80)
        logger.info("🔥 PHASE 3: ADVERSARIAL ATTACK GENERATION")
        logger.info("=" * 80)
        
        # Validate environment
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            return 1
        
        # Import required packages after validation
        import torch
        import pandas as pd
        
        # Load baseline model
        model, input_features = load_baseline_model()
        
        # Prepare test data
        test_loader = prepare_test_data()
        
        # Execute adversarial attacks
        attack_results, attacker = execute_adversarial_attacks(model, test_loader)
        
        if not attack_results:
            logger.error("❌ No attacks were successfully generated")
            return 1
        
        # Evaluate attack effectiveness
        evaluation_results = evaluate_attack_effectiveness(
            attack_results, attacker, test_loader, model
        )
        
        # Save all results
        report_path = save_attack_results(attack_results, evaluation_results, attacker)
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PHASE 3 COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        total_samples = sum(r['sample_count'] for r in attack_results.values())
        successful_configs = len(attack_results)
        
        logger.info(f"📊 ATTACK GENERATION SUMMARY:")
        logger.info(f"  🎯 Successful Attack Configurations: {successful_configs}")
        logger.info(f"  💥 Total Adversarial Samples Generated: {total_samples}")
        logger.info(f"  📄 Comprehensive Report: {report_path}")
        
        # Display top attack effectiveness
        if evaluation_results:
            logger.info(f"\n🏆 MOST EFFECTIVE ATTACKS:")
            sorted_attacks = sorted(
                evaluation_results.items(),
                key=lambda x: x[1]['effectiveness']['attack_success_rate'],
                reverse=True
            )
            
            for i, (attack_name, results) in enumerate(sorted_attacks[:3]):
                effectiveness = results['effectiveness']
                logger.info(f"  {i+1}. {attack_name}:")
                logger.info(f"     Success Rate: {effectiveness['attack_success_rate']:.4f}")
                logger.info(f"     Accuracy Drop: {effectiveness['accuracy_drop']:.4f}")
        
        logger.info("\n🚀 Ready for Phase 4: Adversarial Defense Implementation")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Execution interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)