#!/usr/bin/env python3
"""
Phase 2: Baseline Deep Learning Model Training
============================================

Industrial-grade execution script for training the baseline MLP model
with comprehensive monitoring and validation.

Security Classification: CONFIDENTIAL
Performance Target: ≥94% accuracy, ≤100ms inference time
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.models.baseline_model import BaselineTrainer
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
    log_filename = logs_dir / f"phase2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("Phase2Execution")
    logger.info(f"Logging initialized - Log file: {log_filename}")
    return logger


def validate_environment() -> bool:
    """Validate execution environment and prerequisites"""
    logger = logging.getLogger("Phase2Execution")
    
    try:
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
        
        # Check critical dependencies
        required_packages = ['torch', 'pandas', 'numpy', 'sklearn', 'tqdm', 'joblib']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            return False
        
        # Check GPU availability
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for training")
        
        # Check processed data availability
        data_dir = project_root / "data" / "processed"
        required_files = ["train.csv", "val.csv", "test.csv", "feature_columns.json"]
        
        for file in required_files:
            if not (data_dir / file).exists():
                logger.error(f"Required file not found: {data_dir / file}")
                return False
        
        logger.info("✅ Environment validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def execute_phase2_training() -> bool:
    """Execute Phase 2: Baseline Model Training"""
    logger = logging.getLogger("Phase2Execution")
    
    try:
        logger.info("=" * 80)
        logger.info("🚀 STARTING PHASE 2: BASELINE DEEP LEARNING MODEL TRAINING")
        logger.info("=" * 80)
        
        # Define paths
        data_dir = str(project_root / "data" / "processed")
        model_dir = str(project_root / "models" / "baseline")
        results_dir = str(project_root / "results" / "baseline")
        
        # Create output directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Model directory: {model_dir}")
        logger.info(f"Results directory: {results_dir}")
        
        # Training configuration
        training_config = {
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 50,
            'patience': 10,
            'min_delta': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip_value': 1.0,
            'num_workers': min(4, os.cpu_count())
        }
        
        logger.info("Training configuration:")
        for key, value in training_config.items():
            logger.info(f"  {key}: {value}")
        
        # Initialize trainer
        trainer = BaselineTrainer(config=training_config)
        
        # Execute training
        logger.info("\n🎯 Starting training process...")
        training_start_time = datetime.now()
        
        results = trainer.train(data_dir, model_dir, results_dir)
        
        training_end_time = datetime.now()
        training_duration = training_end_time - training_start_time
        
        # Log final results
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PHASE 2 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        
        final_metrics = results['final_test_metrics']
        logger.info(f"📊 FINAL PERFORMANCE METRICS:")
        logger.info(f"  🎯 Accuracy: {final_metrics['accuracy']:.4f}")
        logger.info(f"  🎯 F1 Score: {final_metrics['f1']:.4f}")
        logger.info(f"  🎯 Precision: {final_metrics['precision']:.4f}")
        logger.info(f"  🎯 Recall: {final_metrics['recall']:.4f}")
        logger.info(f"  ⚡ Inference Time: {final_metrics['avg_inference_time_ms']:.2f}ms")
        logger.info(f"  🕒 Training Duration: {training_duration}")
        
        # Performance validation
        success_criteria = []
        
        # Check accuracy target
        if final_metrics['accuracy'] >= 0.94:
            logger.info("✅ ACCURACY TARGET ACHIEVED! (≥94%)")
            success_criteria.append(True)
        else:
            logger.warning(f"❌ Accuracy target not met: {final_metrics['accuracy']:.4f} < 0.94")
            success_criteria.append(False)
        
        # Check inference time target
        if final_metrics['avg_inference_time_ms'] <= 100:
            logger.info("✅ INFERENCE TIME TARGET ACHIEVED! (≤100ms)")
            success_criteria.append(True)
        else:
            logger.warning(f"❌ Inference time target not met: {final_metrics['avg_inference_time_ms']:.2f}ms > 100ms")
            success_criteria.append(False)
        
        # Overall success assessment
        overall_success = all(success_criteria)
        
        if overall_success:
            logger.info("🏆 ALL PERFORMANCE TARGETS ACHIEVED!")
            logger.info("📈 Ready for Phase 3: Adversarial Attack Implementation")
        else:
            logger.warning("⚠️  Some performance targets not met - consider hyperparameter tuning")
        
        # Save phase completion status
        phase_status = {
            'phase': 2,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'performance_targets_met': overall_success,
            'final_metrics': final_metrics,
            'training_duration_seconds': training_duration.total_seconds(),
            'next_phase': 'Phase 3: Adversarial Attack Implementation'
        }
        
        status_file = results_dir + "/phase2_status.json"
        with open(status_file, 'w') as f:
            json.dump(phase_status, f, indent=2)
        
        logger.info(f"📋 Phase status saved: {status_file}")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"Phase 2 execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main execution function"""
    # Setup logging
    logger = setup_logging()
    
    try:
        logger.info("🔧 Phase 2 Execution Started")
        
        # Validate environment
        if not validate_environment():
            logger.error("❌ Environment validation failed")
            return 1
        
        # Execute Phase 2 training
        success = execute_phase2_training()
        
        if success:
            logger.info("🎉 Phase 2 completed successfully!")
            logger.info("🚀 Ready to proceed to Phase 3: Adversarial Attacks")
            return 0
        else:
            logger.error("❌ Phase 2 completed with issues")
            return 1
            
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