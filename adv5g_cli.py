#!/usr/bin/env python3
"""
5G Adversarial IDS - Master Command Line Interface

This is the single entry point for the entire 5G adversarial intrusion detection system.
Orchestrates baseline training, adversarial attacks, defense hardening, and evaluation.

Usage:
    python adv5g_cli.py --mode baseline --config configs/baseline.yaml
    python adv5g_cli.py --mode attack --config configs/attack.yaml --target models/rf_baseline.joblib
    python adv5g_cli.py --mode defense --config configs/defense.yaml
    python adv5g_cli.py --mode evaluate --config configs/evaluation.yaml
    python adv5g_cli.py --mode pipeline --config configs/full_pipeline.yaml

Author: Capstone Team
Date: September 3, 2025
"""

import argparse
import sys
import logging
import yaml
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Import CLI modules (to be created)
from src.cli.baseline_cli import BaselineCLI
from src.cli.attack_cli import AttackCLI
from src.cli.defense_cli import DefenseCLI
from src.cli.evaluation_cli import EvaluationCLI
from src.cli.pipeline_cli import PipelineCLI
from src.cli.universal_data_cli import integrate_with_existing_pipeline, UNIVERSAL_PROCESSING_CONFIG
from src.cli.utils import setup_logging, validate_config, check_dependencies

# Version and metadata
__version__ = "0.3.0"
__status__ = "Phase 3 - System Integration"

class Adv5GCLI:
    """Master CLI orchestrator for 5G Adversarial IDS system."""
    
    def __init__(self):
        self.logger = None
        self.config = None
        self.start_time = None
        
    def setup_logging(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """Initialize comprehensive logging system."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Default log file with timestamp
        if log_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = logs_dir / f"adv5g_{timestamp}.log"
        
        # Configure logging with UTF-8 encoding
        import io
        
        # Create stream handler with UTF-8 encoding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Create file handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            handlers=[file_handler, console_handler]
        )
        
        self.logger = logging.getLogger("Adv5G-CLI")
        self.logger.info(f"[STARTUP] 5G Adversarial IDS CLI v{__version__} - {__status__}")
        self.logger.info(f"[LOGGING] Logging to: {log_file}")
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"✅ Configuration loaded from: {config_path}")
            
            # Validate configuration structure
            validate_config(config, self.logger)
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            sys.exit(1)
    
    def run_baseline(self, config: Dict[str, Any]) -> bool:
        """Execute baseline IDS training and evaluation."""
        self.logger.info("🏗️  Starting baseline IDS training...")
        
        try:
            baseline_cli = BaselineCLI(config, self.logger)
            success = baseline_cli.execute()
            
            if success:
                self.logger.info("✅ Baseline training completed successfully")
            else:
                self.logger.error("❌ Baseline training failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Baseline execution error: {e}")
            return False
    
    def run_attack(self, config: Dict[str, Any]) -> bool:
        """Execute adversarial attack generation and evaluation."""
        self.logger.info("⚔️  Starting adversarial attack execution...")
        
        try:
            attack_cli = AttackCLI(config, self.logger)
            success = attack_cli.execute()
            
            if success:
                self.logger.info("✅ Adversarial attacks completed successfully")
            else:
                self.logger.error("❌ Adversarial attacks failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Attack execution error: {e}")
            return False
    
    def run_defense(self, config: Dict[str, Any]) -> bool:
        """Execute adversarial defense training and hardening."""
        self.logger.info("🛡️  Starting adversarial defense training...")
        
        try:
            defense_cli = DefenseCLI(config, self.logger)
            success = defense_cli.execute()
            
            if success:
                self.logger.info("✅ Defense training completed successfully")
            else:
                self.logger.error("❌ Defense training failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Defense execution error: {e}")
            return False
    
    def run_evaluation(self, config: Dict[str, Any]) -> bool:
        """Execute comprehensive system evaluation."""
        self.logger.info("📊 Starting comprehensive evaluation...")
        
        try:
            eval_cli = EvaluationCLI(config, self.logger)
            success = eval_cli.execute()
            
            if success:
                self.logger.info("✅ Evaluation completed successfully")
            else:
                self.logger.error("❌ Evaluation failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation execution error: {e}")
            return False
    
    def run_pipeline(self, config: Dict[str, Any]) -> bool:
        """Execute full end-to-end pipeline."""
        self.logger.info("🔄 Starting full pipeline execution...")
        
        try:
            pipeline_cli = PipelineCLI(config, self.logger)
            success = pipeline_cli.execute()
            
            if success:
                self.logger.info("✅ Full pipeline completed successfully")
            else:
                self.logger.error("❌ Full pipeline failed")
                
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution error: {e}")
            return False
    
    def print_summary(self):
        """Print execution summary and statistics."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.info(f"⏱️  Total execution time: {duration:.2f} seconds")
        
        self.logger.info("🏁 Execution completed")
        
    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="5G Adversarial IDS - Master CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python adv5g_cli.py --mode baseline --config configs/baseline.yaml
  python adv5g_cli.py --mode attack --config configs/attack.yaml --target models/rf_baseline.joblib
  python adv5g_cli.py --mode defense --config configs/defense.yaml
  python adv5g_cli.py --mode evaluate --config configs/evaluation.yaml
  python adv5g_cli.py --mode pipeline --config configs/full_pipeline.yaml --verbose
            """
        )
        
        parser.add_argument(
            "--mode",
            required=True,
            choices=["baseline", "attack", "defense", "evaluate", "pipeline", "process_data"],
            help="Execution mode"
        )
        
        parser.add_argument(
            "--config",
            required=True,
            help="Path to YAML configuration file"
        )
        
        parser.add_argument(
            "--target",
            help="Target model file for attacks (required for attack mode)"
        )
        
        parser.add_argument(
            "--data",
            help="Path to data file/directory for processing (required for process_data mode)"
        )
        
        parser.add_argument(
            "--output-dir",
            default="results",
            help="Output directory for results (default: results)"
        )
        
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level (default: INFO)"
        )
        
        parser.add_argument(
            "--log-file",
            help="Custom log file path"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )
        
        parser.add_argument(
            "--version",
            action="version",
            version=f"5G Adversarial IDS CLI v{__version__}"
        )
        
        args = parser.parse_args()
        
        # Adjust log level for verbose mode
        if args.verbose:
            args.log_level = "DEBUG"
        
        # Initialize logging
        self.setup_logging(args.log_level, args.log_file)
        self.start_time = time.time()
        
        # Load configuration
        self.config = self.load_config(args.config)
        
        # Check dependencies and environment
        check_dependencies(self.logger)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        self.config['output_dir'] = str(output_dir)
        
        # Validate mode-specific requirements
        if args.mode == "attack" and not args.target:
            self.logger.error("❌ Attack mode requires --target parameter")
            sys.exit(1)
        
        if args.mode == "process_data" and not args.data:
            self.logger.error("❌ Process_data mode requires --data parameter")
            sys.exit(1)
        
        if args.target:
            target_path = Path(args.target)
            if not target_path.exists():
                self.logger.error(f"❌ Target model file not found: {args.target}")
                sys.exit(1)
            self.config['target_model'] = str(target_path)
        
        # Execute based on mode
        success = False
        
        # MANDATORY FIRST STAGE: Universal Data Processing
        try:
            self.logger.info("🚀 EXECUTING UNIVERSAL DATA PROCESSING (MANDATORY FIRST STAGE)")
            self.logger.info("=" * 80)
            
            # Merge universal processing config with user config
            if 'universal_processing' not in self.config:
                self.config.update(UNIVERSAL_PROCESSING_CONFIG)
            
            # Execute universal data processing
            processed_data = integrate_with_existing_pipeline(
                mode=args.mode,
                config=self.config,
                logger=self.logger
            )
            
            if processed_data:
                self.logger.info("✅ UNIVERSAL DATA PROCESSING COMPLETED - Proceeding with main pipeline")
                self.logger.info("=" * 80)
            else:
                self.logger.info("ℹ️  Universal processing not required for this mode - Proceeding directly")
            
        except Exception as e:
            self.logger.error(f"❌ UNIVERSAL DATA PROCESSING FAILED: {str(e)}")
            self.logger.error("🛑 Cannot proceed without properly processed data")
            sys.exit(1)
        
        # Main pipeline execution (now with processed data)
        if args.mode == "process_data":
            # Standalone data processing mode
            from src.cli.universal_data_cli import UniversalDataCLI
            
            self.logger.info("📊 STANDALONE UNIVERSAL DATA PROCESSING MODE")
            universal_cli = UniversalDataCLI(self.config, self.logger)
            result = universal_cli.execute(args.data)
            success = result['success']
            
        elif args.mode == "baseline":
            success = self.run_baseline(self.config)
        elif args.mode == "attack":
            success = self.run_attack(self.config)
        elif args.mode == "defense":
            success = self.run_defense(self.config)
        elif args.mode == "evaluate":
            success = self.run_evaluation(self.config)
        elif args.mode == "pipeline":
            success = self.run_pipeline(self.config)
        
        # Print summary and exit
        self.print_summary()
        
        if success:
            self.logger.info("🎉 SUCCESS: All operations completed successfully")
            sys.exit(0)
        else:
            self.logger.error("💥 FAILURE: One or more operations failed")
            sys.exit(1)

if __name__ == "__main__":
    cli = Adv5GCLI()
    cli.main()
