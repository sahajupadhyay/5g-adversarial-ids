"""
Pipeline CLI Module

Handles end-to-end pipeline execution orchestrating baseline training,
attack generation, defense hardening, and comprehensive evaluation.

Author: Capstone Team
Date: September 3, 2025
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from src.cli.utils import (
    create_output_directory, print_experiment_header, 
    print_experiment_footer, save_results
)
from src.cli.baseline_cli import BaselineCLI
from src.cli.attack_cli import AttackCLI
from src.cli.defense_cli import DefenseCLI
from src.cli.evaluation_cli import EvaluationCLI

class PipelineCLI:
    """CLI interface for end-to-end pipeline execution."""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.pipeline_config = config.get('pipeline', {})
        self.pipeline_results = {}
        
    def validate_pipeline_config(self) -> bool:
        """Validate pipeline-specific configuration."""
        # Check which stages are enabled
        stages = self.pipeline_config.get('stages', ['baseline', 'attack', 'defense', 'evaluation'])
        
        valid_stages = ['baseline', 'attack', 'defense', 'evaluation']
        for stage in stages:
            if stage not in valid_stages:
                self.logger.error(f"❌ Invalid pipeline stage: {stage}")
                return False
        
        self.pipeline_config['stages'] = stages
        self.logger.info(f"🔄 Pipeline stages: {' -> '.join(stages)}")
        
        return True
    
    def execute_baseline_stage(self) -> bool:
        """Execute baseline training stage."""
        if 'baseline' not in self.pipeline_config.get('stages', []):
            self.logger.info("⏭️  Skipping baseline stage")
            return True
            
        self.logger.info("🏗️  STAGE 1: Baseline Training")
        
        try:
            baseline_cli = BaselineCLI(self.config, self.logger)
            success = baseline_cli.execute()
            
            if success:
                self.pipeline_results['baseline'] = {
                    'status': 'completed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.info("✅ Baseline stage completed successfully")
            else:
                self.pipeline_results['baseline'] = {
                    'status': 'failed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.error("❌ Baseline stage failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Baseline stage error: {e}")
            self.pipeline_results['baseline'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            return False
    
    def execute_attack_stage(self) -> bool:
        """Execute adversarial attack stage."""
        if 'attack' not in self.pipeline_config.get('stages', []):
            self.logger.info("⏭️  Skipping attack stage")
            return True
            
        self.logger.info("⚔️  STAGE 2: Adversarial Attacks")
        
        try:
            # Ensure target model is available
            target_model = self.config.get('target_model')
            if not target_model:
                # Use newly trained baseline model if available
                baseline_model_path = "models/rf_baseline_tuned.joblib"
                if Path(baseline_model_path).exists():
                    self.config['target_model'] = baseline_model_path
                    self.logger.info(f"🎯 Using baseline model as target: {baseline_model_path}")
                else:
                    self.logger.error("❌ No target model available for attacks")
                    return False
            
            attack_cli = AttackCLI(self.config, self.logger)
            success = attack_cli.execute()
            
            if success:
                self.pipeline_results['attack'] = {
                    'status': 'completed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.info("✅ Attack stage completed successfully")
            else:
                self.pipeline_results['attack'] = {
                    'status': 'failed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.error("❌ Attack stage failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Attack stage error: {e}")
            self.pipeline_results['attack'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            return False
    
    def execute_defense_stage(self) -> bool:
        """Execute adversarial defense stage."""
        if 'defense' not in self.pipeline_config.get('stages', []):
            self.logger.info("⏭️  Skipping defense stage")
            return True
            
        self.logger.info("🛡️  STAGE 3: Adversarial Defense")
        
        try:
            defense_cli = DefenseCLI(self.config, self.logger)
            success = defense_cli.execute()
            
            if success:
                self.pipeline_results['defense'] = {
                    'status': 'completed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.info("✅ Defense stage completed successfully")
            else:
                self.pipeline_results['defense'] = {
                    'status': 'failed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.error("❌ Defense stage failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Defense stage error: {e}")
            self.pipeline_results['defense'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            return False
    
    def execute_evaluation_stage(self) -> bool:
        """Execute comprehensive evaluation stage."""
        if 'evaluation' not in self.pipeline_config.get('stages', []):
            self.logger.info("⏭️  Skipping evaluation stage")
            return True
            
        self.logger.info("📊 STAGE 4: Comprehensive Evaluation")
        
        try:
            eval_cli = EvaluationCLI(self.config, self.logger)
            success = eval_cli.execute()
            
            if success:
                self.pipeline_results['evaluation'] = {
                    'status': 'completed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.info("✅ Evaluation stage completed successfully")
            else:
                self.pipeline_results['evaluation'] = {
                    'status': 'failed',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                self.logger.error("❌ Evaluation stage failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"❌ Evaluation stage error: {e}")
            self.pipeline_results['evaluation'] = {
                'status': 'error',
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            return False
    
    def check_stage_dependencies(self) -> bool:
        """Check if all stage dependencies are satisfied."""
        stages = self.pipeline_config.get('stages', [])
        
        # Check dependencies
        dependencies = {
            'attack': ['baseline'],  # Attack needs baseline model
            'defense': ['baseline'],  # Defense can use baseline as starting point
            'evaluation': ['baseline']  # Evaluation needs at least baseline
        }
        
        for stage in stages:
            if stage in dependencies:
                required_stages = dependencies[stage]
                for required in required_stages:
                    if required not in stages:
                        # Check if required artifacts exist
                        if stage == 'attack' and required == 'baseline':
                            if not Path("models/rf_baseline_tuned.joblib").exists():
                                self.logger.error(f"❌ Attack stage requires baseline model")
                                return False
                        # Add more dependency checks as needed
        
        return True
    
    def generate_pipeline_report(self, output_dir: Path, total_time: float) -> bool:
        """Generate comprehensive pipeline execution report."""
        report_path = output_dir / 'pipeline_report.md'
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 5G Adversarial IDS - Pipeline Execution Report\n\n")
                f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Total Execution Time**: {total_time:.2f} seconds\n\n")
                
                # Pipeline overview
                stages = self.pipeline_config.get('stages', [])
                f.write("## Pipeline Overview\n\n")
                f.write(f"**Stages Executed**: {' -> '.join(stages)}\n\n")
                
                # Stage results
                f.write("## Stage Results\n\n")
                f.write("| Stage | Status | Timestamp | Notes |\n")
                f.write("|-------|--------|-----------|-------|\n")
                
                for stage in stages:
                    if stage in self.pipeline_results:
                        result = self.pipeline_results[stage]
                        status = result['status']
                        timestamp = result['timestamp']
                        error = result.get('error', '')
                        
                        # Status emoji
                        status_emoji = {
                            'completed': '[OK]',
                            'failed': '[FAIL]',
                            'error': '[ERR]'
                        }.get(status, '[?]')
                        
                        f.write(f"| {stage} | {status_emoji} {status} | {timestamp} | {error} |\n")
                    else:
                        f.write(f"| {stage} | [SKIP] skipped | - | - |\n")
                
                f.write("\n")
                
                # Success summary
                completed_stages = [stage for stage, result in self.pipeline_results.items() 
                                  if result['status'] == 'completed']
                failed_stages = [stage for stage, result in self.pipeline_results.items() 
                               if result['status'] in ['failed', 'error']]
                
                f.write("## Summary\n\n")
                f.write(f"- **Completed Stages**: {len(completed_stages)}/{len(stages)}\n")
                f.write(f"- **Failed Stages**: {len(failed_stages)}\n")
                f.write(f"- **Success Rate**: {len(completed_stages)/len(stages)*100:.1f}%\n\n")
                
                if completed_stages:
                    f.write(f"**Successful Stages**: {', '.join(completed_stages)}\n")
                
                if failed_stages:
                    f.write(f"**Failed Stages**: {', '.join(failed_stages)}\n")
                
                # Next steps
                f.write("\n## Next Steps\n\n")
                if failed_stages:
                    f.write("1. Review error logs for failed stages\n")
                    f.write("2. Fix configuration or dependencies\n")
                    f.write("3. Re-run failed stages individually\n")
                else:
                    f.write("All stages completed successfully.\n")
                    f.write("1. Review generated reports and results\n")
                    f.write("2. Analyze model performance\n")
                    f.write("3. Prepare final documentation\n")
            
            self.logger.info(f"✅ Pipeline report generated: {report_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to generate pipeline report: {e}")
            return False
    
    def save_pipeline_artifacts(self, total_time: float) -> bool:
        """Save pipeline execution results and artifacts."""
        try:
            # Create output directory
            output_dir = create_output_directory(
                self.config.get('output_dir', 'results'),
                'full_pipeline',
                self.logger
            )
            
            # Save pipeline results
            pipeline_summary = {
                'execution_time': total_time,
                'configuration': self.pipeline_config,
                'stage_results': self.pipeline_results,
                'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            results_path = output_dir / 'pipeline_results.json'
            save_results(pipeline_summary, str(results_path), self.logger)
            
            # Generate comprehensive report
            self.generate_pipeline_report(output_dir, total_time)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save pipeline artifacts: {e}")
            return False
    
    def execute(self) -> bool:
        """Execute complete end-to-end pipeline."""
        print_experiment_header("FULL PIPELINE EXECUTION", self.config, self.logger)
        
        start_time = time.time()
        overall_success = True
        
        try:
            # Validate configuration
            if not self.validate_pipeline_config():
                return False
            
            # Check stage dependencies
            if not self.check_stage_dependencies():
                return False
            
            stages = self.pipeline_config.get('stages', [])
            self.logger.info(f"🚀 Starting pipeline with {len(stages)} stages")
            
            # Execute each stage
            for i, stage in enumerate(stages, 1):
                self.logger.info(f"📍 STAGE {i}/{len(stages)}: {stage.upper()}")
                
                stage_start_time = time.time()
                
                if stage == 'baseline':
                    success = self.execute_baseline_stage()
                elif stage == 'attack':
                    success = self.execute_attack_stage()
                elif stage == 'defense':
                    success = self.execute_defense_stage()
                elif stage == 'evaluation':
                    success = self.execute_evaluation_stage()
                else:
                    self.logger.error(f"❌ Unknown stage: {stage}")
                    success = False
                
                stage_time = time.time() - stage_start_time
                self.logger.info(f"⏱️  Stage {stage} completed in {stage_time:.2f}s")
                
                if not success:
                    overall_success = False
                    # Check if we should continue on failure
                    continue_on_failure = self.pipeline_config.get('continue_on_failure', False)
                    if not continue_on_failure:
                        self.logger.error(f"❌ Pipeline stopped due to {stage} stage failure")
                        break
                    else:
                        self.logger.warning(f"⚠️  Continuing pipeline despite {stage} stage failure")
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Save pipeline artifacts
            self.save_pipeline_artifacts(total_time)
            
            # Log final summary
            completed_stages = len([r for r in self.pipeline_results.values() if r['status'] == 'completed'])
            total_stages = len(stages)
            
            self.logger.info("🔄 PIPELINE EXECUTION SUMMARY:")
            self.logger.info(f"   Total time: {total_time:.2f}s")
            self.logger.info(f"   Stages completed: {completed_stages}/{total_stages}")
            self.logger.info(f"   Success rate: {completed_stages/total_stages*100:.1f}%")
            
            if overall_success:
                self.logger.info("🎉 All pipeline stages completed successfully!")
            else:
                self.logger.warning("⚠️  Some pipeline stages failed or had errors")
            
            print_experiment_footer("FULL PIPELINE EXECUTION", overall_success, self.logger)
            return overall_success
            
        except Exception as e:
            total_time = time.time() - start_time
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            self.save_pipeline_artifacts(total_time)
            print_experiment_footer("FULL PIPELINE EXECUTION", False, self.logger)
            return False
