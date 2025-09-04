"""
Universal Data Processing CLI

Integrates the universal data processor into the main 5G IDS pipeline
as the mandatory first stage for all operations.

Author: Capstone Team
Date: September 4, 2025
"""

import logging
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd

from src.data.universal_processor import UniversalDataProcessor
from src.cli.utils import (
    create_output_directory, print_experiment_header, save_results,
    setup_logging
)

class UniversalDataCLI:
    """CLI for universal data processing"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.processor = None
        
    def execute(self, data_source: Union[str, Path, pd.DataFrame, dict]) -> Dict[str, Any]:
        """Execute universal data processing"""
        print_experiment_header("UNIVERSAL DATA PROCESSING", self.config, self.logger)
        
        start_time = time.time()
        
        try:
            # Initialize processor
            self.logger.info("🔧 Initializing Universal Data Processor...")
            
            processing_config = self.config.get('processing', {})
            self.processor = UniversalDataProcessor(
                target_features=processing_config.get('target_features', 43),
                target_samples_min=processing_config.get('target_samples_min', 100),
                logger=self.logger
            )
            
            # Process the data
            self.logger.info(f"📥 Processing data source: {data_source}")
            processed_data = self.processor.process_data(data_source, processing_config)
            
            # Save processed data
            output_dir = create_output_directory(
                self.config.get('output_dir', 'results'),
                'universal_processing',
                self.logger
            )
            
            # Save the processed datasets
            self._save_processed_data(processed_data, output_dir)
            
            # Generate processing report
            self._generate_processing_report(processed_data, output_dir, time.time() - start_time)
            
            # Save processing pipeline for reproducibility
            pipeline_path = output_dir / 'processing_pipeline.json'
            self.processor.save_processing_pipeline(str(pipeline_path))
            
            self.logger.info("✅ Universal data processing completed successfully")
            
            return {
                'success': True,
                'processed_data': processed_data,
                'output_directory': str(output_dir),
                'processing_time': time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Universal data processing failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def _save_processed_data(self, processed_data: Dict[str, Any], output_dir: Path):
        """Save processed data in multiple formats"""
        
        # Save as NumPy arrays (for ML models)
        np.save(output_dir / 'X_processed.npy', processed_data['X_train'])
        if processed_data['y_train'] is not None:
            np.save(output_dir / 'y_processed.npy', processed_data['y_train'])
        
        # Save as pandas-compatible format
        X_df = pd.DataFrame(
            processed_data['X_train'], 
            columns=[f'feature_{i}' for i in range(processed_data['X_train'].shape[1])]
        )
        X_df.to_csv(output_dir / 'X_processed.csv', index=False)
        
        if processed_data['y_train'] is not None:
            y_df = pd.DataFrame({'target': processed_data['y_train']})
            y_df.to_csv(output_dir / 'y_processed.csv', index=False)
        
        # Save metadata
        save_results(processed_data['metadata'], str(output_dir / 'processing_metadata.json'), self.logger)
        save_results(processed_data['data_info'], str(output_dir / 'data_info.json'), self.logger)
        
        self.logger.info(f"💾 Processed data saved to: {output_dir}")
    
    def _generate_processing_report(self, processed_data: Dict[str, Any], output_dir: Path, processing_time: float):
        """Generate comprehensive processing report"""
        
        data_info = processed_data['data_info']
        metadata = processed_data['metadata']
        
        report = f"""# Universal Data Processing Report

## Processing Summary
- **Processing Time**: {processing_time:.2f} seconds
- **Original Data Shape**: {data_info['original_shape']}
- **Processed Data Shape**: {data_info['processed_shape']}
- **Feature Count**: {data_info['feature_count']}
- **Sample Count**: {data_info['sample_count']}
- **Has Labels**: {data_info['has_labels']}

## Data Source Information
- **Source Type**: {metadata.get('source_type', 'Unknown')}
- **Format**: {metadata.get('format', 'Unknown')}
"""
        
        if 'file_path' in metadata:
            report += f"- **File Path**: {metadata['file_path']}\n"
            report += f"- **File Size**: {metadata.get('file_size', 'Unknown')} bytes\n"
        
        # Data Cleaning Report
        if 'cleaning_report' in metadata:
            cleaning = metadata['cleaning_report']
            report += f"""
## Data Cleaning Results
- **Quality Score**: {cleaning.get('quality_score', 'N/A'):.3f}/1.0
- **Operations Performed**: {', '.join(cleaning.get('operations_performed', []))}
- **Duplicates Removed**: {cleaning.get('duplicates_removed', 0)}
"""
            
            if 'missing_values' in cleaning:
                missing = cleaning['missing_values']
                if missing['columns_with_missing']:
                    report += f"""
### Missing Value Handling
"""
                    for col, info in missing['columns_with_missing'].items():
                        report += f"- **{col}**: {info['count']} missing ({info['percentage']:.1f}%)\n"
        
        # Feature Engineering Report
        if 'feature_report' in metadata:
            features = metadata['feature_report']
            report += f"""
## Feature Engineering Results
- **Original Features**: {features.get('original_features', 'Unknown')}
- **Final Features**: {features.get('final_features', 'Unknown')}
- **Features Added**: {features.get('features_added', 0)}
- **Operations**: {', '.join(features.get('operations_performed', []))}
"""
        
        # Standardization Report
        if 'standardization_report' in metadata:
            std = metadata['standardization_report']
            report += f"""
## Format Standardization
- **Target Features**: {std['target_features']}
- **Final Shape**: {std['final_shape']}
- **Operations**: {', '.join(std.get('operations_performed', []))}
"""
        
        # Validation Report
        if 'validation_report' in metadata:
            validation = metadata['validation_report']
            report += f"""
## Data Validation
- **Overall Valid**: {'✅ PASS' if validation.get('overall_valid', False) else '❌ FAIL'}
- **Feature Count Valid**: {'✅' if validation.get('feature_count_valid', False) else '❌'}
- **Sample Count Adequate**: {'✅' if validation.get('sample_count_adequate', False) else '❌'}
- **No Missing Values**: {'✅' if validation.get('no_missing_values', False) else '❌'}
- **No Infinite Values**: {'✅' if validation.get('no_infinite_values', False) else '❌'}
- **Feature Variance Adequate**: {'✅' if validation.get('feature_variance_adequate', False) else '❌'}
"""
        
        # Recommendations
        report += f"""
## Recommendations

### Data Quality
"""
        if metadata.get('validation_report', {}).get('overall_valid', False):
            report += "✅ **Data is ready for 5G IDS model training**\n"
        else:
            report += "⚠️ **Data quality issues detected - review validation results**\n"
        
        report += f"""
### Next Steps
1. **Model Training**: Use the processed data with your 5G IDS models
2. **Performance Monitoring**: Track model performance on this processed data
3. **Pipeline Reproducibility**: Use the saved processing pipeline for new data

### Files Generated
- `X_processed.npy` - Features in NumPy format
- `y_processed.npy` - Labels in NumPy format (if available)
- `X_processed.csv` - Features in CSV format
- `y_processed.csv` - Labels in CSV format (if available)
- `processing_metadata.json` - Complete processing metadata
- `processing_pipeline.json` - Reproducible processing pipeline
"""
        
        # Save report
        with open(output_dir / 'processing_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"📋 Processing report generated: {output_dir / 'processing_report.md'}")


def integrate_with_existing_pipeline(mode: str, config: Dict[str, Any], logger: logging.Logger, **kwargs):
    """
    Integrate universal data processing with existing pipeline modes
    
    This function acts as a preprocessor for all existing modes
    """
    
    # Check if universal processing is enabled
    if not config.get('universal_processing', {}).get('enabled', True):
        logger.info("ℹ️ Universal processing disabled - using original data pipeline")
        return None
    
    logger.info("🚀 STAGE 0: Universal Data Processing (Mandatory)")
    logger.info("=" * 60)
    
    # Determine data source based on mode
    data_source = None
    
    if mode in ['baseline', 'defense']:
        # For training modes, use configured data directory
        data_source = config.get('data', {}).get('processed_dir', 'data/processed')
        
    elif mode == 'attack':
        # For attack mode, might need to process attack data
        data_source = kwargs.get('attack_data') or config.get('data', {}).get('processed_dir', 'data/processed')
        
    elif mode == 'evaluate':
        # For evaluation, use test data
        data_source = config.get('data', {}).get('test_dir') or config.get('data', {}).get('processed_dir', 'data/processed')
    
    elif mode == 'pipeline':
        # For pipeline mode, use main data source
        data_source = config.get('data', {}).get('processed_dir', 'data/processed')
    
    else:
        logger.info(f"ℹ️ Mode '{mode}' doesn't require universal processing")
        return None
    
    if not data_source:
        logger.warning("⚠️ No data source specified for universal processing")
        return None
    
    # Initialize and execute universal processing
    universal_cli = UniversalDataCLI(config, logger)
    result = universal_cli.execute(data_source)
    
    if result['success']:
        logger.info("✅ Universal data processing completed successfully")
        logger.info(f"📊 Processed data: {result['processed_data']['data_info']['processed_shape']}")
        
        # Update config with processed data paths
        output_dir = Path(result['output_directory'])
        config['processed_data'] = {
            'X_path': str(output_dir / 'X_processed.npy'),
            'y_path': str(output_dir / 'y_processed.npy'),
            'metadata_path': str(output_dir / 'processing_metadata.json'),
            'pipeline_path': str(output_dir / 'processing_pipeline.json')
        }
        
        return result['processed_data']
    else:
        logger.error(f"❌ Universal data processing failed: {result['error']}")
        raise Exception(f"Universal data processing failed: {result['error']}")


# Configuration template for universal processing
UNIVERSAL_PROCESSING_CONFIG = {
    'universal_processing': {
        'enabled': True,
        'target_features': 43,
        'target_samples_min': 100,
        'processing': {
            'cleaning': {
                'missing_strategy': 'auto',  # auto, mean, median, knn
                'outlier_strategy': 'iqr',   # iqr, zscore, none
                'remove_duplicates': True
            },
            'feature_engineering': {
                'encoding_strategy': 'auto',  # auto, label, onehot
                'create_interactions': True,
                'max_interactions': 5,
                'create_polynomial': False,
                'poly_degree': 2,
                'create_time_features': True,
                'create_statistical_features': True
            },
            'standardization': {
                'feature_selection_method': 'auto',  # auto, variance, statistical, pca
                'scaling_method': 'standard'  # standard, minmax, robust
            }
        }
    }
}
