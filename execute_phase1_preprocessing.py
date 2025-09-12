#!/usr/bin/env python3
"""
Simplified Industrial-Grade Data Preprocessing Pipeline
=====================================================

Simplified version of the preprocessing pipeline that works with basic configuration
while maintaining industrial-grade standards.

Author: AI Assistant  
Date: September 12, 2025
Version: 1.0.1 (Simplified Industrial Grade)
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataPreprocessor")


class IndustrialDataProcessor:
    """
    Simplified but industrial-grade data processor.
    
    Implements your friend's proven methodology with robust error handling.
    """
    
    def __init__(self, project_root: Path):
        """Initialize processor with project paths."""
        self.project_root = Path(project_root)
        self.dataset_path = self.project_root / "Combined_DS" / "dataset.csv"
        self.processed_dir = self.project_root / "data" / "processed"
        self.target_column = "Label"
        
        # Processing state
        self.scaler = None
        self.feature_columns = None
        
        logger.info(f"Processor initialized: {self.dataset_path}")
    
    def load_and_validate_dataset(self) -> pd.DataFrame:
        """Load dataset with comprehensive validation."""
        try:
            logger.info(f"Loading dataset from: {self.dataset_path}")
            
            # Load dataset
            df = pd.read_csv(self.dataset_path, low_memory=False)
            
            # Basic validation
            if df.empty:
                raise ValueError("Dataset is empty")
            
            if len(df) < 1000:
                raise ValueError(f"Dataset too small: {len(df)} samples")
            
            # Check target column
            if self.target_column not in df.columns:
                available = list(df.columns)[:10]
                raise ValueError(f"Target column '{self.target_column}' not found. Available: {available}")
            
            # Log statistics
            logger.info(f"Dataset loaded successfully:")
            logger.info(f"  Shape: {df.shape}")
            logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Target distribution
            target_dist = df[self.target_column].value_counts()
            logger.info(f"  Target distribution:")
            for value, count in target_dist.items():
                pct = (count / len(df)) * 100
                logger.info(f"    {value}: {count:,} ({pct:.2f}%)")
            
            return df
            
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            raise
    
    def clean_and_sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data using friend's proven methodology."""
        try:
            logger.info("Starting data cleaning and sanitization")
            
            df_clean = df.copy()
            
            # 1. Handle infinity values (friend's approach)
            numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
            
            # Count infinity values
            inf_counts = {}
            for col in numeric_cols:
                inf_count = np.isinf(df_clean[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
            
            if inf_counts:
                total_inf = sum(inf_counts.values())
                logger.info(f"Replacing {total_inf:,} infinity values in {len(inf_counts)} columns")
                df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 2. Handle missing values with zero-fill (friend's strategy)
            missing_before = df_clean.isnull().sum().sum()
            if missing_before > 0:
                logger.info(f"Filling {missing_before:,} missing values with zeros")
                df_clean.fillna(0, inplace=True)
            
            # 3. Validate cleaning
            missing_after = df_clean.isnull().sum().sum()
            inf_after = np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()
            
            if missing_after > 0:
                raise ValueError(f"Missing values remain after cleaning: {missing_after}")
            if inf_after > 0:
                raise ValueError(f"Infinity values remain after cleaning: {inf_after}")
            
            logger.info("Data cleaning completed successfully")
            return df_clean
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {e}")
            raise
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Select features using friend's programmatic approach."""
        try:
            logger.info("Starting feature selection")
            
            # Programmatic feature selection (friend's approach)
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column
            if self.target_column in numeric_features:
                numeric_features.remove(self.target_column)
            
            if not numeric_features:
                raise ValueError("No numeric features found")
            
            logger.info(f"Selected {len(numeric_features)} numeric features")
            
            # Store feature columns
            self.feature_columns = numeric_features
            
            # Create features DataFrame
            df_features = df[numeric_features].copy()
            
            # Basic feature validation
            constant_features = []
            for col in numeric_features:
                if df_features[col].nunique() <= 1:
                    constant_features.append(col)
            
            if constant_features:
                logger.warning(f"Found {len(constant_features)} constant features: {constant_features[:5]}")
            
            return df_features, numeric_features
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            raise
    
    def create_stratified_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create stratified splits preserving class balance."""
        try:
            logger.info("Creating stratified data splits")
            
            # Prepare features and target
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            
            # Log class distribution
            class_counts = y.value_counts()
            logger.info("Class distribution before splitting:")
            for value, count in class_counts.items():
                pct = (count / len(y)) * 100
                logger.info(f"  {value}: {count:,} ({pct:.2f}%)")
            
            # First split: train vs (val + test) [70% vs 30%]
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=0.30,  # 30% for val + test
                stratify=y,
                random_state=42,
                shuffle=True
            )
            
            # Second split: val vs test [15% vs 15%]
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=0.50,  # 50% of remaining 30% = 15% total
                stratify=y_temp,
                random_state=42,
                shuffle=True
            )
            
            # Create datasets
            splits = {
                'train': pd.concat([X_train, y_train], axis=1),
                'val': pd.concat([X_val, y_val], axis=1), 
                'test': pd.concat([X_test, y_test], axis=1)
            }
            
            # Validate and log splits
            total_samples = sum(len(split_df) for split_df in splits.values())
            logger.info("Data splitting completed:")
            
            for split_name, split_df in splits.items():
                split_dist = split_df[self.target_column].value_counts()
                pct = (len(split_df) / total_samples) * 100
                logger.info(f"  {split_name}: {len(split_df):,} samples ({pct:.1f}%)")
                
                # Log class distribution per split
                for value, count in split_dist.items():
                    split_pct = (count / len(split_df)) * 100
                    logger.info(f"    {value}: {count:,} ({split_pct:.2f}%)")
            
            return splits
            
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            raise
    
    def fit_and_apply_scaling(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Fit scaler on training data and apply to all splits."""
        try:
            logger.info("Fitting and applying feature scaling")
            
            # Initialize StandardScaler
            self.scaler = StandardScaler()
            
            # Fit scaler on training data only (prevent data leakage)
            X_train = splits['train'][self.feature_columns]
            self.scaler.fit(X_train)
            
            logger.info(f"Scaler fitted on training data: {len(X_train):,} samples, {len(self.feature_columns)} features")
            
            # Apply scaling to all splits
            scaled_splits = {}
            for split_name, split_df in splits.items():
                # Separate features and target
                X_split = split_df[self.feature_columns]
                y_split = split_df[self.target_column]
                
                # Scale features
                X_scaled = self.scaler.transform(X_split)
                
                # Create scaled DataFrame
                X_scaled_df = pd.DataFrame(
                    X_scaled,
                    columns=self.feature_columns,
                    index=split_df.index
                )
                
                # Combine with target
                scaled_splits[split_name] = pd.concat([X_scaled_df, y_split], axis=1)
                
                logger.info(f"Applied scaling to {split_name}: {len(scaled_splits[split_name]):,} samples")
            
            # Validate scaling on training data
            train_features = scaled_splits['train'][self.feature_columns]
            mean_vals = train_features.mean()
            std_vals = train_features.std()
            
            logger.info(f"Scaling validation (training data):")
            logger.info(f"  Mean range: [{mean_vals.min():.3f}, {mean_vals.max():.3f}] (should be ~0)")
            logger.info(f"  Std range: [{std_vals.min():.3f}, {std_vals.max():.3f}] (should be ~1)")
            
            return scaled_splits
            
        except Exception as e:
            logger.error(f"Feature scaling failed: {e}")
            raise
    
    def save_artifacts(self, scaled_splits: Dict[str, pd.DataFrame]) -> None:
        """Save processed datasets and artifacts."""
        try:
            logger.info("Saving processed datasets and artifacts")
            
            # Ensure output directory exists
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save datasets
            dataset_paths = {
                'train': self.processed_dir / 'train.csv',
                'val': self.processed_dir / 'val.csv', 
                'test': self.processed_dir / 'test.csv'
            }
            
            for split_name, split_df in scaled_splits.items():
                save_path = dataset_paths[split_name]
                split_df.to_csv(save_path, index=False)
                logger.info(f"Saved {split_name} dataset: {save_path} ({len(split_df):,} samples)")
            
            # Save scaler
            scaler_path = self.processed_dir / 'scaler.joblib'
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved fitted scaler: {scaler_path}")
            
            # Save feature schema
            feature_path = self.processed_dir / 'feature_columns.json'
            with open(feature_path, 'w') as f:
                json.dump(self.feature_columns, f, indent=2)
            logger.info(f"Saved feature schema: {feature_path} ({len(self.feature_columns)} features)")
            
            # Save metadata
            metadata = {
                'total_samples': sum(len(df) for df in scaled_splits.values()),
                'feature_count': len(self.feature_columns),
                'train_samples': len(scaled_splits['train']),
                'val_samples': len(scaled_splits['val']),
                'test_samples': len(scaled_splits['test']),
                'target_column': self.target_column,
                'scaling_method': 'StandardScaler',
                'processing_date': str(pd.Timestamp.now())
            }
            
            metadata_path = self.processed_dir / 'processing_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved processing metadata: {metadata_path}")
            
        except Exception as e:
            logger.error(f"Artifact saving failed: {e}")
            raise
    
    def process_complete_pipeline(self) -> Dict[str, pd.DataFrame]:
        """Execute complete preprocessing pipeline."""
        try:
            logger.info("=" * 80)
            logger.info("🚀 STARTING INDUSTRIAL-GRADE DATA PREPROCESSING PIPELINE")
            logger.info("=" * 80)
            
            # Stage 1: Load and validate
            df_raw = self.load_and_validate_dataset()
            
            # Stage 2: Clean and sanitize
            df_clean = self.clean_and_sanitize_data(df_raw)
            
            # Stage 3: Feature selection
            df_features, feature_names = self.select_features(df_clean)
            
            # Combine features with target
            df_complete = pd.concat([df_features, df_clean[self.target_column]], axis=1)
            
            # Stage 4: Create splits
            splits = self.create_stratified_splits(df_complete)
            
            # Stage 5: Apply scaling
            scaled_splits = self.fit_and_apply_scaling(splits)
            
            # Stage 6: Save artifacts
            self.save_artifacts(scaled_splits)
            
            logger.info("=" * 80)
            logger.info("✅ DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            # Summary
            total_samples = sum(len(df) for df in scaled_splits.values())
            logger.info("PIPELINE SUMMARY:")
            logger.info(f"  📊 Total samples: {total_samples:,}")
            logger.info(f"  🎯 Features: {len(self.feature_columns)}")
            logger.info(f"  🚂 Train: {len(scaled_splits['train']):,}")
            logger.info(f"  🔍 Val: {len(scaled_splits['val']):,}")
            logger.info(f"  🧪 Test: {len(scaled_splits['test']):,}")
            
            return scaled_splits
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ DATA PROCESSING PIPELINE FAILED")
            logger.error(f"Error: {e}")
            logger.error("=" * 80)
            raise


def main():
    """Main execution function."""
    try:
        # Initialize processor
        project_root = Path(__file__).parent
        processor = IndustrialDataProcessor(project_root)
        
        # Run pipeline
        datasets = processor.process_complete_pipeline()
        
        # Validation
        logger.info("🔍 Validating outputs...")
        
        # Check files exist
        processed_dir = project_root / "data" / "processed"
        required_files = ['train.csv', 'val.csv', 'test.csv', 'scaler.joblib', 'feature_columns.json']
        
        for filename in required_files:
            filepath = processed_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Missing output file: {filepath}")
            logger.info(f"✓ {filename} created successfully")
        
        # Success
        logger.info("🎉 " + "=" * 76)
        logger.info("🎉 PHASE 1 COMPLETE: INDUSTRIAL-GRADE DATA PREPROCESSING SUCCESS!")
        logger.info("🎉 " + "=" * 76) 
        logger.info("📊 Ready for Phase 2: Baseline Model Implementation")
        logger.info("📁 All artifacts saved in: data/processed/")
        
        return 0
        
    except Exception as e:
        logger.error("💥 " + "=" * 76)
        logger.error("💥 CRITICAL FAILURE IN PREPROCESSING PIPELINE")
        logger.error(f"💥 Error: {e}")
        logger.error("💥 " + "=" * 76)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)