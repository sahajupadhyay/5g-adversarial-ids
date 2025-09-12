"""
Industrial-Grade Data Processing Pipeline
=======================================

Production-ready data preprocessing pipeline with comprehensive error handling,
validation, security features, and reproducibility guarantees.

This module implements your friend's excellent preprocessing methodology enhanced
to meet industrial software engineering standards.

Author: AI Assistant  
Date: September 12, 2025
Version: 1.0.0 (Industrial Grade)
License: MIT
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import DataConversionWarning

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Local imports - will be properly configured when package structure is complete
try:
    from ...config.data_config import (
        DataConfig, 
        DataSplitStrategy, 
        ImputationStrategy, 
        ScalingStrategy,
        create_default_config
    )
except ImportError:
    # Fallback for development - will create simplified config classes
    from pathlib import Path
    from dataclasses import dataclass, field
    from enum import Enum
    
    class DataSplitStrategy(Enum):
        STRATIFIED = "stratified"
    
    class ImputationStrategy(Enum):
        ZERO_FILL = "zero"
        MEDIAN = "median"
    
    class ScalingStrategy(Enum):
        STANDARD = "standard"
        NONE = "none"
    
    @dataclass
    class DataConfig:
        """Simplified config for development"""
        dataset_path: Path
        target_column: str = "Label"
        train_ratio: float = 0.70
        validation_ratio: float = 0.15
        test_ratio: float = 0.15
        random_state: int = 42
        
        def get_summary(self):
            return {"dataset_path": str(self.dataset_path)}
        
        def validate_environment(self):
            return True
    
    def create_default_config(project_root):
        """Simple config creation for development"""
        project_root = Path(project_root)
        return DataConfig(
            dataset_path=project_root / "Combined_DS" / "dataset.csv"
        )


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


class DataValidationError(DataProcessingError):
    """Custom exception for data validation failures."""
    pass


class FeatureEngineeringError(DataProcessingError):
    """Custom exception for feature engineering failures."""
    pass


class DataProcessor:
    """
    Industrial-grade data preprocessing pipeline.
    
    This class implements a comprehensive data processing pipeline that handles:
    - Data loading with validation and error recovery
    - Missing value imputation with multiple strategies
    - Feature selection and engineering
    - Stratified data splitting preserving class balance
    - Feature standardization with proper train/validation isolation
    - Artifact persistence for reproducible deployment
    
    The implementation follows your friend's proven methodology enhanced with
    enterprise-grade error handling, logging, and validation.
    
    Attributes:
        config: Data processing configuration
        logger: Structured logger for operations tracking
        scaler: Fitted StandardScaler for feature normalization
        feature_columns: List of selected feature column names
        
    Example:
        >>> from pathlib import Path
        >>> config = create_default_config(Path("ADVERSARIAL_IDS_DEEP_LEARNING"))
        >>> processor = DataProcessor(config)
        >>> datasets = processor.process_complete_pipeline()
        >>> print(f"Training samples: {len(datasets['train'])}")
        
    Raises:
        DataProcessingError: For general processing failures
        DataValidationError: For data quality validation failures
        FeatureEngineeringError: For feature engineering issues
    """
    
    def __init__(self, config: DataConfig) -> None:
        """
        Initialize the data processor with configuration validation.
        
        Args:
            config: Comprehensive data processing configuration
            
        Raises:
            ValueError: If configuration is invalid
            EnvironmentError: If environment validation fails
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate environment before processing
        self.config.validate_environment()
        
        # Initialize processing state
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: Optional[List[str]] = None
        self._processing_metadata: Dict[str, Any] = {}
        
        self.logger.info("DataProcessor initialized with configuration: %s", 
                        config.get_summary())
    
    def _get_dataset_path(self) -> Path:
        """Get dataset path from config, handling both full and simplified configs."""
        if hasattr(self.config, 'paths'):
            return self.config.paths.dataset_path
        else:
            return self.config.dataset_path
    
    def _get_processed_dir(self) -> Path:
        """Get processed data directory path."""
        if hasattr(self.config, 'paths'):
            return self.config.paths.processed_data_dir
        else:
            return Path(self.config.dataset_path).parent.parent / "data" / "processed"
    
    def load_and_validate_dataset(self) -> pd.DataFrame:
        """
        Load dataset with comprehensive validation and error handling.
        
        Returns:
            pd.DataFrame: Validated dataset ready for processing
            
        Raises:
            DataValidationError: If dataset fails validation checks
            FileNotFoundError: If dataset file doesn't exist
            pd.errors.EmptyDataError: If dataset is empty
            pd.errors.ParserError: If dataset format is invalid
            
        Example:
            >>> processor = DataProcessor(config)
            >>> df = processor.load_and_validate_dataset()
            >>> print(f"Dataset shape: {df.shape}")
        """
        try:
            dataset_path = self._get_dataset_path()
            self.logger.info("Loading dataset from: %s", dataset_path)
            
            # Load dataset with error handling for various issues
            try:
                df = pd.read_csv(dataset_path, low_memory=False)
            except pd.errors.EmptyDataError as e:
                raise DataValidationError(f"Dataset file is empty: {dataset_path}") from e
            except pd.errors.ParserError as e:
                raise DataValidationError(f"Dataset parsing failed: {e}") from e
            except MemoryError as e:
                raise DataProcessingError(f"Insufficient memory to load dataset: {e}") from e
            
            # Basic validation
            if df.empty:
                raise DataValidationError("Dataset is empty after loading")
            
            if len(df) < self.config.quality_config.min_samples:
                raise DataValidationError(
                    f"Dataset has insufficient samples: {len(df)} < {self.config.quality_config.min_samples}"
                )
            
            # Validate target column exists  
            target_column = getattr(self.config, 'target_column', 'Label')
            if target_column not in df.columns:
                available_cols = list(df.columns)
                raise DataValidationError(
                    f"Target column '{target_column}' not found. "
                    f"Available columns: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}"
                )
            
            # Check for duplicate rows if enabled
            if self.config.quality_config.duplicate_detection_enabled:
                duplicate_count = df.duplicated().sum()
                if duplicate_count > 0:
                    self.logger.warning("Found %d duplicate rows (%.2f%%)", 
                                      duplicate_count, (duplicate_count / len(df)) * 100)
            
            # Log dataset statistics
            self._log_dataset_statistics(df)
            
            self.logger.info("Dataset loaded successfully: shape=%s", df.shape)
            return df
            
        except Exception as e:
            self.logger.error("Dataset loading failed: %s", str(e))
            raise
    
    def _log_dataset_statistics(self, df: pd.DataFrame) -> None:
        """
        Log comprehensive dataset statistics for monitoring.
        
        Args:
            df: Dataset to analyze
        """
        try:
            target_col = self.config.processing_config.target_column
            
            # Basic statistics
            self.logger.info("Dataset Statistics:")
            self.logger.info("  Shape: %s", df.shape)
            self.logger.info("  Memory usage: %.2f MB", df.memory_usage(deep=True).sum() / 1024**2)
            
            # Missing values analysis
            missing_counts = df.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            if len(missing_cols) > 0:
                self.logger.info("  Columns with missing values: %d", len(missing_cols))
                for col, count in missing_cols.head(5).items():
                    pct = (count / len(df)) * 100
                    self.logger.info("    %s: %d (%.2f%%)", col, count, pct)
            
            # Target distribution
            if target_col in df.columns:
                target_dist = df[target_col].value_counts()
                self.logger.info("  Target distribution:")
                for value, count in target_dist.items():
                    pct = (count / len(df)) * 100
                    self.logger.info("    %s: %d (%.2f%%)", value, count, pct)
            
            # Data types summary
            dtype_counts = df.dtypes.value_counts()
            self.logger.info("  Data types: %s", dtype_counts.to_dict())
            
        except Exception as e:
            self.logger.warning("Failed to log dataset statistics: %s", str(e))
    
    def clean_and_sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and sanitize dataset following your friend's proven methodology.
        
        This implements the robust data cleaning approach from your friend's
        preprocessing.py enhanced with comprehensive error handling.
        
        Args:
            df: Raw dataset to clean
            
        Returns:
            pd.DataFrame: Cleaned and sanitized dataset
            
        Raises:
            DataProcessingError: If cleaning operations fail
            
        Example:
            >>> df_raw = processor.load_and_validate_dataset()
            >>> df_clean = processor.clean_and_sanitize_data(df_raw)
            >>> print(f"Infinity values removed: {np.isinf(df_clean.select_dtypes(include=np.number)).sum().sum()}")
        """
        try:
            self.logger.info("Starting data cleaning and sanitization")
            df_clean = df.copy()
            
            # 1. Handle infinity values (following friend's approach)
            if self.config.processing_config.handle_infinity:
                numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                
                # Count infinity values before cleaning
                inf_counts = {}
                for col in numeric_cols:
                    inf_count = np.isinf(df_clean[col]).sum()
                    if inf_count > 0:
                        inf_counts[col] = inf_count
                
                if inf_counts:
                    self.logger.info("Replacing infinity values in %d columns", len(inf_counts))
                    for col, count in list(inf_counts.items())[:5]:  # Log first 5
                        self.logger.info("  %s: %d infinity values", col, count)
                    
                    # Replace infinity with NaN, then handle with imputation strategy
                    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # 2. Handle missing values based on strategy
            missing_before = df_clean.isnull().sum().sum()
            if missing_before > 0:
                self.logger.info("Handling %d missing values using strategy: %s", 
                               missing_before, self.config.processing_config.imputation_strategy.value)
                
                df_clean = self._apply_imputation_strategy(df_clean)
                
                missing_after = df_clean.isnull().sum().sum()
                self.logger.info("Missing values after imputation: %d", missing_after)
            
            # 3. Validate data quality after cleaning
            self._validate_cleaned_data(df_clean)
            
            # 4. Store cleaning metadata
            self._processing_metadata['cleaning'] = {
                'infinity_values_replaced': sum(inf_counts.values()) if 'inf_counts' in locals() else 0,
                'missing_values_imputed': missing_before,
                'imputation_strategy': self.config.processing_config.imputation_strategy.value,
                'final_missing_count': df_clean.isnull().sum().sum()
            }
            
            self.logger.info("Data cleaning completed successfully")
            return df_clean
            
        except Exception as e:
            self.logger.error("Data cleaning failed: %s", str(e))
            raise DataProcessingError(f"Data cleaning failed: {e}") from e
    
    def _apply_imputation_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the configured imputation strategy to handle missing values.
        
        Args:
            df: DataFrame with missing values to impute
            
        Returns:
            pd.DataFrame: DataFrame with missing values imputed
            
        Raises:
            FeatureEngineeringError: If imputation fails
        """
        try:
            strategy = self.config.processing_config.imputation_strategy
            
            if strategy == ImputationStrategy.ZERO_FILL:
                # Friend's proven approach: simple and effective
                return df.fillna(0)
                
            elif strategy == ImputationStrategy.MEDIAN:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_imputed = df.copy()
                for col in numeric_cols:
                    if df_imputed[col].isnull().any():
                        median_val = df_imputed[col].median()
                        df_imputed[col].fillna(median_val, inplace=True)
                return df_imputed
                
            elif strategy == ImputationStrategy.MEAN:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df_imputed = df.copy()
                for col in numeric_cols:
                    if df_imputed[col].isnull().any():
                        mean_val = df_imputed[col].mean()
                        df_imputed[col].fillna(mean_val, inplace=True)
                return df_imputed
                
            elif strategy == ImputationStrategy.DROP:
                return df.dropna()
                
            else:
                raise FeatureEngineeringError(f"Unsupported imputation strategy: {strategy}")
                
        except Exception as e:
            raise FeatureEngineeringError(f"Imputation failed: {e}") from e
    
    def _validate_cleaned_data(self, df: pd.DataFrame) -> None:
        """
        Validate data quality after cleaning operations.
        
        Args:
            df: Cleaned dataset to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        # Check for remaining infinity values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(df[numeric_cols]).sum().sum()
        if inf_count > 0:
            raise DataValidationError(f"Infinity values remain after cleaning: {inf_count}")
        
        # Check missing value ratio
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > self.config.quality_config.max_missing_ratio:
            raise DataValidationError(
                f"Missing value ratio too high: {missing_ratio:.3f} > {self.config.quality_config.max_missing_ratio}"
            )
        
        # Validate target column integrity
        target_col = self.config.processing_config.target_column
        if target_col in df.columns:
            if df[target_col].isnull().any():
                raise DataValidationError(f"Target column '{target_col}' contains missing values after cleaning")
    
    def select_and_engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using your friend's programmatic approach with enhancements.
        
        Implements the robust feature selection methodology from your friend's work:
        - Programmatic selection of numeric columns
        - Exclusion of specified columns
        - Validation of feature quality
        
        Args:
            df: Cleaned dataset for feature selection
            
        Returns:
            tuple: (DataFrame with selected features, list of feature column names)
            
        Raises:
            FeatureEngineeringError: If feature selection fails
            
        Example:
            >>> df_features, feature_names = processor.select_and_engineer_features(df_clean)
            >>> print(f"Selected {len(feature_names)} features")
        """
        try:
            self.logger.info("Starting feature selection and engineering")
            
            # 1. Programmatic feature selection (friend's approach)
            if self.config.processing_config.dtype_selection == "number":
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                raise FeatureEngineeringError(f"Unsupported dtype_selection: {self.config.processing_config.dtype_selection}")
            
            # 2. Remove target column and excluded columns
            exclude_cols = [self.config.processing_config.target_column] + self.config.processing_config.exclude_columns
            feature_columns = [col for col in feature_columns if col not in exclude_cols]
            
            if not feature_columns:
                raise FeatureEngineeringError("No valid features found after selection and exclusion")
            
            self.logger.info("Selected %d numeric features", len(feature_columns))
            
            # 3. Validate feature quality
            self._validate_feature_quality(df[feature_columns])
            
            # 4. Store feature information
            self.feature_columns = feature_columns
            self._processing_metadata['feature_engineering'] = {
                'total_features_selected': len(feature_columns),
                'selection_strategy': self.config.processing_config.dtype_selection,
                'excluded_columns': exclude_cols,
                'feature_names': feature_columns
            }
            
            # 5. Create features DataFrame
            df_features = df[feature_columns].copy()
            
            self.logger.info("Feature selection completed: %d features selected", len(feature_columns))
            return df_features, feature_columns
            
        except Exception as e:
            self.logger.error("Feature selection failed: %s", str(e))
            raise FeatureEngineeringError(f"Feature selection failed: {e}") from e
    
    def _validate_feature_quality(self, df_features: pd.DataFrame) -> None:
        """
        Validate quality of selected features.
        
        Args:
            df_features: DataFrame containing selected features
            
        Raises:
            DataValidationError: If feature quality is insufficient
        """
        # Check for constant features
        constant_features = []
        for col in df_features.columns:
            if df_features[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            self.logger.warning("Found %d constant features: %s", 
                              len(constant_features), constant_features[:5])
        
        # Check for highly correlated features (if enabled)
        if (self.config.quality_config.feature_correlation_threshold < 1.0 and 
            len(df_features.columns) > 1):
            
            try:
                corr_matrix = df_features.corr().abs()
                high_corr_pairs = []
                
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if corr_val > self.config.quality_config.feature_correlation_threshold:
                            high_corr_pairs.append((
                                corr_matrix.columns[i], 
                                corr_matrix.columns[j], 
                                corr_val
                            ))
                
                if high_corr_pairs:
                    self.logger.warning("Found %d highly correlated feature pairs (>%.2f)", 
                                      len(high_corr_pairs), 
                                      self.config.quality_config.feature_correlation_threshold)
                    
            except Exception as e:
                self.logger.warning("Correlation analysis failed: %s", str(e))
    
    def create_stratified_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create stratified train/validation/test splits preserving class balance.
        
        Implements your friend's proven approach with enhanced error handling
        and validation.
        
        Args:
            df: Complete dataset with features and target
            
        Returns:
            dict: Dictionary with 'train', 'val', 'test' DataFrames
            
        Raises:
            DataProcessingError: If splitting fails
            DataValidationError: If class balance validation fails
            
        Example:
            >>> splits = processor.create_stratified_splits(df_complete)
            >>> print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        """
        try:
            self.logger.info("Creating stratified data splits")
            
            target_col = self.config.processing_config.target_column
            
            # Prepare features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Validate class distribution
            class_counts = y.value_counts()
            self.logger.info("Target class distribution: %s", class_counts.to_dict())
            
            # Check minimum class samples
            min_class_count = class_counts.min()
            if min_class_count < self.config.quality_config.min_class_samples:
                raise DataValidationError(
                    f"Insufficient samples in minority class: {min_class_count} < {self.config.quality_config.min_class_samples}"
                )
            
            # First split: train vs (val + test)
            val_test_size = self.config.split_config.validation_ratio + self.config.split_config.test_ratio
            
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=val_test_size,
                stratify=y,
                random_state=self.config.split_config.random_state,
                shuffle=self.config.split_config.shuffle
            )
            
            # Second split: val vs test
            test_ratio_adjusted = self.config.split_config.test_ratio / val_test_size
            
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=test_ratio_adjusted,
                stratify=y_temp,
                random_state=self.config.split_config.random_state,
                shuffle=self.config.split_config.shuffle
            )
            
            # Create split datasets
            splits = {
                'train': pd.concat([X_train, y_train], axis=1),
                'val': pd.concat([X_val, y_val], axis=1),
                'test': pd.concat([X_test, y_test], axis=1)
            }
            
            # Validate splits
            self._validate_splits(splits, target_col)
            
            # Log split statistics
            for split_name, split_df in splits.items():
                split_dist = split_df[target_col].value_counts()
                self.logger.info("Split '%s': %d samples, distribution: %s", 
                               split_name, len(split_df), split_dist.to_dict())
            
            # Store split metadata
            self._processing_metadata['splitting'] = {
                'strategy': self.config.split_config.strategy.value,
                'train_size': len(splits['train']),
                'val_size': len(splits['val']),
                'test_size': len(splits['test']),
                'random_state': self.config.split_config.random_state
            }
            
            self.logger.info("Data splitting completed successfully")
            return splits
            
        except Exception as e:
            self.logger.error("Data splitting failed: %s", str(e))
            raise DataProcessingError(f"Data splitting failed: {e}") from e
    
    def _validate_splits(self, splits: Dict[str, pd.DataFrame], target_col: str) -> None:
        """
        Validate that splits maintain proper class balance and size.
        
        Args:
            splits: Dictionary of split DataFrames
            target_col: Name of target column
            
        Raises:
            DataValidationError: If validation fails
        """
        # Check that all classes are present in each split
        original_classes = set()
        for split_df in splits.values():
            original_classes.update(split_df[target_col].unique())
        
        for split_name, split_df in splits.items():
            split_classes = set(split_df[target_col].unique())
            if split_classes != original_classes:
                missing_classes = original_classes - split_classes
                raise DataValidationError(
                    f"Split '{split_name}' missing classes: {missing_classes}"
                )
        
        # Validate split sizes
        total_samples = sum(len(split_df) for split_df in splits.values())
        for split_name, split_df in splits.items():
            if len(split_df) == 0:
                raise DataValidationError(f"Split '{split_name}' is empty")
        
        self.logger.info("Split validation passed: %d total samples across %d splits", 
                        total_samples, len(splits))
    
    def fit_and_apply_scaling(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Fit scaler on training data and apply to all splits.
        
        Implements your friend's critical approach: fit scaler only on training data
        to prevent data leakage, then transform validation and test sets.
        
        Args:
            splits: Dictionary of train/val/test DataFrames
            
        Returns:
            dict: Dictionary of scaled DataFrames
            
        Raises:
            DataProcessingError: If scaling fails
            
        Example:
            >>> scaled_splits = processor.fit_and_apply_scaling(splits)
            >>> # Scaler fitted only on training data, applied to all splits
        """
        try:
            self.logger.info("Fitting and applying feature scaling")
            
            if not self.feature_columns:
                raise DataProcessingError("Feature columns not defined. Run feature selection first.")
            
            target_col = self.config.processing_config.target_column
            
            # Initialize scaler based on configuration
            if self.config.processing_config.scaling_strategy == ScalingStrategy.STANDARD:
                self.scaler = StandardScaler()
            elif self.config.processing_config.scaling_strategy == ScalingStrategy.NONE:
                self.logger.info("Scaling disabled, returning original splits")
                return splits
            else:
                raise DataProcessingError(f"Unsupported scaling strategy: {self.config.processing_config.scaling_strategy}")
            
            # Fit scaler on training data only (critical for preventing data leakage)
            X_train = splits['train'][self.feature_columns]
            self.scaler.fit(X_train)
            
            self.logger.info("Scaler fitted on training data: %d samples, %d features", 
                           len(X_train), len(self.feature_columns))
            
            # Apply scaling to all splits
            scaled_splits = {}
            for split_name, split_df in splits.items():
                # Separate features and target
                X_split = split_df[self.feature_columns]
                y_split = split_df[target_col]
                
                # Scale features
                X_scaled = self.scaler.transform(X_split)
                
                # Create scaled DataFrame
                X_scaled_df = pd.DataFrame(
                    X_scaled, 
                    columns=self.feature_columns, 
                    index=split_df.index
                )
                
                # Combine scaled features with target
                scaled_splits[split_name] = pd.concat([X_scaled_df, y_split], axis=1)
            
            # Validate scaling results
            self._validate_scaling(scaled_splits)
            
            # Store scaling metadata
            self._processing_metadata['scaling'] = {
                'strategy': self.config.processing_config.scaling_strategy.value,
                'scaler_type': type(self.scaler).__name__,
                'feature_count': len(self.feature_columns),
                'scaler_fitted': True
            }
            
            self.logger.info("Feature scaling completed successfully")
            return scaled_splits
            
        except Exception as e:
            self.logger.error("Feature scaling failed: %s", str(e))
            raise DataProcessingError(f"Feature scaling failed: {e}") from e
    
    def _validate_scaling(self, scaled_splits: Dict[str, pd.DataFrame]) -> None:
        """
        Validate that scaling was applied correctly.
        
        Args:
            scaled_splits: Dictionary of scaled DataFrames
            
        Raises:
            DataValidationError: If scaling validation fails
        """
        # Check that training data is properly standardized (mean ≈ 0, std ≈ 1)
        if 'train' in scaled_splits and self.config.processing_config.scaling_strategy == ScalingStrategy.STANDARD:
            train_features = scaled_splits['train'][self.feature_columns]
            
            mean_vals = train_features.mean()
            std_vals = train_features.std()
            
            # Check means are close to 0 (within tolerance)
            high_mean_cols = mean_vals[np.abs(mean_vals) > 0.1].index.tolist()
            if high_mean_cols:
                self.logger.warning("Features with high mean after scaling: %s", high_mean_cols[:5])
            
            # Check standard deviations are close to 1
            non_unit_std_cols = std_vals[np.abs(std_vals - 1.0) > 0.1].index.tolist()
            if non_unit_std_cols:
                self.logger.warning("Features with non-unit std after scaling: %s", non_unit_std_cols[:5])
    
    def save_artifacts(self, scaled_splits: Dict[str, pd.DataFrame]) -> None:
        """
        Save processed datasets and critical artifacts for deployment.
        
        Saves the essential artifacts needed for reproducible model deployment:
        - Processed train/val/test datasets
        - Fitted scaler object  
        - Feature schema JSON
        - Processing metadata
        
        Args:
            scaled_splits: Dictionary of processed DataFrames to save
            
        Raises:
            DataProcessingError: If artifact saving fails
            
        Example:
            >>> processor.save_artifacts(scaled_splits)
            >>> # Artifacts saved for model training and deployment
        """
        try:
            self.logger.info("Saving processed datasets and artifacts")
            
            # Save processed datasets
            for split_name, split_df in scaled_splits.items():
                if split_name == 'train':
                    save_path = self.config.paths.train_path
                elif split_name == 'val':
                    save_path = self.config.paths.validation_path
                elif split_name == 'test':
                    save_path = self.config.paths.test_path
                else:
                    save_path = self.config.paths.processed_data_dir / f"{split_name}.csv"
                
                split_df.to_csv(save_path, index=False)
                self.logger.info("Saved %s dataset: %s (%d samples)", 
                               split_name, save_path, len(split_df))
            
            # Save fitted scaler (critical for deployment)
            if self.scaler is not None:
                joblib.dump(self.scaler, self.config.paths.scaler_path)
                self.logger.info("Saved fitted scaler: %s", self.config.paths.scaler_path)
            
            # Save feature schema (critical for deployment)
            if self.feature_columns is not None:
                with open(self.config.paths.feature_schema_path, 'w') as f:
                    json.dump(self.feature_columns, f, indent=2)
                self.logger.info("Saved feature schema: %s (%d features)", 
                               self.config.paths.feature_schema_path, len(self.feature_columns))
            
            # Save processing metadata
            metadata_path = self.config.paths.processed_data_dir / "processing_metadata.json"
            self._processing_metadata['config_summary'] = self.config.get_summary()
            
            with open(metadata_path, 'w') as f:
                json.dump(self._processing_metadata, f, indent=2)
            self.logger.info("Saved processing metadata: %s", metadata_path)
            
            self.logger.info("All artifacts saved successfully")
            
        except Exception as e:
            self.logger.error("Artifact saving failed: %s", str(e))
            raise DataProcessingError(f"Artifact saving failed: {e}") from e
    
    def process_complete_pipeline(self) -> Dict[str, pd.DataFrame]:
        """
        Execute the complete data processing pipeline end-to-end.
        
        This method orchestrates the entire preprocessing workflow:
        1. Load and validate dataset
        2. Clean and sanitize data  
        3. Select and engineer features
        4. Create stratified splits
        5. Fit and apply scaling
        6. Save all artifacts
        
        Returns:
            dict: Dictionary containing processed train/val/test DataFrames
            
        Raises:
            DataProcessingError: If any pipeline stage fails
            
        Example:
            >>> from pathlib import Path
            >>> config = create_default_config(Path("ADVERSARIAL_IDS_DEEP_LEARNING"))
            >>> processor = DataProcessor(config)
            >>> datasets = processor.process_complete_pipeline()
            >>> print("Pipeline completed successfully!")
            >>> print(f"Training samples: {len(datasets['train'])}")
        """
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING COMPLETE DATA PROCESSING PIPELINE")
            self.logger.info("=" * 60)
            
            # Stage 1: Load and validate
            df_raw = self.load_and_validate_dataset()
            
            # Stage 2: Clean and sanitize  
            df_clean = self.clean_and_sanitize_data(df_raw)
            
            # Stage 3: Feature selection and engineering
            df_features, feature_names = self.select_and_engineer_features(df_clean)
            
            # Combine features with target for splitting
            target_col = self.config.processing_config.target_column
            df_complete = pd.concat([df_features, df_clean[target_col]], axis=1)
            
            # Stage 4: Create stratified splits
            splits = self.create_stratified_splits(df_complete)
            
            # Stage 5: Fit and apply scaling
            scaled_splits = self.fit_and_apply_scaling(splits)
            
            # Stage 6: Save artifacts
            self.save_artifacts(scaled_splits)
            
            self.logger.info("=" * 60)
            self.logger.info("DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 60)
            
            # Summary statistics
            total_samples = sum(len(split_df) for split_df in scaled_splits.values())
            self.logger.info("PIPELINE SUMMARY:")
            self.logger.info("  Total samples processed: %d", total_samples)
            self.logger.info("  Features selected: %d", len(self.feature_columns))
            self.logger.info("  Train samples: %d", len(scaled_splits['train']))
            self.logger.info("  Validation samples: %d", len(scaled_splits['val']))
            self.logger.info("  Test samples: %d", len(scaled_splits['test']))
            self.logger.info("  Artifacts saved: scaler, feature_schema, datasets")
            
            return scaled_splits
            
        except Exception as e:
            self.logger.error("=" * 60)
            self.logger.error("DATA PROCESSING PIPELINE FAILED")
            self.logger.error("Error: %s", str(e))
            self.logger.error("=" * 60)
            raise


# Convenience function for quick pipeline execution
def run_preprocessing_pipeline(project_root: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to run the complete preprocessing pipeline.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        dict: Processed datasets ready for model training
        
    Raises:
        DataProcessingError: If pipeline execution fails
        
    Example:
        >>> datasets = run_preprocessing_pipeline("ADVERSARIAL_IDS_DEEP_LEARNING")
        >>> print("Preprocessing completed!")
    """
    config = create_default_config(project_root)
    processor = DataProcessor(config)
    return processor.process_complete_pipeline()


# Export main classes and functions
__all__ = [
    'DataProcessor',
    'DataProcessingError', 
    'DataValidationError',
    'FeatureEngineeringError',
    'run_preprocessing_pipeline'
]