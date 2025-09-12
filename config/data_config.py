"""
Industrial-Grade Data Configuration Management
===========================================

Centralized configuration for data processing pipeline with comprehensive
validation, security, and reproducibility features.

Author: AI Assistant
Date: September 12, 2025
Version: 1.0.0 (Industrial Grade)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from enum import Enum


class DataSplitStrategy(Enum):
    """Enumeration of supported data splitting strategies."""
    STRATIFIED = "stratified"
    RANDOM = "random"
    TEMPORAL = "temporal"


class ImputationStrategy(Enum):
    """Enumeration of supported missing value imputation strategies."""
    ZERO_FILL = "zero"
    MEDIAN = "median"
    MEAN = "mean"
    MODE = "mode"
    DROP = "drop"


class ScalingStrategy(Enum):
    """Enumeration of supported feature scaling strategies."""
    STANDARD = "standard"
    MIN_MAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


@dataclass(frozen=True)
class DataPaths:
    """
    Immutable configuration for data file paths with validation.
    
    Attributes:
        raw_data_dir: Directory containing raw datasets
        processed_data_dir: Directory for processed/cleaned datasets
        adversarial_data_dir: Directory for adversarial attack datasets
        dataset_filename: Primary dataset filename
        backup_dir: Directory for data backups (optional)
        
    Raises:
        ValueError: If required directories don't exist or are not accessible
        PermissionError: If insufficient permissions for data operations
    """
    
    raw_data_dir: Path
    processed_data_dir: Path
    adversarial_data_dir: Path
    dataset_filename: str
    backup_dir: Optional[Path] = None
    
    def __post_init__(self) -> None:
        """
        Validate all data paths exist and are accessible.
        
        Raises:
            FileNotFoundError: If required directories don't exist
            PermissionError: If directories are not writable
        """
        for directory in [self.raw_data_dir, self.processed_data_dir, self.adversarial_data_dir]:
            if not directory.exists():
                raise FileNotFoundError(f"Data directory does not exist: {directory}")
            if not directory.is_dir():
                raise NotADirectoryError(f"Path is not a directory: {directory}")
            if not (directory.stat().st_mode & 0o200):  # Check write permission
                raise PermissionError(f"Directory is not writable: {directory}")
    
    @property
    def dataset_path(self) -> Path:
        """Full path to the primary dataset file."""
        return self.raw_data_dir / self.dataset_filename
    
    @property
    def train_path(self) -> Path:
        """Path to training dataset."""
        return self.processed_data_dir / "train.csv"
    
    @property
    def validation_path(self) -> Path:
        """Path to validation dataset."""
        return self.processed_data_dir / "val.csv"
    
    @property
    def test_path(self) -> Path:
        """Path to test dataset."""
        return self.processed_data_dir / "test.csv"
    
    @property
    def scaler_path(self) -> Path:
        """Path to fitted StandardScaler artifact."""
        return self.processed_data_dir / "scaler.joblib"
    
    @property
    def feature_schema_path(self) -> Path:
        """Path to feature schema JSON file."""
        return self.processed_data_dir / "feature_columns.json"


@dataclass
class DataSplitConfig:
    """
    Configuration for train/validation/test data splitting.
    
    Attributes:
        train_ratio: Proportion of data for training (0 < train_ratio < 1)
        validation_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        strategy: Splitting strategy (stratified recommended for imbalanced data)
        random_state: Random seed for reproducibility
        shuffle: Whether to shuffle data before splitting
        
    Raises:
        ValueError: If ratios don't sum to 1.0 or are invalid
    """
    
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    strategy: DataSplitStrategy = DataSplitStrategy.STRATIFIED
    random_state: int = 42
    shuffle: bool = True
    
    def __post_init__(self) -> None:
        """
        Validate data split configuration.
        
        Raises:
            ValueError: If ratios are invalid or don't sum to 1.0
        """
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Data split ratios must sum to 1.0, got {total_ratio}")
        
        for ratio_name, ratio_value in [
            ("train_ratio", self.train_ratio),
            ("validation_ratio", self.validation_ratio), 
            ("test_ratio", self.test_ratio)
        ]:
            if not (0.0 < ratio_value < 1.0):
                raise ValueError(f"{ratio_name} must be between 0 and 1, got {ratio_value}")


@dataclass
class DataProcessingConfig:
    """
    Comprehensive configuration for data preprocessing pipeline.
    
    Attributes:
        target_column: Name of the target/label column
        exclude_columns: Columns to exclude from feature selection
        imputation_strategy: How to handle missing values
        scaling_strategy: Feature scaling method
        handle_infinity: Whether to handle infinity values
        infinity_replacement: Value to replace infinity with
        validation_split_first: Whether to split before or after cleaning
        preserve_raw_copy: Whether to keep backup of raw data
        
    Raises:
        ValueError: If configuration parameters are invalid
    """
    
    target_column: str = "Label"
    exclude_columns: List[str] = field(default_factory=list)
    imputation_strategy: ImputationStrategy = ImputationStrategy.ZERO_FILL
    scaling_strategy: ScalingStrategy = ScalingStrategy.STANDARD
    handle_infinity: bool = True
    infinity_replacement: float = 0.0
    validation_split_first: bool = False
    preserve_raw_copy: bool = True
    dtype_selection: str = "number"  # For pd.select_dtypes()
    
    def __post_init__(self) -> None:
        """
        Validate data processing configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not self.target_column:
            raise ValueError("target_column cannot be empty")
        
        if self.dtype_selection not in ["number", "object", "category"]:
            raise ValueError(f"Invalid dtype_selection: {self.dtype_selection}")


@dataclass
class QualityValidationConfig:
    """
    Configuration for data quality validation and monitoring.
    
    Attributes:
        min_samples: Minimum number of samples required
        max_missing_ratio: Maximum allowed ratio of missing values per column
        min_class_samples: Minimum samples required per class
        max_class_imbalance: Maximum allowed class imbalance ratio
        feature_correlation_threshold: Threshold for detecting highly correlated features
        outlier_detection_enabled: Whether to detect and log outliers
        duplicate_detection_enabled: Whether to check for duplicate rows
        
    Raises:
        ValueError: If validation thresholds are invalid
    """
    
    min_samples: int = 1000
    max_missing_ratio: float = 0.50
    min_class_samples: int = 100
    max_class_imbalance: float = 0.99
    feature_correlation_threshold: float = 0.95
    outlier_detection_enabled: bool = True
    duplicate_detection_enabled: bool = True
    
    def __post_init__(self) -> None:
        """
        Validate quality validation configuration.
        
        Raises:
            ValueError: If validation parameters are invalid
        """
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        
        if not (0.0 <= self.max_missing_ratio <= 1.0):
            raise ValueError("max_missing_ratio must be between 0 and 1")
        
        if self.min_class_samples <= 0:
            raise ValueError("min_class_samples must be positive")
        
        if not (0.5 <= self.max_class_imbalance <= 1.0):
            raise ValueError("max_class_imbalance must be between 0.5 and 1.0")


@dataclass
class DataConfig:
    """
    Master configuration class for the entire data processing pipeline.
    
    This class aggregates all data-related configuration and provides
    validation, serialization, and environment-specific overrides.
    
    Attributes:
        paths: Data file paths and directories
        split_config: Train/validation/test split configuration
        processing_config: Data cleaning and preprocessing options
        quality_config: Data quality validation parameters
        logging_level: Logging verbosity for data operations
        
    Example:
        >>> from pathlib import Path
        >>> paths = DataPaths(
        ...     raw_data_dir=Path("data/raw"),
        ...     processed_data_dir=Path("data/processed"),
        ...     adversarial_data_dir=Path("data/adversarial"),
        ...     dataset_filename="dataset.csv"
        ... )
        >>> config = DataConfig(paths=paths)
        >>> # Use config throughout data pipeline
    """
    
    paths: DataPaths
    split_config: DataSplitConfig = field(default_factory=DataSplitConfig)
    processing_config: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    quality_config: QualityValidationConfig = field(default_factory=QualityValidationConfig)
    logging_level: int = logging.INFO
    
    def __post_init__(self) -> None:
        """
        Validate complete data configuration and initialize logging.
        
        Raises:
            ValueError: If any configuration component is invalid
        """
        # Validate that primary dataset exists
        if not self.paths.dataset_path.exists():
            raise FileNotFoundError(f"Primary dataset not found: {self.paths.dataset_path}")
        
        # Initialize logging for data operations
        logging.basicConfig(
            level=self.logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Data configuration initialized successfully")
    
    def validate_environment(self) -> bool:
        """
        Validate that the environment is ready for data processing.
        
        Returns:
            bool: True if environment is valid, False otherwise
            
        Raises:
            EnvironmentError: If critical environment issues are detected
        """
        try:
            # Check disk space (require at least 1GB free)
            import shutil
            free_space = shutil.disk_usage(self.paths.processed_data_dir).free
            if free_space < 1_000_000_000:  # 1GB in bytes
                raise EnvironmentError(f"Insufficient disk space: {free_space / 1e9:.2f}GB available")
            
            # Check Python dependencies
            required_packages = ['pandas', 'numpy', 'scikit-learn', 'joblib']
            for package in required_packages:
                try:
                    __import__(package)
                except ImportError as e:
                    raise EnvironmentError(f"Required package not installed: {package}") from e
            
            self.logger.info("Environment validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            raise
    
    def get_summary(self) -> Dict[str, Union[str, int, float]]:
        """
        Generate a summary of the data configuration.
        
        Returns:
            dict: Configuration summary with key parameters
        """
        return {
            "dataset_path": str(self.paths.dataset_path),
            "train_ratio": self.split_config.train_ratio,
            "validation_ratio": self.split_config.validation_ratio,
            "test_ratio": self.split_config.test_ratio,
            "split_strategy": self.split_config.strategy.value,
            "imputation_strategy": self.processing_config.imputation_strategy.value,
            "scaling_strategy": self.processing_config.scaling_strategy.value,
            "target_column": self.processing_config.target_column,
            "random_state": self.split_config.random_state,
            "min_samples": self.quality_config.min_samples,
            "max_missing_ratio": self.quality_config.max_missing_ratio
        }


# Factory function for easy configuration creation
def create_default_config(project_root: Union[str, Path]) -> DataConfig:
    """
    Create a default data configuration for the project.
    
    Args:
        project_root: Path to the project root directory
        
    Returns:
        DataConfig: Configured data configuration instance
        
    Raises:
        ValueError: If project_root is invalid
        FileNotFoundError: If required directories don't exist
        
    Example:
        >>> config = create_default_config("/path/to/ADVERSARIAL_IDS_DEEP_LEARNING")
        >>> # Configuration ready for use in preprocessing pipeline
    """
    project_root = Path(project_root)
    
    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")
    
    paths = DataPaths(
        raw_data_dir=project_root / "Combined_DS",
        processed_data_dir=project_root / "data" / "processed",
        adversarial_data_dir=project_root / "data" / "adversarial",
        dataset_filename="dataset.csv"
    )
    
    return DataConfig(paths=paths)


# Export main configuration classes
__all__ = [
    'DataConfig',
    'DataPaths', 
    'DataSplitConfig',
    'DataProcessingConfig',
    'QualityValidationConfig',
    'DataSplitStrategy',
    'ImputationStrategy', 
    'ScalingStrategy',
    'create_default_config'
]