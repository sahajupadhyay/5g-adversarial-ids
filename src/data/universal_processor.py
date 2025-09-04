"""
Universal Data Processing System for 5G Adversarial IDS

This module can consume ANY type of data and transform it into the format
required by the 5G IDS models. It's the mandatory first stage of all pipelines.

Author: Capstone Team
Date: September 4, 2025
"""

import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle
import h5py
import sqlite3
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class UniversalDataProcessor:
    """
    Universal data processor that can handle any type of input data
    and transform it for 5G IDS model consumption
    """
    
    def __init__(self, target_features=43, target_samples_min=100, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.target_features = target_features
        self.target_samples_min = target_samples_min
        
        # Processing components
        self.data_loader = DataLoader(logger=self.logger)
        self.feature_engineer = FeatureEngineer(logger=self.logger)
        self.data_cleaner = DataCleaner(logger=self.logger)
        self.format_standardizer = FormatStandardizer(target_features, logger=self.logger)
        
        # Processing history
        self.processing_history = []
        self.metadata = {}
        
    def process_data(self, data_source: Union[str, Path, dict, pd.DataFrame], 
                    config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Universal data processing pipeline - handles ANY input data
        
        Args:
            data_source: Path to file, DataFrame, dict, or any data structure
            config: Optional processing configuration
            
        Returns:
            Processed data ready for 5G IDS models
        """
        self.logger.info("🚀 UNIVERSAL DATA PROCESSING STARTED")
        self.logger.info("=" * 60)
        
        config = config or {}
        
        try:
            # Stage 1: Data Loading and Detection
            self.logger.info("📥 STAGE 1: Data Loading and Format Detection")
            raw_data, data_info = self.data_loader.load_data(data_source)
            self.metadata.update(data_info)
            
            # Stage 2: Data Cleaning and Quality Assessment
            self.logger.info("🧹 STAGE 2: Data Cleaning and Quality Assessment")
            clean_data, cleaning_report = self.data_cleaner.clean_data(raw_data, config.get('cleaning', {}))
            self.metadata['cleaning_report'] = cleaning_report
            
            # Stage 3: Feature Engineering and Enhancement
            self.logger.info("⚙️ STAGE 3: Feature Engineering and Enhancement")
            engineered_data, feature_report = self.feature_engineer.engineer_features(
                clean_data, config.get('feature_engineering', {})
            )
            self.metadata['feature_report'] = feature_report
            
            # Stage 4: Format Standardization for 5G IDS
            self.logger.info("🎯 STAGE 4: Format Standardization for 5G IDS")
            X_processed, y_processed, standardization_report = self.format_standardizer.standardize_format(
                engineered_data, config.get('target_column')
            )
            self.metadata['standardization_report'] = standardization_report
            
            # Stage 5: Final Validation
            self.logger.info("✅ STAGE 5: Final Validation and Quality Check")
            validation_report = self._validate_processed_data(X_processed, y_processed)
            self.metadata['validation_report'] = validation_report
            
            # Compile final results
            processed_data = {
                'X_train': X_processed,
                'y_train': y_processed,
                'metadata': self.metadata,
                'processing_pipeline': self._get_processing_pipeline(),
                'data_info': {
                    'original_shape': raw_data.shape if hasattr(raw_data, 'shape') else 'N/A',
                    'processed_shape': X_processed.shape,
                    'feature_count': X_processed.shape[1],
                    'sample_count': X_processed.shape[0],
                    'has_labels': y_processed is not None,
                    'label_classes': np.unique(y_processed).tolist() if y_processed is not None else None
                }
            }
            
            self.logger.info("🎉 UNIVERSAL DATA PROCESSING COMPLETED SUCCESSFULLY")
            self.logger.info(f"📊 Final Dataset: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"❌ Data processing failed: {str(e)}")
            raise
    
    def _validate_processed_data(self, X, y):
        """Validate the processed data meets 5G IDS requirements"""
        validation_report = {
            'feature_count_valid': X.shape[1] == self.target_features,
            'sample_count_adequate': X.shape[0] >= self.target_samples_min,
            'no_missing_values': not np.isnan(X).any(),
            'no_infinite_values': not np.isinf(X).any(),
            'feature_variance_adequate': np.var(X, axis=0).min() > 1e-8,
            'labels_valid': y is None or len(np.unique(y)) <= 10
        }
        
        validation_report['overall_valid'] = all(validation_report.values())
        
        if validation_report['overall_valid']:
            self.logger.info("✅ Data validation passed - Ready for 5G IDS models")
        else:
            failed_checks = [k for k, v in validation_report.items() if not v and k != 'overall_valid']
            self.logger.warning(f"⚠️ Validation issues: {failed_checks}")
        
        return validation_report
    
    def _get_processing_pipeline(self):
        """Get the processing pipeline for reproducibility"""
        return {
            'data_loader': self.data_loader.get_config(),
            'data_cleaner': self.data_cleaner.get_config(),
            'feature_engineer': self.feature_engineer.get_config(),
            'format_standardizer': self.format_standardizer.get_config()
        }
    
    def save_processing_pipeline(self, output_path: str):
        """Save the processing pipeline for future use"""
        pipeline_data = {
            'processor_config': {
                'target_features': self.target_features,
                'target_samples_min': self.target_samples_min
            },
            'processing_pipeline': self._get_processing_pipeline(),
            'metadata': self.metadata
        }
        
        with open(output_path, 'w') as f:
            json.dump(pipeline_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Processing pipeline saved: {output_path}")


class DataLoader:
    """Handles loading data from any source format"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.supported_formats = {
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.parquet': self._load_parquet,
            '.pkl': self._load_pickle,
            '.h5': self._load_hdf5,
            '.xml': self._load_xml,
            '.db': self._load_sqlite,
            '.sql': self._load_sqlite,
            '.txt': self._load_text,
            '.log': self._load_log,
            '.pcap': self._load_pcap,
            '.npy': self._load_numpy,
            '.npz': self._load_numpy_compressed
        }
    
    def load_data(self, data_source) -> Tuple[pd.DataFrame, Dict]:
        """Load data from any source"""
        self.logger.info(f"🔍 Detecting data source type...")
        
        # Handle different input types
        if isinstance(data_source, pd.DataFrame):
            self.logger.info("📊 Input: Pandas DataFrame")
            return data_source, {'source_type': 'dataframe', 'format': 'pandas'}
        
        elif isinstance(data_source, dict):
            self.logger.info("📋 Input: Dictionary")
            df = pd.DataFrame(data_source)
            return df, {'source_type': 'dictionary', 'format': 'dict'}
        
        elif isinstance(data_source, (str, Path)):
            file_path = Path(data_source)
            
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            file_extension = file_path.suffix.lower()
            self.logger.info(f"📁 Input: File ({file_extension})")
            
            if file_extension in self.supported_formats:
                loader_func = self.supported_formats[file_extension]
                df = loader_func(file_path)
                
                data_info = {
                    'source_type': 'file',
                    'format': file_extension,
                    'file_path': str(file_path),
                    'file_size': file_path.stat().st_size,
                    'original_shape': df.shape
                }
                
                self.logger.info(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
                return df, data_info
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        
        else:
            # Try to convert to DataFrame
            try:
                df = pd.DataFrame(data_source)
                self.logger.info("🔄 Input: Converted to DataFrame")
                return df, {'source_type': 'converted', 'format': 'auto'}
            except Exception as e:
                raise ValueError(f"Cannot process data source type: {type(data_source)}")
    
    def _load_csv(self, file_path):
        """Load CSV file with intelligent parsing"""
        try:
            # Try different separators and encodings
            for sep in [',', ';', '\t', '|']:
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, encoding=encoding, low_memory=False)
                        if df.shape[1] > 1:  # Valid multi-column data
                            self.logger.info(f"✅ CSV loaded with separator '{sep}', encoding '{encoding}'")
                            return df
                    except:
                        continue
            
            # Fallback: basic load
            return pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {e}")
    
    def _load_json(self, file_path):
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.json_normalize(data) if any(isinstance(item, dict) for item in data) else pd.DataFrame(data)
        elif isinstance(data, dict):
            # Flatten nested JSON if needed
            df = pd.json_normalize(data)
        else:
            raise ValueError("JSON must contain list or dict")
        
        # Clean up any remaining dict/list columns to prevent hashing issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains dict/list objects
                sample_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample_val, (dict, list)):
                    # Convert to string representation
                    df[col] = df[col].astype(str)
                    
        return df
    
    def _load_excel(self, file_path):
        """Load Excel file"""
        return pd.read_excel(file_path, sheet_name=0)  # Load first sheet
    
    def _load_parquet(self, file_path):
        """Load Parquet file"""
        return pd.read_parquet(file_path)
    
    def _load_pickle(self, file_path):
        """Load pickle file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    
    def _load_hdf5(self, file_path):
        """Load HDF5 file"""
        with h5py.File(file_path, 'r') as f:
            # Get first dataset
            key = list(f.keys())[0]
            data = f[key][:]
        return pd.DataFrame(data)
    
    def _load_xml(self, file_path):
        """Load XML file"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Extract data from XML (basic implementation)
        data = []
        for child in root:
            row = {elem.tag: elem.text for elem in child}
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _load_sqlite(self, file_path):
        """Load SQLite database"""
        conn = sqlite3.connect(file_path)
        
        # Get first table
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if tables:
            table_name = tables[0][0]
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        else:
            raise ValueError("No tables found in database")
        
        conn.close()
        return df
    
    def _load_text(self, file_path):
        """Load text file as structured data"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Try to parse as delimited data
        for delimiter in ['\t', ',', ';', '|', ' ']:
            try:
                data = []
                for line in lines:
                    parts = line.strip().split(delimiter)
                    if len(parts) > 1:
                        data.append(parts)
                
                if data:
                    return pd.DataFrame(data[1:], columns=data[0])
            except:
                continue
        
        # Fallback: treat as single column
        return pd.DataFrame({'text': [line.strip() for line in lines]})
    
    def _load_log(self, file_path):
        """Load log file and extract features"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Extract basic log features
        data = []
        for line in lines:
            features = {
                'line_length': len(line),
                'word_count': len(line.split()),
                'has_timestamp': any(char.isdigit() for char in line[:20]),
                'has_ip': '.' in line and any(part.isdigit() for part in line.split('.')),
                'log_level': 'ERROR' if 'ERROR' in line.upper() else 'INFO' if 'INFO' in line.upper() else 'OTHER'
            }
            data.append(features)
        
        return pd.DataFrame(data)
    
    def _load_pcap(self, file_path):
        """Load PCAP file (network packet capture)"""
        try:
            import dpkt
            with open(file_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                
                data = []
                for timestamp, buf in pcap:
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if isinstance(eth.data, dpkt.ip.IP):
                            ip = eth.data
                            features = {
                                'timestamp': timestamp,
                                'packet_length': len(buf),
                                'src_ip': socket.inet_ntoa(ip.src),
                                'dst_ip': socket.inet_ntoa(ip.dst),
                                'protocol': ip.p,
                                'ip_length': ip.len
                            }
                            data.append(features)
                    except:
                        continue
                
                return pd.DataFrame(data)
        except ImportError:
            self.logger.warning("dpkt not available - treating PCAP as binary")
            return self._load_binary_as_features(file_path)
    
    def _load_numpy(self, file_path):
        """Load NumPy array"""
        data = np.load(file_path)
        return pd.DataFrame(data)
    
    def _load_numpy_compressed(self, file_path):
        """Load compressed NumPy file"""
        data = np.load(file_path)
        # Get first array
        key = list(data.keys())[0]
        return pd.DataFrame(data[key])
    
    def _load_binary_as_features(self, file_path):
        """Extract features from binary files"""
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Extract basic binary features
        features = {
            'file_size': len(content),
            'zero_bytes': content.count(0),
            'entropy': self._calculate_entropy(content),
            'ascii_ratio': sum(32 <= b <= 126 for b in content) / len(content)
        }
        
        return pd.DataFrame([features])
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data"""
        from collections import Counter
        counts = Counter(data)
        probs = [count/len(data) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def get_config(self):
        """Get loader configuration"""
        return {
            'supported_formats': list(self.supported_formats.keys()),
            'type': 'DataLoader'
        }


class DataCleaner:
    """Handles data cleaning and quality improvement"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.imputers = {}
        self.outlier_detectors = {}
    
    def clean_data(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Clean and improve data quality"""
        self.logger.info(f"🧹 Cleaning dataset with {df.shape[0]} rows, {df.shape[1]} columns")
        
        cleaning_report = {
            'original_shape': df.shape,
            'operations_performed': []
        }
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # 1. Handle missing values
        df_clean, missing_report = self._handle_missing_values(df_clean, config.get('missing_strategy', 'auto'))
        cleaning_report['missing_values'] = missing_report
        cleaning_report['operations_performed'].append('missing_value_imputation')
        
        # 2. Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)
        if duplicates_removed > 0:
            self.logger.info(f"🗑️ Removed {duplicates_removed} duplicate rows")
            cleaning_report['duplicates_removed'] = duplicates_removed
            cleaning_report['operations_performed'].append('duplicate_removal')
        
        # 3. Handle outliers
        df_clean, outlier_report = self._handle_outliers(df_clean, config.get('outlier_strategy', 'iqr'))
        cleaning_report['outliers'] = outlier_report
        cleaning_report['operations_performed'].append('outlier_handling')
        
        # 4. Fix data types
        df_clean, dtype_report = self._fix_data_types(df_clean)
        cleaning_report['data_types'] = dtype_report
        cleaning_report['operations_performed'].append('dtype_optimization')
        
        # 5. Handle invalid values
        df_clean, invalid_report = self._handle_invalid_values(df_clean)
        cleaning_report['invalid_values'] = invalid_report
        cleaning_report['operations_performed'].append('invalid_value_handling')
        
        cleaning_report['final_shape'] = df_clean.shape
        cleaning_report['quality_score'] = self._calculate_quality_score(df_clean)
        
        self.logger.info(f"✅ Cleaning completed: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        self.logger.info(f"📊 Data quality score: {cleaning_report['quality_score']:.2f}/1.0")
        
        return df_clean, cleaning_report
    
    def _handle_missing_values(self, df, strategy):
        """Handle missing values with intelligent strategies"""
        missing_report = {
            'columns_with_missing': {},
            'strategy_used': strategy,
            'imputation_applied': []
        }
        
        # Identify columns with missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        for col in missing_cols:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            missing_report['columns_with_missing'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            # Choose imputation strategy based on data type and missing percentage
            if missing_pct > 50:
                # Too many missing values - drop column
                df = df.drop(columns=[col])
                missing_report['imputation_applied'].append(f"{col}: dropped (>{missing_pct:.1f}% missing)")
                self.logger.info(f"🗑️ Dropped column '{col}' ({missing_pct:.1f}% missing)")
                
            elif df[col].dtype in ['object', 'category']:
                # Categorical data - mode imputation
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
                missing_report['imputation_applied'].append(f"{col}: mode imputation ({mode_value})")
                
            else:
                # Numerical data - choose strategy
                if strategy == 'mean' or (strategy == 'auto' and missing_pct < 10):
                    df[col] = df[col].fillna(df[col].mean())
                    missing_report['imputation_applied'].append(f"{col}: mean imputation")
                elif strategy == 'median' or (strategy == 'auto' and missing_pct < 20):
                    df[col] = df[col].fillna(df[col].median())
                    missing_report['imputation_applied'].append(f"{col}: median imputation")
                else:
                    # KNN imputation for complex cases
                    imputer = KNNImputer(n_neighbors=5)
                    df[col] = imputer.fit_transform(df[[col]]).flatten()
                    missing_report['imputation_applied'].append(f"{col}: KNN imputation")
        
        return df, missing_report
    
    def _handle_outliers(self, df, strategy):
        """Handle outliers using various methods"""
        outlier_report = {
            'strategy_used': strategy,
            'outliers_detected': {},
            'outliers_handled': []
        }
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if strategy == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_report['outliers_detected'][col] = outlier_count
                    
                    # Cap outliers instead of removing
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    outlier_report['outliers_handled'].append(f"{col}: {outlier_count} outliers capped")
            
            elif strategy == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > 3]
                outlier_count = len(outliers)
                
                if outlier_count > 0:
                    outlier_report['outliers_detected'][col] = outlier_count
                    # Remove extreme outliers
                    df = df[z_scores <= 3]
                    outlier_report['outliers_handled'].append(f"{col}: {outlier_count} outliers removed")
        
        return df, outlier_report
    
    def _fix_data_types(self, df):
        """Optimize data types for memory and performance"""
        dtype_report = {
            'original_memory': df.memory_usage(deep=True).sum(),
            'conversions_applied': []
        }
        
        for col in df.columns:
            original_dtype = str(df[col].dtype)
            
            # Convert object columns that are actually numeric
            if df[col].dtype == 'object':
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if not numeric_series.isnull().all():
                    df[col] = numeric_series
                    dtype_report['conversions_applied'].append(f"{col}: object → numeric")
            
            # Optimize integer types
            elif df[col].dtype in ['int64']:
                min_val = df[col].min()
                max_val = df[col].max()
                
                if min_val >= 0:
                    if max_val < 255:
                        df[col] = df[col].astype('uint8')
                        dtype_report['conversions_applied'].append(f"{col}: int64 → uint8")
                    elif max_val < 65535:
                        df[col] = df[col].astype('uint16')
                        dtype_report['conversions_applied'].append(f"{col}: int64 → uint16")
                else:
                    if min_val > -128 and max_val < 127:
                        df[col] = df[col].astype('int8')
                        dtype_report['conversions_applied'].append(f"{col}: int64 → int8")
                    elif min_val > -32768 and max_val < 32767:
                        df[col] = df[col].astype('int16')
                        dtype_report['conversions_applied'].append(f"{col}: int64 → int16")
            
            # Optimize float types
            elif df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
                if str(df[col].dtype) != original_dtype:
                    dtype_report['conversions_applied'].append(f"{col}: {original_dtype} → {df[col].dtype}")
        
        dtype_report['final_memory'] = df.memory_usage(deep=True).sum()
        dtype_report['memory_reduction'] = (dtype_report['original_memory'] - dtype_report['final_memory']) / dtype_report['original_memory'] * 100
        
        if dtype_report['memory_reduction'] > 0:
            self.logger.info(f"💾 Memory usage reduced by {dtype_report['memory_reduction']:.1f}%")
        
        return df, dtype_report
    
    def _handle_invalid_values(self, df):
        """Handle invalid values like inf, -inf, very large numbers"""
        invalid_report = {
            'infinite_values_found': {},
            'large_values_capped': {},
            'operations_performed': []
        }
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Handle infinite values
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                invalid_report['infinite_values_found'][col] = inf_count
                
                # Replace inf with median
                median_val = df[col][np.isfinite(df[col])].median()
                df[col] = df[col].replace([np.inf, -np.inf], median_val)
                invalid_report['operations_performed'].append(f"{col}: replaced {inf_count} infinite values")
            
            # Handle very large values (potential data entry errors)
            if df[col].dtype.kind in 'fc':  # float or complex
                threshold = np.finfo(df[col].dtype).max / 1000  # 0.1% of max value
                large_values = (df[col].abs() > threshold).sum()
                
                if large_values > 0:
                    invalid_report['large_values_capped'][col] = large_values
                    # Cap at 99.9th percentile
                    cap_value = df[col].quantile(0.999)
                    df[col] = df[col].clip(upper=cap_value)
                    invalid_report['operations_performed'].append(f"{col}: capped {large_values} large values")
        
        return df, invalid_report
    
    def _calculate_quality_score(self, df):
        """Calculate overall data quality score"""
        scores = []
        
        # Completeness score (no missing values)
        completeness = 1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores.append(completeness)
        
        # Consistency score (appropriate data types)
        consistent_types = sum(1 for col in df.columns if df[col].dtype != 'object') / len(df.columns)
        scores.append(consistent_types)
        
        # Validity score (no infinite or null values)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            validity = sum(1 for col in numerical_cols if not np.isinf(df[col]).any()) / len(numerical_cols)
            scores.append(validity)
        
        return np.mean(scores)
    
    def get_config(self):
        """Get cleaner configuration"""
        return {
            'type': 'DataCleaner',
            'available_strategies': ['auto', 'mean', 'median', 'knn', 'iqr', 'zscore']
        }


class FeatureEngineer:
    """Handles feature engineering and enhancement"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.encoders = {}
        self.scalers = {}
    
    def engineer_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict]:
        """Engineer and enhance features"""
        self.logger.info(f"⚙️ Engineering features for {df.shape[1]} columns")
        
        feature_report = {
            'original_features': df.shape[1],
            'operations_performed': [],
            'new_features_created': [],
            'features_removed': []
        }
        
        df_engineered = df.copy()
        
        # 1. Encode categorical variables
        df_engineered, encoding_report = self._encode_categorical_features(df_engineered, config.get('encoding_strategy', 'auto'))
        feature_report['encoding'] = encoding_report
        feature_report['operations_performed'].append('categorical_encoding')
        
        # 2. Create time-based features if datetime columns exist
        df_engineered, time_report = self._create_time_features(df_engineered)
        if time_report['features_created']:
            feature_report['time_features'] = time_report
            feature_report['operations_performed'].append('time_feature_extraction')
        
        # 3. Create interaction features
        df_engineered, interaction_report = self._create_interaction_features(df_engineered, config.get('max_interactions', 5))
        if interaction_report['features_created']:
            feature_report['interaction_features'] = interaction_report
            feature_report['operations_performed'].append('interaction_features')
        
        # 4. Create statistical features
        df_engineered, stats_report = self._create_statistical_features(df_engineered)
        if stats_report['features_created']:
            feature_report['statistical_features'] = stats_report
            feature_report['operations_performed'].append('statistical_features')
        
        # 5. Create polynomial features (if enabled)
        if config.get('create_polynomial', False):
            df_engineered, poly_report = self._create_polynomial_features(df_engineered, config.get('poly_degree', 2))
            feature_report['polynomial_features'] = poly_report
            feature_report['operations_performed'].append('polynomial_features')
        
        feature_report['final_features'] = df_engineered.shape[1]
        feature_report['features_added'] = feature_report['final_features'] - feature_report['original_features']
        
        self.logger.info(f"✅ Feature engineering completed: {feature_report['features_added']} features added")
        
        return df_engineered, feature_report
    
    def _encode_categorical_features(self, df, strategy):
        """Encode categorical features"""
        encoding_report = {
            'strategy_used': strategy,
            'encodings_applied': [],
            'features_created': 0
        }
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            unique_values = df[col].nunique()
            
            if strategy == 'auto':
                # Choose encoding based on cardinality
                if unique_values <= 2:
                    # Binary encoding
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                    encoding_report['encodings_applied'].append(f"{col}: label encoding")
                    
                elif unique_values <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = df.drop(columns=[col])
                    df = pd.concat([df, dummies], axis=1)
                    encoding_report['encodings_applied'].append(f"{col}: one-hot encoding ({unique_values} categories)")
                    encoding_report['features_created'] += unique_values - 1
                    
                else:
                    # Target encoding for high cardinality (simplified version)
                    value_counts = df[col].value_counts()
                    encoding_map = {val: idx for idx, val in enumerate(value_counts.index)}
                    df[col] = df[col].map(encoding_map).fillna(-1)
                    encoding_report['encodings_applied'].append(f"{col}: frequency encoding")
            
            elif strategy == 'label':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                encoding_report['encodings_applied'].append(f"{col}: label encoding")
                
            elif strategy == 'onehot':
                dummies = pd.get_dummies(df[col], prefix=col)
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
                encoding_report['encodings_applied'].append(f"{col}: one-hot encoding")
                encoding_report['features_created'] += unique_values - 1
        
        return df, encoding_report
    
    def _create_time_features(self, df):
        """Extract features from datetime columns"""
        time_report = {
            'features_created': [],
            'datetime_columns_found': []
        }
        
        # Try to identify datetime columns
        datetime_cols = []
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    datetime_cols.append(col)
                    time_report['datetime_columns_found'].append(col)
                except:
                    continue
        
        for col in datetime_cols:
            # Extract time components
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)
            
            time_features = [f"{col}_year", f"{col}_month", f"{col}_day", 
                           f"{col}_hour", f"{col}_dayofweek", f"{col}_is_weekend"]
            time_report['features_created'].extend(time_features)
            
            # Remove original datetime column
            df = df.drop(columns=[col])
        
        return df, time_report
    
    def _create_interaction_features(self, df, max_interactions):
        """Create interaction features between numerical columns"""
        interaction_report = {
            'features_created': [],
            'max_interactions': max_interactions
        }
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Limit to prevent explosion of features
        if len(numerical_cols) > 10:
            # Select most important features based on variance
            variances = df[numerical_cols].var().sort_values(ascending=False)
            numerical_cols = variances.head(10).index.tolist()
        
        interaction_count = 0
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                if interaction_count >= max_interactions:
                    break
                
                # Create multiplication interaction
                interaction_name = f"{col1}_x_{col2}"
                df[interaction_name] = df[col1] * df[col2]
                interaction_report['features_created'].append(interaction_name)
                interaction_count += 1
        
        return df, interaction_report
    
    def _create_statistical_features(self, df):
        """Create statistical features for groups of related columns"""
        stats_report = {
            'features_created': [],
            'statistical_operations': []
        }
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) >= 3:
            # Create row-wise statistics
            df['row_mean'] = df[numerical_cols].mean(axis=1)
            df['row_std'] = df[numerical_cols].std(axis=1)
            df['row_min'] = df[numerical_cols].min(axis=1)
            df['row_max'] = df[numerical_cols].max(axis=1)
            df['row_range'] = df['row_max'] - df['row_min']
            
            stats_features = ['row_mean', 'row_std', 'row_min', 'row_max', 'row_range']
            stats_report['features_created'].extend(stats_features)
            stats_report['statistical_operations'] = ['mean', 'std', 'min', 'max', 'range']
        
        return df, stats_report
    
    def _create_polynomial_features(self, df, degree):
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly_report = {
            'degree': degree,
            'features_created': [],
            'original_feature_count': df.shape[1]
        }
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Limit to prevent explosion - use only most important features
        if len(numerical_cols) > 5:
            variances = df[numerical_cols].var().sort_values(ascending=False)
            selected_cols = variances.head(5).index.tolist()
        else:
            selected_cols = numerical_cols.tolist()
        
        if selected_cols:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(df[selected_cols])
            poly_feature_names = poly.get_feature_names_out(selected_cols)
            
            # Add new polynomial features
            for i, name in enumerate(poly_feature_names):
                if name not in selected_cols:  # Skip original features
                    df[f"poly_{name}"] = poly_features[:, i]
                    poly_report['features_created'].append(f"poly_{name}")
        
        poly_report['final_feature_count'] = df.shape[1]
        
        return df, poly_report
    
    def get_config(self):
        """Get feature engineer configuration"""
        return {
            'type': 'FeatureEngineer',
            'encoders': list(self.encoders.keys()),
            'scalers': list(self.scalers.keys())
        }


class FormatStandardizer:
    """Standardizes data format for 5G IDS models"""
    
    def __init__(self, target_features=43, logger=None):
        self.target_features = target_features
        self.logger = logger or logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.pca = None
    
    def standardize_format(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """Standardize data format for 5G IDS consumption"""
        self.logger.info(f"🎯 Standardizing format: {df.shape[1]} → {self.target_features} features")
        
        standardization_report = {
            'original_shape': df.shape,
            'target_features': self.target_features,
            'operations_performed': []
        }
        
        # Separate features and target
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Encode target labels
            if y.dtype == 'object' or y.dtype.name == 'category':
                le = LabelEncoder()
                y = le.fit_transform(y)
                standardization_report['target_encoding'] = {
                    'encoder': 'LabelEncoder',
                    'classes': le.classes_.tolist()
                }
            
            y = np.array(y)
        else:
            X = df
            y = None
            standardization_report['target_encoding'] = None
        
        # Convert to numpy array
        X_numeric = X.select_dtypes(include=[np.number])
        
        if X_numeric.shape[1] == 0:
            raise ValueError("No numerical features found after processing")
        
        X_array = X_numeric.values
        
        # Handle feature count mismatch
        current_features = X_array.shape[1]
        self.logger.info(f"🔍 Current features before standardization: {current_features}, Target: {self.target_features}")
        
        if current_features == self.target_features:
            self.logger.info("✅ Feature count matches target")
            X_processed = X_array
            
        elif current_features > self.target_features:
            self.logger.info(f"📉 Reducing features: {current_features} → {self.target_features}")
            X_processed = self._reduce_features(X_array, X_numeric.columns)
            standardization_report['operations_performed'].append('feature_reduction')
            
        else:
            self.logger.info(f"📈 Expanding features: {current_features} → {self.target_features}")
            X_processed = self._expand_features(X_array)
            # Verify expansion worked
            if X_processed.shape[1] != self.target_features:
                self.logger.warning(f"⚠️ Expansion failed: got {X_processed.shape[1]}, expected {self.target_features}")
                # Force to exact count
                current_after_expansion = X_processed.shape[1]
                if current_after_expansion < self.target_features:
                    remaining = self.target_features - current_after_expansion
                    random_features = np.random.normal(0, 0.1, (X_processed.shape[0], remaining))
                    X_processed = np.column_stack([X_processed, random_features])
                    self.logger.info(f"✅ Force-added {remaining} features to reach {self.target_features}")
                elif current_after_expansion > self.target_features:
                    X_processed = X_processed[:, :self.target_features]
                    self.logger.info(f"✅ Trimmed to {self.target_features} features")
            standardization_report['operations_performed'].append('feature_expansion')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_processed)
        standardization_report['operations_performed'].append('feature_scaling')
        
        # Final validation
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            self.logger.warning("⚠️ Invalid values detected - applying final cleaning")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
            standardization_report['operations_performed'].append('invalid_value_cleaning')
        
        standardization_report['final_shape'] = X_scaled.shape
        standardization_report['scaling_stats'] = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist()
        }
        
        self.logger.info(f"✅ Standardization completed: {X_scaled.shape}")
        
        return X_scaled, y, standardization_report
    
    def _reduce_features(self, X, feature_names):
        """Reduce features to target count using intelligent selection"""
        # Method 1: Variance-based selection
        selector = VarianceThreshold(threshold=0.01)
        X_var_selected = selector.fit_transform(X)
        
        if X_var_selected.shape[1] <= self.target_features:
            return X_var_selected
        
        # Method 2: Statistical feature selection
        if X_var_selected.shape[1] > self.target_features:
            # Create dummy target for feature selection
            dummy_target = np.random.randint(0, 2, X_var_selected.shape[0])
            
            selector = SelectKBest(f_classif, k=self.target_features)
            X_selected = selector.fit_transform(X_var_selected, dummy_target)
            self.feature_selector = selector
            
            return X_selected
        
        # Method 3: PCA as fallback
        if X_var_selected.shape[1] > self.target_features:
            self.pca = PCA(n_components=self.target_features)
            X_pca = self.pca.fit_transform(X_var_selected)
            
            self.logger.info(f"📊 PCA explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
            return X_pca
        
        return X_var_selected
    
    def _expand_features(self, X):
        """Expand features to target count"""
        current_features = X.shape[1]
        needed_features = self.target_features - current_features
        
        # Method 1: Polynomial features
        if needed_features <= current_features * 2:
            # Create interaction features
            interactions = []
            for i in range(current_features):
                for j in range(i + 1, current_features):
                    if len(interactions) >= needed_features:
                        break
                    interactions.append(X[:, i] * X[:, j])
                if len(interactions) >= needed_features:
                    break
            
            # Add squared features if needed
            squared_features = []
            for i in range(current_features):
                if len(interactions) + len(squared_features) >= needed_features:
                    break
                squared_features.append(X[:, i] ** 2)
            
            additional_features = interactions + squared_features
            additional_features = additional_features[:needed_features]
            
            if additional_features:
                X_expanded = np.column_stack([X] + additional_features)
                return X_expanded
        
        # Method 2: Statistical features
        if needed_features > 0:
            # Add statistical transformations
            additional_features = []
            
            # Log transform (handle negative values)
            for i in range(current_features):
                if len(additional_features) >= needed_features:
                    break
                col_data = X[:, i]
                if col_data.min() > 0:
                    additional_features.append(np.log1p(col_data))
                else:
                    # Shift to positive and log transform
                    shifted = col_data - col_data.min() + 1
                    additional_features.append(np.log1p(shifted))
            
            # Sqrt transform
            for i in range(current_features):
                if len(additional_features) >= needed_features:
                    break
                col_data = X[:, i]
                if col_data.min() >= 0:
                    additional_features.append(np.sqrt(col_data))
                else:
                    # Use absolute value
                    additional_features.append(np.sqrt(np.abs(col_data)))
            
            # Use only what we need
            additional_features = additional_features[:needed_features]
            
            if additional_features:
                X_expanded = np.column_stack([X] + additional_features)
                return X_expanded
        
        # Method 3: Random features as last resort
        current_count = X.shape[1]
        if current_count < self.target_features:
            remaining = self.target_features - current_count
            random_features = np.random.normal(0, 0.1, (X.shape[0], remaining))
            X_expanded = np.column_stack([X, random_features])
            
            self.logger.info(f"✅ Added {remaining} random features to reach {self.target_features} total features")
            return X_expanded
        
        return X
    
    def get_config(self):
        """Get standardizer configuration"""
        return {
            'type': 'FormatStandardizer',
            'target_features': self.target_features,
            'has_scaler': self.scaler is not None,
            'has_feature_selector': self.feature_selector is not None,
            'has_pca': self.pca is not None
        }
