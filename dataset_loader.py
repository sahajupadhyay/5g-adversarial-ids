import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PFCPDatasetLoader:
    def __init__(self, data_path):
        """
        5G PFCP Dataset Loader for Adversarial IDS Project
        Expected dataset: EU Horizon 5G-CARMEN PFCP Dataset (SANCUS Project)
        
        Dataset Structure:
        - 5 classes: Normal, Mal_Estab, Mal_Del, Mal_Mod, Mal_Mod2
        - Multiple timeout values: 15s, 20s, 60s, 120s, 240s
        - Pre-split: 70% training, 30% testing
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Expected classes from SANCUS dataset
        self.expected_classes = ["Normal", "Mal_Estab", "Mal_Del", "Mal_Mod", "Mal_Mod2"]
        self.timeout_options = ["15s", "20s", "60s", "120s", "240s"]
        
    def load_sancus_dataset(self, timeout="60s", layer="PFCP"):
        """
        Load SANCUS 5G PFCP Dataset with specific timeout and layer
        
        Args:
            timeout: Flow timeout value ("15s", "20s", "60s", "120s", "240s")
            layer: "PFCP" for PFCP APP layer or "TCP" for TCP-IP layer
        """
        if timeout not in self.timeout_options:
            raise ValueError(f"Invalid timeout. Choose from: {self.timeout_options}")
            
        # Determine folder structure (match actual folder names)
        if layer == "PFCP":
            base_folder = "Balanced PFCP APP Layer"
        else:
            base_folder = "Balanced TCP-IP Layer"
            
        timeout_folder = f"{timeout.replace('s', '')}-sec-CSV"
        
        # Load training data (files include 's' suffix)
        train_path = os.path.join(self.data_path, base_folder, timeout_folder, "Training", f"Training_{timeout}.csv")
        test_path = os.path.join(self.data_path, base_folder, timeout_folder, "Testing", f"Testing_{timeout}.csv")
        
        print(f"Loading SANCUS Dataset:")
        print(f"- Layer: {layer}")
        print(f"- Timeout: {timeout}")
        print(f"- Training file: {train_path}")
        print(f"- Testing file: {test_path}")
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Combine for initial exploration
            self.raw_data = pd.concat([train_df, test_df], ignore_index=True)
            self.train_df = train_df
            self.test_df = test_df
            
            print(f"✅ Dataset loaded successfully!")
            print(f"Training set: {train_df.shape}")
            print(f"Testing set: {test_df.shape}")
            print(f"Combined dataset: {self.raw_data.shape}")
            
            return self.raw_data
            
        except FileNotFoundError as e:
            print(f"❌ Dataset files not found: {e}")
            print(f"Expected structure:")
            print(f"  {self.data_path}/")
            print(f"  ├── Balanced PFCP APP Layer/")
            print(f"  │   └── {timeout_folder}/")
            print(f"  │       ├── Training/Training_{timeout}.csv")
            print(f"  │       └── Testing/Testing_{timeout}.csv")
            raise
    
    def explore_sancus_dataset(self):
        """Comprehensive EDA for SANCUS 5G PFCP Dataset"""
        if self.raw_data is None:
            raise ValueError("Dataset not loaded. Call load_sancus_dataset() first.")
            
        print("=" * 60)
        print("SANCUS 5G PFCP DATASET EXPLORATION")
        print("=" * 60)
        
        # Basic info
        print(f"Dataset Shape: {self.raw_data.shape}")
        print(f"Memory Usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()
        
        # Verify Label column (should be "Label" based on documentation)
        if 'Label' in self.raw_data.columns:
            print("✅ Label column found")
            print("Class Distribution:")
            class_counts = self.raw_data['Label'].value_counts()
            print(class_counts)
            
            # Verify expected classes
            found_classes = set(self.raw_data['Label'].unique())
            expected_classes = set(self.expected_classes)
            
            if found_classes == expected_classes:
                print("✅ All expected classes present:", list(found_classes))
            else:
                print("⚠️  Class mismatch!")
                print(f"Expected: {expected_classes}")  
                print(f"Found: {found_classes}")
                
            print(f"Dataset Balance Ratio: {class_counts.min()/class_counts.max():.3f}")
        else:
            print("❌ Label column not found!")
            print("Available columns:", list(self.raw_data.columns))
            return None
            
        print()
        
        # PFCP-specific feature analysis
        pfcp_features = [col for col in self.raw_data.columns if 'PFCP' in col]
        print(f"PFCP-specific features found: {len(pfcp_features)}")
        if len(pfcp_features) > 0:
            print("Sample PFCP features:", pfcp_features[:5])
            
        # Check for missing values
        missing = self.raw_data.isnull().sum()
        if missing.sum() > 0:
            print("Missing Values:")
            print(missing[missing > 0].head())
        else:
            print("✅ No missing values detected")
            
        print()
        
        # Data types
        print("Data Types Summary:")
        dtype_counts = self.raw_data.dtypes.value_counts()
        print(dtype_counts)
        
        # Feature count verification
        expected_features = {
            "PFCP": 35,  # Based on Table 3 in documentation
            "TCP": 80    # Based on Table 2 in documentation  
        }
        
        feature_count = len(self.raw_data.columns) - 1  # Excluding Label
        print(f"Feature count (excluding Label): {feature_count}")
        
        return 'Label'
    
    def visualize_data_distribution(self, target_col=None):
        """Create visualization plots for dataset understanding"""
        if target_col is None:
            print("⚠️  Target column not specified, skipping class distribution plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Class distribution
        self.raw_data[target_col].value_counts().plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('5G PFCP Attack Class Distribution')
        axes[0,0].set_xlabel('Attack Type')
        axes[0,0].set_ylabel('Count')
        
        # Feature correlation heatmap (sample of numeric features)
        numeric_cols = self.raw_data.select_dtypes(include=[np.number]).columns[:10]
        if len(numeric_cols) > 1:
            corr_matrix = self.raw_data[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=axes[0,1])
            axes[0,1].set_title('Feature Correlation Matrix (Sample)')
        
        # Dataset size by class
        class_sizes = self.raw_data[target_col].value_counts()
        axes[1,0].pie(class_sizes.values, labels=class_sizes.index, autopct='%1.1f%%')
        axes[1,0].set_title('Class Distribution (Percentage)')
        
        # Feature statistics
        numeric_features = self.raw_data.select_dtypes(include=[np.number])
        if len(numeric_features.columns) > 0:
            feature_stats = numeric_features.describe().T
            # Limit to top 10 features and handle shape mismatch
            top_features = min(10, len(feature_stats))
            axes[1,1].bar(range(top_features), feature_stats['mean'][:top_features])
            axes[1,1].set_title(f'Top {top_features} Features - Mean Values')
            axes[1,1].set_xlabel('Feature Index')
        
        plt.tight_layout()
        plt.show()
        
    def preprocess_sancus_data(self, use_presplit=True):
        """
        Preprocess SANCUS dataset - can use pre-split or create new split
        
        Args:
            use_presplit: If True, use the provided 70/30 train/test split
        """
        if self.raw_data is None:
            raise ValueError("Dataset not loaded")
            
        print("🔄 Preprocessing SANCUS PFCP dataset...")
        
        target_col = 'Label'
        
        if use_presplit and hasattr(self, 'train_df') and hasattr(self, 'test_df'):
            print("Using pre-provided train/test split (70/30)")
            
            # Separate features and target for training data
            X_train = self.train_df.drop(columns=[target_col])
            y_train = self.train_df[target_col]
            
            # Separate features and target for test data  
            X_test = self.test_df.drop(columns=[target_col])
            y_test = self.test_df[target_col]
            
        else:
            print("Creating new stratified split")
            X = self.raw_data.drop(columns=[target_col])
            y = self.raw_data[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        
        # Handle categorical features (should be minimal in PFCP data)
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding {len(categorical_cols)} categorical features")
            X_train = pd.get_dummies(X_train, columns=categorical_cols)
            X_test = pd.get_dummies(X_test, columns=categorical_cols)
            
            # Ensure both sets have same columns
            train_cols = set(X_train.columns)
            test_cols = set(X_test.columns)
            
            for col in train_cols - test_cols:
                X_test[col] = 0
            for col in test_cols - train_cols:
                X_train[col] = 0
                
            X_train = X_train[sorted(X_train.columns)]
            X_test = X_test[sorted(X_test.columns)]
        
        # Handle infinite and missing values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median (robust for PFCP features)
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use train median for test
        
        # Encode target labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        
        # Store original DataFrames for reference
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train_encoded  
        self.y_test = y_test_encoded
        
        print(f"✅ Preprocessing complete!")
        print(f"Training set: {self.X_train_scaled.shape}")
        print(f"Test set: {self.X_test_scaled.shape}")
        print(f"Classes: {list(self.label_encoder.classes_)}")
        print(f"Features: {X_train.shape[1]}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def save_processed_data(self, output_dir="data/processed/"):
        """Save processed data for team members"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as numpy arrays for efficiency
        np.save(f"{output_dir}/X_train.npy", self.X_train_scaled)
        np.save(f"{output_dir}/X_test.npy", self.X_test_scaled)
        np.save(f"{output_dir}/y_train.npy", self.y_train)
        np.save(f"{output_dir}/y_test.npy", self.y_test)
        
        # Save metadata
        metadata = {
            'classes': list(self.label_encoder.classes_),
            'n_features': self.X_train.shape[1],
            'train_size': len(self.y_train),
            'test_size': len(self.y_test)
        }
        
        import json
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Processed data saved to {output_dir}")

# Updated Usage Example for SANCUS Dataset
if __name__ == "__main__":
    import os
    
    # Initialize loader for SANCUS dataset
    # Adjust path to your dataset location
    loader = PFCPDatasetLoader("data/raw/")  
    
    # Load SANCUS dataset - let's use 60s timeout for balance of samples vs features
    try:
        print("🚀 Loading SANCUS 5G PFCP Dataset...")
        dataset = loader.load_sancus_dataset(timeout="60s", layer="PFCP")
        print("✅ Dataset loaded successfully")
        
        # Explore the dataset
        target_col = loader.explore_sancus_dataset()
        
        if target_col:
            # Create visualizations
            loader.visualize_data_distribution(target_col)
            
            # Preprocess using pre-provided split
            X_train, X_test, y_train, y_test = loader.preprocess_sancus_data(use_presplit=True)
            
            # Save processed data for team
            loader.save_processed_data()
            
            print("\n🎯 DATASET VALIDATION:")
            print(f"✅ Expected 5 classes: {len(loader.label_encoder.classes_) == 5}")
            print(f"✅ Balanced classes: {abs(max(np.bincount(y_train)) - min(np.bincount(y_train))) <= 5}")
            print(f"✅ PFCP features: {X_train.shape[1] >= 30}")
            print(f"✅ Sufficient samples: {len(y_train) >= 1000}")
            
            print("\n🚀 READY FOR BASELINE MODEL TRAINING!")
            print("Expected performance targets:")
            print("- Random Forest: ~95% accuracy")  
            print("- SVM: ~93% accuracy")
            print("- Neural Network: ~94% accuracy")
            
        else:
            print("❌ Cannot proceed without proper Label column")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔍 TROUBLESHOOTING STEPS:")
        print("1. Ensure you have extracted the SANCUS dataset")
        print("2. Verify folder structure matches documentation")
        print("3. Check that Balanced_PFCP_APP_Layer folder exists")
        print("4. Confirm CSV files are in correct subfolders")