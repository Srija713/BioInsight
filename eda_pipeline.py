"""
BioInsight Lite - EDA & Preprocessing Pipeline
Handles data cleaning, feature engineering, and exploratory analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class BioactivityPreprocessor:
    def __init__(self, df):
        """
        Initialize preprocessor with bioactivity data
        
        Args:
            df: pandas DataFrame with bioactivity data
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
    
    def explore_data(self):
        """Generate comprehensive EDA report"""
        print("=" * 80)
        print("BIOACTIVITY DATA EXPLORATION REPORT")
        print("=" * 80)
        
        # Basic info
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing': missing[missing > 0],
            'Percentage': missing_pct[missing > 0]
        }).sort_values('Percentage', ascending=False)
        print(missing_df)
        
        # Data types
        print("\n--- Data Types ---")
        print(self.df.dtypes.value_counts())
        
        # Numerical features summary
        print("\n--- Numerical Features Summary ---")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe())
        
        # Categorical features
        print("\n--- Categorical Features ---")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:5]:  # Show first 5
            print(f"\n{col}: {self.df[col].nunique()} unique values")
            print(self.df[col].value_counts().head())
        
        return {
            'shape': self.df.shape,
            'missing': missing_df,
            'numerical_cols': numerical_cols.tolist(),
            'categorical_cols': categorical_cols.tolist()
        }
    
    def create_target_variable(self, threshold_type='median'):
        """
        Create binary target variable for bioactivity prediction
        
        Args:
            threshold_type: 'median', 'mean', or custom value
        
        Returns:
            DataFrame with 'is_active' binary target
        """
        # Use pchembl_value (negative log of activity) as primary measure
        # Higher pchembl_value = more potent compound
        
        if 'pchembl_value' in self.df.columns:
            activity_col = 'pchembl_value'
            valid_data = self.df[self.df[activity_col].notna()].copy()
        else:
            # Fallback to standard_value (convert to pChEMBL scale)
            activity_col = 'standard_value'
            valid_data = self.df[self.df[activity_col].notna()].copy()
            # Convert to pChEMBL: -log10(IC50 in M)
            # Assuming standard_value is in nM
            valid_data['pchembl_value'] = -np.log10(valid_data[activity_col] * 1e-9)
            activity_col = 'pchembl_value'
        
        # Determine threshold
        if threshold_type == 'median':
            threshold = valid_data[activity_col].median()
        elif threshold_type == 'mean':
            threshold = valid_data[activity_col].mean()
        else:
            threshold = threshold_type
        
        # Create binary target: 1 = active (pchembl >= threshold), 0 = inactive
        valid_data['is_active'] = (valid_data[activity_col] >= threshold).astype(int)
        
        print(f"\n--- Target Variable Creation ---")
        print(f"Threshold ({threshold_type}): {threshold:.2f}")
        print(f"Active compounds: {valid_data['is_active'].sum()} ({valid_data['is_active'].mean()*100:.1f}%)")
        print(f"Inactive compounds: {(1-valid_data['is_active']).sum()} ({(1-valid_data['is_active']).mean()*100:.1f}%)")
        
        self.df = valid_data
        return self.df
    
    def clean_and_engineer_features(self):
        """Clean data and engineer features"""
        print("\n--- Feature Engineering ---")
        
        df = self.df.copy()
        
        # 1. Handle missing values in key numerical features
        numerical_features = [
            'mw_freebase', 'alogp', 'hba', 'hbd', 'psa', 'rtb',
            'aromatic_rings', 'heavy_atoms', 'num_ro5_violations'
        ]
        
        for col in numerical_features:
            if col in df.columns:
                # Fill with median
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                self.feature_stats[f'{col}_median'] = median_val
        
        # 2. Create derived features
        if 'mw_freebase' in df.columns and 'heavy_atoms' in df.columns:
            df['mw_per_heavy_atom'] = df['mw_freebase'] / (df['heavy_atoms'] + 1)
        
        if 'hba' in df.columns and 'hbd' in df.columns:
            df['hb_total'] = df['hba'] + df['hbd']
            df['hba_hbd_ratio'] = df['hba'] / (df['hbd'] + 1)
        
        if 'aromatic_rings' in df.columns:
            df['has_aromatic'] = (df['aromatic_rings'] > 0).astype(int)
        
        # 3. Encode categorical features
        categorical_features = ['assay_type', 'molecule_type', 'target_type']
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                valid_mask = df[col].notna()
                df.loc[valid_mask, f'{col}_encoded'] = le.fit_transform(df.loc[valid_mask, col])
                df[f'{col}_encoded'].fillna(-1, inplace=True)
                self.label_encoders[col] = le
        
        # 4. Create Lipinski's Rule of Five compliance score
        if all(col in df.columns for col in ['mw_freebase', 'alogp', 'hba', 'hbd']):
            df['lipinski_violations'] = 0
            df.loc[df['mw_freebase'] > 500, 'lipinski_violations'] += 1
            df.loc[df['alogp'] > 5, 'lipinski_violations'] += 1
            df.loc[df['hba'] > 10, 'lipinski_violations'] += 1
            df.loc[df['hbd'] > 5, 'lipinski_violations'] += 1
            df['lipinski_pass'] = (df['lipinski_violations'] == 0).astype(int)
        
        print(f"Engineered features. New shape: {df.shape}")
        self.df = df
        return df
    
    def visualize_distributions(self, save_path=None):
        """Create comprehensive visualization of feature distributions"""
        
        numerical_cols = [
            'mw_freebase', 'alogp', 'hba', 'hbd', 'psa', 'rtb',
            'pchembl_value', 'num_ro5_violations'
        ]
        
        # Filter available columns
        available_cols = [col for col in numerical_cols if col in self.df.columns]
        
        # Create subplots
        n_cols = 3
        n_rows = (len(available_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(available_cols):
            ax = axes[idx]
            
            # Histogram with KDE
            self.df[col].hist(bins=50, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Distribution of {col}')
            
            # Add statistics
            mean_val = self.df[col].mean()
            median_val = self.df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(len(available_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def correlation_analysis(self):
        """Generate correlation heatmap"""
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numerical_cols].corr()
        
        # Create interactive heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(x="Features", y="Features", color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        
        fig.update_layout(width=1000, height=900)
        return fig, corr_matrix
    
    def activity_analysis(self):
        """Analyze bioactivity patterns"""
        if 'is_active' not in self.df.columns:
            print("Target variable 'is_active' not found. Create it first.")
            return None
        
        # Compare feature distributions between active and inactive
        numerical_cols = ['mw_freebase', 'alogp', 'hba', 'hbd', 'psa']
        available_cols = [col for col in numerical_cols if col in self.df.columns]
        
        fig, axes = plt.subplots(1, len(available_cols), figsize=(20, 4))
        
        for idx, col in enumerate(available_cols):
            active_data = self.df[self.df['is_active'] == 1][col]
            inactive_data = self.df[self.df['is_active'] == 0][col]
            
            axes[idx].hist(inactive_data, bins=30, alpha=0.5, label='Inactive', color='blue')
            axes[idx].hist(active_data, bins=30, alpha=0.5, label='Active', color='red')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{col} by Activity')
            axes[idx].legend()
        
        plt.tight_layout()
        return fig
    
    def prepare_ml_features(self):
        """Prepare final feature set for ML models"""
        feature_columns = [
            'mw_freebase', 'alogp', 'hba', 'hbd', 'psa', 'rtb',
            'aromatic_rings', 'heavy_atoms', 'num_ro5_violations',
            'mw_per_heavy_atom', 'hb_total', 'hba_hbd_ratio',
            'lipinski_violations', 'assay_type_encoded', 'molecule_type_encoded'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        X = self.df[available_features].copy()
        y = self.df['is_active'].copy() if 'is_active' in self.df.columns else None
        
        # Handle any remaining NaN values
        X.fillna(X.median(), inplace=True)
        
        print(f"\n--- ML Feature Preparation ---")
        print(f"Features: {len(available_features)}")
        print(f"Samples: {len(X)}")
        if y is not None:
            print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('bioactivity_data.csv')
    
    # Initialize preprocessor
    preprocessor = BioactivityPreprocessor(df)
    
    # Explore data
    stats = preprocessor.explore_data()
    
    # Create target variable
    preprocessor.create_target_variable(threshold_type='median')
    
    # Clean and engineer features
    preprocessor.clean_and_engineer_features()
    
    # Visualizations
    dist_fig = preprocessor.visualize_distributions(save_path='distributions.png')
    corr_fig, corr_matrix = preprocessor.correlation_analysis()
    activity_fig = preprocessor.activity_analysis()
    
    # Prepare ML features
    X, y, feature_names = preprocessor.prepare_ml_features()
    
    print("\nPreprocessing complete!")
    print(f"Final dataset: X shape = {X.shape}, y shape = {y.shape}")