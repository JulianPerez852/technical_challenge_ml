"""
Dataset Processor for MLOps Pipeline
Converts analysis notebook functionality into a reusable class for dataset generation
"""

import pandas as pd
import numpy as np
import joblib
import logging
import warnings
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

import sys
import os

try:
    from exceptions.exceptions import (
        DataLoadingException, 
        DataSavingException, 
        DataAnalysisException
    )
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from exceptions.exceptions import (
        DataLoadingException, 
        DataSavingException, 
        DataAnalysisException
    )

warnings.filterwarnings('ignore')

class DatasetProcessor:
    """
    A comprehensive dataset processor for MLOps pipeline that handles:
    - Data loading and preprocessing
    - Feature selection using multiple methods
    - PCA dimensionality reduction
    - Dataset generation with specified configurations
    """
    
    def __init__(self, data_dir: str = "data", target_variance: float = 0.95):
        """
        Initialize the DatasetProcessor
        
        Args:
            data_dir: Directory containing data files
            target_variance: Target variance for PCA and feature selection
        """
        self.data_dir = Path(data_dir)
        self.target_variance = target_variance
        self.logger = self._setup_logging()
        
        # Model components
        self.scaler = None
        self.pca_model = None
        self.feature_selector = None
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        
        # Analysis results
        self.correlation_analysis = {}
        self.feature_importance = None
        self.selected_features = None
        self.zero_var_features = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_processed_data(self, train_file: str = "training_data_processed.csv", 
                        test_file: str = "test_data_processed.csv") -> None:
        """
        Load preprocessed training and test data
        
        Args:
            train_file: Name of training data file
            test_file: Name of test data file
        """
        try:
            self.logger.info("Loading processed data...")
            
            train_path = self.data_dir / train_file
            test_path = self.data_dir / test_file
            
            self.train_data = pd.read_csv(train_path)
            self.logger.info(f"Training data loaded: {self.train_data.shape}")
            
            if test_path.exists():
                self.test_data = pd.read_csv(test_path)
                self.logger.info(f"Test data loaded: {self.test_data.shape}")
            else:
                self.logger.warning(f"Test file {test_file} not found")
            
            # Separate features and target
            self.X_train = self.train_data.drop('condition', axis=1)
            self.y_train = self.train_data['condition']
            
            if self.test_data is not None:
                self.X_test = self.test_data.copy()
            
            self.logger.info(f"Features shape: {self.X_train.shape}")
            self.logger.info(f"Target distribution:\n{self.y_train.value_counts()}")
            
        except Exception as e:
            raise DataLoadingException(f"Error loading processed data: {str(e)}")
    
    def analyze_correlations(self, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Analyze feature correlations and identify highly correlated pairs
        
        Args:
            threshold: Correlation threshold for identifying pairs
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            self.logger.info("Analyzing feature correlations...")
            
            correlation_matrix = self.X_train.corr()
            high_corr_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > threshold:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_value
                        })
            
            self.correlation_analysis = {
                'correlation_matrix': correlation_matrix,
                'high_corr_pairs': high_corr_pairs,
                'threshold': threshold
            }
            
            self.logger.info(f"Found {len(high_corr_pairs)} highly correlated pairs (>{threshold})")
            return self.correlation_analysis
            
        except Exception as e:
            raise DataAnalysisException(f"Error in correlation analysis: {str(e)}")
    
    def calculate_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance using Random Forest and ANOVA F-test
        
        Returns:
            DataFrame with combined feature importance scores
        """
        try:
            self.logger.info("Calculating feature importance...")
            
            # Random Forest importance
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            
            rf_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'rf_importance': rf.feature_importances_
            })
            
            # ANOVA F-test importance
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(self.X_train, self.y_train)
            
            f_scores = pd.DataFrame({
                'feature': self.X_train.columns,
                'f_score': selector.scores_
            })
            
            # Combine and normalize scores
            rf_scores_norm = ((rf_importance['rf_importance'] - rf_importance['rf_importance'].min()) / 
                             (rf_importance['rf_importance'].max() - rf_importance['rf_importance'].min()))
            f_scores_norm = ((f_scores['f_score'] - f_scores['f_score'].min()) / 
                            (f_scores['f_score'].max() - f_scores['f_score'].min()))
            
            combined_scores = pd.DataFrame({
                'feature': rf_importance['feature'],
                'rf_importance': rf_importance['rf_importance'],
                'f_score': f_scores.set_index('feature').loc[rf_importance['feature'], 'f_score'].values,
                'rf_normalized': rf_scores_norm,
                'f_normalized': f_scores_norm,
                'combined_score': (rf_scores_norm + f_scores_norm) / 2
            }).sort_values('combined_score', ascending=False)
            
            self.feature_importance = combined_scores
            self.logger.info("Feature importance calculation completed")
            return combined_scores
            
        except Exception as e:
            raise DataAnalysisException(f"Error calculating feature importance: {str(e)}")
    
    def select_features_by_importance(self, target_variability: float = None) -> List[str]:
        """
        Select features based on cumulative importance score
        
        Args:
            target_variability: Target cumulative variability (uses instance default if None)
            
        Returns:
            List of selected feature names
        """
        try:
            if target_variability is None:
                target_variability = self.target_variance
                
            if self.feature_importance is None:
                self.calculate_feature_importance()
            
            self.logger.info(f"Selecting features for {target_variability*100}% variability...")
            
            # Calculate cumulative importance
            self.feature_importance['cumulative_importance'] = (
                self.feature_importance['combined_score'].cumsum() / 
                self.feature_importance['combined_score'].sum()
            )
            
            # Select features
            features_mask = self.feature_importance['cumulative_importance'] <= target_variability
            n_features = features_mask.sum()
            
            if self.feature_importance.loc[features_mask, 'cumulative_importance'].max() < target_variability:
                n_features += 1
            
            self.selected_features = self.feature_importance.head(n_features)['feature'].tolist()
            
            actual_variability = self.feature_importance.head(n_features)['cumulative_importance'].max()
            reduction_pct = ((len(self.X_train.columns) - n_features) / len(self.X_train.columns) * 100)
            
            self.logger.info(f"Selected {n_features} features")
            self.logger.info(f"Actual variability: {actual_variability:.4f}")
            self.logger.info(f"Feature reduction: {reduction_pct:.1f}%")
            
            return self.selected_features
            
        except Exception as e:
            raise DataAnalysisException(f"Error in feature selection: {str(e)}")
    
    def apply_pca(self, n_components: int = None, target_variance: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply PCA dimensionality reduction
        
        Args:
            n_components: Number of components (calculated if None)
            target_variance: Target variance for component selection
            
        Returns:
            Tuple of (X_train_pca, X_test_pca)
        """
        try:
            if target_variance is None:
                target_variance = self.target_variance
                
            self.logger.info("Applying PCA transformation...")
            
            # Remove zero variance features
            feature_variance = self.X_train.var()
            self.zero_var_features = feature_variance[feature_variance == 0].index.tolist()
            
            if self.zero_var_features:
                self.logger.info(f"Removing {len(self.zero_var_features)} zero variance features")
                X_train_clean = self.X_train.drop(columns=self.zero_var_features)
                if self.X_test is not None:
                    X_test_clean = self.X_test.drop(columns=self.zero_var_features, errors='ignore')
            else:
                X_train_clean = self.X_train.copy()
                X_test_clean = self.X_test.copy() if self.X_test is not None else None
            
            # Standardize data
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_clean)
            
            # Determine number of components
            if n_components is None:
                pca_full = PCA()
                pca_full.fit(X_train_scaled)
                cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                n_components = np.argmax(cumulative_variance >= target_variance) + 1
            
            # Apply PCA
            self.pca_model = PCA(n_components=n_components, random_state=42)
            X_train_pca = self.pca_model.fit_transform(X_train_scaled)
            
            X_test_pca = None
            if X_test_clean is not None:
                X_test_scaled = self.scaler.transform(X_test_clean)
                X_test_pca = self.pca_model.transform(X_test_scaled)
            
            variance_explained = self.pca_model.explained_variance_ratio_.sum()
            
            self.logger.info(f"PCA applied: {X_train_scaled.shape[1]} -> {n_components} components")
            self.logger.info(f"Variance explained: {variance_explained:.4f}")
            
            return X_train_pca, X_test_pca
            
        except Exception as e:
            raise DataAnalysisException(f"Error in PCA application: {str(e)}")
    
    def generate_feature_selected_datasets(self, output_dir: str = None, 
                                        selected_features: List[str] = None) -> None:
        """
        Generate datasets with selected features
        
        Args:
            output_dir: Output directory (uses instance data_dir if None)
            selected_features: List of features to include (uses instance selection if None)
        """
        try:
            if output_dir is None:
                output_dir = self.data_dir
            else:
                output_dir = Path(output_dir)
                
            if selected_features is None:
                if self.selected_features is None:
                    self.select_features_by_importance()
                selected_features = self.selected_features
            
            self.logger.info("Generating feature-selected datasets...")
            
            # Create training dataset
            train_selected = self.X_train[selected_features].copy()
            train_selected['condition'] = self.y_train.values
            
            train_output_path = output_dir / "training_data_features_selected.csv"
            train_selected.to_csv(train_output_path, index=False)
            self.logger.info(f"Training dataset saved: {train_output_path}")
            self.logger.info(f"Shape: {train_selected.shape}")
            
            # Create test dataset if available
            if self.X_test is not None:
                test_selected = self.X_test[selected_features].copy()
                test_output_path = output_dir / "test_data_features_selected.csv"
                test_selected.to_csv(test_output_path, index=False)
                self.logger.info(f"Test dataset saved: {test_output_path}")
                self.logger.info(f"Shape: {test_selected.shape}")
            
        except Exception as e:
            raise DataSavingException(f"Error generating feature-selected datasets: {str(e)}")
    
    def generate_pca_datasets(self, output_dir: str = None, target_variance: float = None) -> None:
        """
        Generate datasets with PCA transformation
        
        Args:
            output_dir: Output directory (uses instance data_dir if None)
            target_variance: Target variance for PCA (uses instance default if None)
        """
        try:
            if output_dir is None:
                output_dir = self.data_dir
            else:
                output_dir = Path(output_dir)
                
            if target_variance is None:
                target_variance = self.target_variance
            
            self.logger.info("Generating PCA-transformed datasets...")
            
            # Apply PCA
            X_train_pca, X_test_pca = self.apply_pca(target_variance=target_variance)
            
            # Create column names
            pca_columns = [f'PC{i+1}' for i in range(X_train_pca.shape[1])]
            
            # Create training dataset
            train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns)
            train_pca_df['condition'] = self.y_train.values
            
            train_output_path = output_dir / "training_data_pca.csv"
            train_pca_df.to_csv(train_output_path, index=False)
            self.logger.info(f"PCA training dataset saved: {train_output_path}")
            self.logger.info(f"Shape: {train_pca_df.shape}")
            
            # Create test dataset if available
            if X_test_pca is not None:
                test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns)
                test_output_path = output_dir / "test_data_pca.csv"
                test_pca_df.to_csv(test_output_path, index=False)
                self.logger.info(f"PCA test dataset saved: {test_output_path}")
                self.logger.info(f"Shape: {test_pca_df.shape}")
            
        except Exception as e:
            raise DataSavingException(f"Error generating PCA datasets: {str(e)}")
    
    def save_models(self, output_dir: str = None) -> None:
        """
        Save trained models and preprocessing objects
        
        Args:
            output_dir: Output directory (uses instance data_dir if None)
        """
        try:
            if output_dir is None:
                output_dir = self.data_dir
            else:
                output_dir = Path(output_dir)
            
            self.logger.info("Saving models and preprocessing objects...")
            
            # Save scaler
            if self.scaler is not None:
                scaler_path = output_dir / "pca_scaler.joblib"
                joblib.dump(self.scaler, scaler_path)
                self.logger.info(f"Scaler saved: {scaler_path}")
            
            # Save PCA model
            if self.pca_model is not None:
                pca_path = output_dir / "pca_model.joblib"
                joblib.dump(self.pca_model, pca_path)
                self.logger.info(f"PCA model saved: {pca_path}")
                
                # Save PCA info
                model_info = {
                    'zero_variance_features': self.zero_var_features,
                    'n_components': self.pca_model.n_components_,
                    'variance_explained': float(self.pca_model.explained_variance_ratio_.sum()),
                    'original_features': self.X_train.columns.tolist(),
                    'pca_columns': [f'PC{i+1}' for i in range(self.pca_model.n_components_)]
                }
                
                info_path = output_dir / "pca_info.joblib"
                joblib.dump(model_info, info_path)
                self.logger.info(f"PCA info saved: {info_path}")
            
            # Save feature importance and selection
            if self.feature_importance is not None:
                importance_path = output_dir / "feature_importance.csv"
                self.feature_importance.to_csv(importance_path, index=False)
                self.logger.info(f"Feature importance saved: {importance_path}")
            
            if self.selected_features is not None:
                features_path = output_dir / "selected_features.joblib"
                joblib.dump(self.selected_features, features_path)
                self.logger.info(f"Selected features saved: {features_path}")
                
        except Exception as e:
            raise DataSavingException(f"Error saving models: {str(e)}")
    
    def run_full_analysis(self, generate_pca: bool = True, generate_features: bool = True,
                        output_dir: str = None) -> Dict[str, Any]:
        """
        Run complete analysis pipeline
        
        Args:
            generate_pca: Whether to generate PCA datasets
            generate_features: Whether to generate feature-selected datasets
            output_dir: Output directory for results
            
        Returns:
            Dictionary with analysis summary
        """
        try:
            self.logger.info("Starting full analysis pipeline...")
            
            # Load data if not already loaded
            if self.train_data is None:
                self.load_processed_data()
            
            # Analyze correlations
            correlation_results = self.analyze_correlations()
            
            # Calculate feature importance
            feature_importance = self.calculate_feature_importance()
            
            # Select features
            selected_features = self.select_features_by_importance()
            
            # Generate datasets
            if generate_features:
                self.generate_feature_selected_datasets(output_dir)
            
            if generate_pca:
                self.generate_pca_datasets(output_dir)
            
            # Save models
            self.save_models(output_dir)
            
            # Prepare summary
            summary = {
                'original_features': len(self.X_train.columns),
                'selected_features': len(selected_features),
                'feature_reduction_pct': ((len(self.X_train.columns) - len(selected_features)) / 
                                        len(self.X_train.columns) * 100),
                'high_correlation_pairs': len(correlation_results['high_corr_pairs']),
                'target_variance': self.target_variance,
                'train_shape': self.train_data.shape,
                'test_shape': self.test_data.shape if self.test_data is not None else None
            }
            
            if self.pca_model is not None:
                summary['pca_components'] = self.pca_model.n_components_
                summary['pca_variance_explained'] = float(self.pca_model.explained_variance_ratio_.sum())
            
            self.logger.info("Full analysis pipeline completed successfully")
            self.logger.info(f"Summary: {summary}")
            
            return summary
            
        except Exception as e:
            raise DataAnalysisException(f"Error in full analysis pipeline: {str(e)}")


def main():
    """
    Main execution function for standalone use
    """
    try:
        # Initialize processor
        processor = DatasetProcessor(data_dir="../../data", target_variance=0.95)
        
        # Run full analysis
        summary = processor.run_full_analysis(
            generate_pca=True,
            generate_features=True
        )
        
        print("\n" + "="*60)
        print("DATASET PROCESSING COMPLETE")
        print("="*60)
        print(f"Original features: {summary['original_features']}")
        print(f"Selected features: {summary['selected_features']}")
        print(f"Feature reduction: {summary['feature_reduction_pct']:.1f}%")
        print(f"High correlation pairs: {summary['high_correlation_pairs']}")
        
        if 'pca_components' in summary:
            print(f"PCA components: {summary['pca_components']}")
            print(f"PCA variance explained: {summary['pca_variance_explained']:.4f}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()