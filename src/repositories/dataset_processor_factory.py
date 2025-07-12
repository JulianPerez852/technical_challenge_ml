import pandas as pd
import ast
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from imblearn.over_sampling import SMOTENC
import logging
import sys
import os
try:
    from exceptions.exceptions import (
        DataProcessingException,
        ProcessorCreationException,
        FeatureTransformationException,
        SMOTEApplicationException
    )
except Exception:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from exceptions.exceptions import (
        DataProcessingException,
        ProcessorCreationException,
        FeatureTransformationException,
        SMOTEApplicationException
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetProcessor(ABC):
    """Abstract base class for dataset processors"""
    
    def __init__(self):
        self.target_column = "condition"
        self.selected_columns = [
            "shipping",
            "non_mercado_pago_payment_methods", 
            "price",
            "variations",
            "listing_type_id",
            "buying_mode",
            "tags",
            "official_store_id",
            "automatic_relist",
            "initial_quantity",
            "condition",
            "warranty_category",
            "sold_quantity",
            "seller_address",
            "title"
        ]
        
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the dataset according to specific requirements"""
        pass
    
    def _select_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select relevant columns for processing"""
        try:
            missing_columns = [col for col in self.selected_columns if col not in data.columns]
            if missing_columns:
                raise DataProcessingException(f"Missing required columns: {missing_columns}")
            return data[self.selected_columns].copy()
        except Exception as e:
            logger.error(f"Error selecting columns: {e}")
            raise FeatureTransformationException(f"Failed to select columns: {e}")
    
    def _process_shipping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract shipping features from shipping column"""
        try:
            if "shipping" not in df.columns:
                raise FeatureTransformationException("shipping column not found")
            
            def safe_extract_shipping(x, key):
                try:
                    if pd.notnull(x):
                        return ast.literal_eval(x).get(key, None)
                    return None
                except (ValueError, SyntaxError, KeyError):
                    return None
            
            df["local_pick_up"] = df["shipping"].apply(lambda x: safe_extract_shipping(x, "local_pick_up"))
            df["free_shipping"] = df["shipping"].apply(lambda x: safe_extract_shipping(x, "free_shipping"))
            df["mode"] = df["shipping"].apply(lambda x: safe_extract_shipping(x, "mode"))
            
            return df.drop(columns=["shipping"])
        except Exception as e:
            logger.error(f"Error processing shipping: {e}")
            raise FeatureTransformationException(f"Failed to process shipping: {e}")
    
    def _process_payment_methods(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract payment method features"""
        descripcion_list = [
            'Visa Electron', 'Mastercard Maestro', 'Acordar con el comprador',
            'Transferencia bancaria', 'Visa', 'Contra reembolso', 'MasterCard',
            'Cheque certificado', 'Tarjeta de crédito', 'Giro postal', 'Diners',
            'Efectivo', 'American Express', 'MercadoPago'
        ]
        
        def get_descriptions(row):
            try:
                items = ast.literal_eval(row) if pd.notnull(row) else []
                return set([item['description'] for item in items])
            except Exception:
                return set()
        
        for desc in descripcion_list:
            df[desc] = df['non_mercado_pago_payment_methods'].apply(
                lambda x: 1 if desc in get_descriptions(x) else 0
            )
        
        return df.drop(columns=["non_mercado_pago_payment_methods"])
    
    def _process_variations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract variation features"""
        def get_items_variation(row):
            try:
                items = ast.literal_eval(row) if pd.notnull(row) else []
                return len(items)
            except Exception:
                return 0
        
        def get_available_quantity_variation(row):
            try:
                items = ast.literal_eval(row) if pd.notnull(row) else []
                if not items:
                    return 0
                total = sum(item.get('available_quantity', 0) for item in items)
                return total / len(items) if len(items) > 0 else 0
            except Exception:
                return 0
        
        df['items_variation'] = df['variations'].apply(get_items_variation)
        df['available_quantity_variation'] = df['variations'].apply(get_available_quantity_variation)
        
        return df.drop(columns=["variations"])
    
    def _process_listing_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for listing type"""
        listing_types = ['bronze', 'silver', 'free', 'gold_special', 'gold', 'gold_premium', 'gold_pro']
        for lt in listing_types:
            df[lt] = (df['listing_type_id'] == lt).astype(int)
        return df.drop(columns=["listing_type_id"])
    
    def _process_buying_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for buying mode"""
        buying_modes = ['buy_it_now', 'classified', 'auction']
        for bm in buying_modes:
            df[bm] = (df['buying_mode'] == bm).astype(int)
        return df.drop(columns=["buying_mode"])
    
    def _process_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process tags column into dummy variables"""
        tag_classes = [
            'dragged_bids_and_visits', 'good_quality_thumbnail', 'dragged_visits',
            'free_relist', 'poor_quality_thumbnail'
        ]
        
        def tag_dummies(tags):
            try:
                tags = ast.literal_eval(tags) if pd.notnull(tags) else []
            except Exception:
                tags = []
            if not isinstance(tags, list):
                return {tag: 0 for tag in tag_classes} | {'sin_tag': 1}
            if len(tags) == 0:
                return {tag: 0 for tag in tag_classes} | {'sin_tag': 1}
            d = {tag: int(tag in tags) for tag in tag_classes}
            d['sin_tag'] = 0
            return d
        
        tag_df = df['tags'].apply(tag_dummies).apply(pd.Series)
        df = pd.concat([df, tag_df], axis=1)
        return df.drop(columns=["tags"])
    
    def _process_official_store(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process official store indicator"""
        df['is_official_store'] = df['official_store_id'].notna().astype(int)
        return df.drop(columns=["official_store_id"])
    
    def _process_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert boolean columns to integers"""
        for col in ["automatic_relist", "local_pick_up", "free_shipping"]:
            df[col] = df[col].replace({False: 0, True: 1})
        return df
    
    def _process_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dummy variables for mode"""
        modes = ['custom', 'not_specified', 'me1', 'me2']
        for mode in modes:
            df[mode] = (df['mode'] == mode).astype(int)
        return df.drop(columns=["mode"])
    
    def _process_warranty(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process warranty category"""
        def categorizar_garantia(row):
            sin_garantia_vals = [
                'Sin garantía',
                'Garantía de autenticidad / descripción',
                'Garantía basada en reputación del vendedor'
            ]
            if row in sin_garantia_vals:
                return pd.Series({'sin_garantia': 1, 'garantia_especifica': 0})
            else:
                return pd.Series({'sin_garantia': 0, 'garantia_especifica': 1})
        
        df[['sin_garantia', 'garantia_especifica']] = df['warranty_category'].apply(categorizar_garantia)
        return df.drop(columns=["warranty_category"])
    
    def _process_condition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert condition to numeric"""
        df['condition'] = df['condition'].replace({'new': 0, 'used': 1})
        return df
    
    def _process_seller_address(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract seller state information"""
        try:
            if "seller_address" not in df.columns:
                raise FeatureTransformationException("seller_address column not found")
            
            def safe_extract_state(x):
                try:
                    if pd.notnull(x):
                        parsed = ast.literal_eval(x)
                        if isinstance(parsed, dict):
                            return parsed.get("state", {}).get("name", None)
                    return None
                except (ValueError, SyntaxError, KeyError):
                    return None
            
            df["seller_state_name"] = df["seller_address"].apply(safe_extract_state)
            
            # Group states
            top_states = ['Capital Federal', 'Buenos Aires', 'Santa Fe', 'Córdoba', 'Mendoza']
            df['state_grouped'] = df['seller_state_name'].where(
                df['seller_state_name'].isin(top_states),
                other='Otros'
            )
            
            df = pd.get_dummies(df, columns=['state_grouped'], drop_first=True)
            return df.drop(columns=["seller_address", "seller_state_name"])
        except Exception as e:
            logger.error(f"Error processing seller address: {e}")
            raise FeatureTransformationException(f"Failed to process seller address: {e}")
    
    def _convert_boolean_columns_to_int(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert any remaining boolean columns to integers"""
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)
        return df
    
    def _process_title_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process title using TitleProcessor if available"""
        try:
            from title_predictor import TitleProcessor
            
            processor = TitleProcessor(
                model_path='../../data/modelo_final.joblib',
                scaler_path='../../data/modelo_scaler.joblib', 
                encoder_path='../../data/modelo_encoder.joblib'
            )
            
            categorias, probabilidades = processor.predict_titles(df["title"].tolist())
            df["categoria_predicha"] = categorias
            df["probabilidad_new"] = probabilidades[:, 0]
            df["probabilidad_used"] = probabilidades[:, 1]
            
            # Drop temporary columns
            df = df.drop(columns=["categoria_predicha", "title"])
            
        except Exception as e:
            logger.warning(f"Could not process titles: {e}")
            df = df.drop(columns=["title"])
        
        return df


class TrainingDatasetProcessor(DatasetProcessor):
    """Processor for training dataset - includes condition variable and applies SMOTE"""
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process training dataset with all transformations and SMOTE"""
        try:
            logger.info("[TRAINING] Starting training dataset processing...")
            logger.info(f"[TRAINING] Input data shape: {data.shape}")
            
            if data is None or data.empty:
                raise DataProcessingException("Input data is None or empty")
            
            # Select columns
            logger.info("[TRAINING] Step 1/15: Selecting columns...")
            df = self._select_columns(data)
            logger.info(f"[TRAINING] Columns selected. Shape: {df.shape}")
            
            # Apply all transformations
            logger.info("[TRAINING] Step 2/15: Processing shipping data...")
            df = self._process_shipping(df)
            logger.info(f"[TRAINING] Shipping processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 3/15: Processing payment methods...")
            df = self._process_payment_methods(df)
            logger.info(f"[TRAINING] Payment methods processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 4/15: Processing variations...")
            df = self._process_variations(df)
            logger.info(f"[TRAINING] Variations processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 5/15: Processing listing types...")
            df = self._process_listing_type(df)
            logger.info(f"[TRAINING] Listing types processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 6/15: Processing buying modes...")
            df = self._process_buying_mode(df)
            logger.info(f"[TRAINING] Buying modes processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 7/15: Processing tags...")
            df = self._process_tags(df)
            logger.info(f"[TRAINING] Tags processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 8/15: Processing official store...")
            df = self._process_official_store(df)
            logger.info(f"[TRAINING] Official store processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 9/15: Converting boolean columns...")
            df = self._process_boolean_columns(df)
            logger.info(f"[TRAINING] Boolean columns converted. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 10/15: Processing mode...")
            df = self._process_mode(df)
            logger.info(f"[TRAINING] Mode processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 11/15: Processing warranty...")
            df = self._process_warranty(df)
            logger.info(f"[TRAINING] Warranty processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 12/15: Processing condition...")
            df = self._process_condition(df)
            logger.info(f"[TRAINING] Condition processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 13/15: Processing seller address...")
            df = self._process_seller_address(df)
            logger.info(f"[TRAINING] Seller address processed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 14/15: Converting remaining boolean columns...")
            df = self._convert_boolean_columns_to_int(df)
            logger.info(f"[TRAINING] Boolean conversion completed. Shape: {df.shape}")
            
            logger.info("[TRAINING] Step 15/15: Processing title predictions...")
            df = self._process_title_predictions(df)
            logger.info(f"[TRAINING] Title predictions processed. Shape: {df.shape}")
            
            # Apply SMOTE balancing
            logger.info("[TRAINING] Applying SMOTE balancing...")
            df = self._apply_smote(df)
            
            logger.info(f"[TRAINING] Training dataset processed successfully! Final shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"[TRAINING] Error processing training dataset: {e}")
            raise DataProcessingException(f"Failed to process training dataset: {e}")
    
    def _apply_smote(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply SMOTE balancing to the dataset"""
        try:
            logger.info("[SMOTE] Validating data for SMOTE...")
            if 'condition' not in df.columns:
                raise SMOTEApplicationException("condition column not found for SMOTE")
            
            X = df.drop('condition', axis=1)
            y = df['condition']
            
            logger.info("[SMOTE] Original class distribution:")
            class_counts = y.value_counts()
            for class_val, count in class_counts.items():
                logger.info(f"[SMOTE]   Class {class_val}: {count} samples")
            
            if len(y.unique()) < 2:
                raise SMOTEApplicationException("Need at least 2 classes for SMOTE")
            
            # Identify categorical features (binary columns)
            logger.info("[SMOTE] Identifying categorical features...")
            categorical_features = [
                i for i, col in enumerate(X.columns)
                if X[col].nunique() == 2
            ]
            logger.info(f"[SMOTE] Found {len(categorical_features)} categorical features out of {len(X.columns)} total")
            
            logger.info("[SMOTE] Applying SMOTENC balancing...")
            smote_nc = SMOTENC(categorical_features=categorical_features, random_state=42)
            X_res, y_res = smote_nc.fit_resample(X, y)
            
            logger.info("[SMOTE] Reconstructing balanced dataframe...")
            df_res = pd.concat([
                pd.DataFrame(X_res, columns=X.columns),
                pd.Series(y_res, name='condition')
            ], axis=1)
            
            logger.info("[SMOTE] New class distribution:")
            new_class_counts = df_res['condition'].value_counts()
            for class_val, count in new_class_counts.items():
                logger.info(f"[SMOTE]   Class {class_val}: {count} samples")
            
            logger.info(f"[SMOTE] SMOTE completed! Original: {df.shape} -> Balanced: {df_res.shape}")
            return df_res
        except Exception as e:
            logger.error(f"[SMOTE] Error applying SMOTE: {e}")
            raise SMOTEApplicationException(f"Failed to apply SMOTE: {e}")


class TestDatasetProcessor(DatasetProcessor):
    """Processor for test dataset - excludes condition variable and SMOTE"""
    
    def __init__(self):
        super().__init__()
        # Remove condition from selected columns for test data
        self.selected_columns = [col for col in self.selected_columns if col != "condition"]
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process test dataset without condition variable and SMOTE"""
        try:
            logger.info("[TEST] Starting test dataset processing...")
            logger.info(f"[TEST] Input data shape: {data.shape}")
            
            if data is None or data.empty:
                raise DataProcessingException("Input data is None or empty")
            
            # Select columns (excluding condition)
            logger.info("[TEST] Step 1/14: Selecting columns (excluding condition)...")
            df = self._select_columns(data)
            logger.info(f"[TEST] Columns selected. Shape: {df.shape}")
            
            # Apply all transformations except condition processing
            logger.info("[TEST] Step 2/14: Processing shipping data...")
            df = self._process_shipping(df)
            logger.info(f"[TEST] Shipping processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 3/14: Processing payment methods...")
            df = self._process_payment_methods(df)
            logger.info(f"[TEST] Payment methods processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 4/14: Processing variations...")
            df = self._process_variations(df)
            logger.info(f"[TEST] Variations processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 5/14: Processing listing types...")
            df = self._process_listing_type(df)
            logger.info(f"[TEST] Listing types processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 6/14: Processing buying modes...")
            df = self._process_buying_mode(df)
            logger.info(f"[TEST] Buying modes processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 7/14: Processing tags...")
            df = self._process_tags(df)
            logger.info(f"[TEST] Tags processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 8/14: Processing official store...")
            df = self._process_official_store(df)
            logger.info(f"[TEST] Official store processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 9/14: Converting boolean columns...")
            df = self._process_boolean_columns(df)
            logger.info(f"[TEST] Boolean columns converted. Shape: {df.shape}")
            
            logger.info("[TEST] Step 10/14: Processing mode...")
            df = self._process_mode(df)
            logger.info(f"[TEST] Mode processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 11/14: Processing warranty...")
            df = self._process_warranty(df)
            logger.info(f"[TEST] Warranty processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 12/14: Processing seller address...")
            df = self._process_seller_address(df)
            logger.info(f"[TEST] Seller address processed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 13/14: Converting remaining boolean columns...")
            df = self._convert_boolean_columns_to_int(df)
            logger.info(f"[TEST] Boolean conversion completed. Shape: {df.shape}")
            
            logger.info("[TEST] Step 14/14: Processing title predictions...")
            df = self._process_title_predictions(df)
            logger.info(f"[TEST] Title predictions processed. Shape: {df.shape}")
            
            # No SMOTE for test data
            logger.info("[TEST] Skipping SMOTE balancing for test data")
            
            logger.info(f"[TEST] Test dataset processed successfully! Final shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"[TEST] Error processing test dataset: {e}")
            raise DataProcessingException(f"Failed to process test dataset: {e}")


class DatasetProcessorFactory:
    """Factory class for creating dataset processors"""
    
    @staticmethod
    def create_processor(dataset_type: str) -> DatasetProcessor:
        """
        Create a dataset processor based on type
        
        Args:
            dataset_type: Either 'training' or 'test'
            
        Returns:
            DatasetProcessor: Appropriate processor instance
            
        Raises:
            ProcessorCreationException: If processor creation fails
        """
        try:
            if not isinstance(dataset_type, str):
                raise ProcessorCreationException(f"dataset_type must be a string, got {type(dataset_type)}")
            
            dataset_type = dataset_type.lower().strip()
            
            if dataset_type == 'training':
                logger.info("Creating training dataset processor")
                return TrainingDatasetProcessor()
            elif dataset_type == 'test':
                logger.info("Creating test dataset processor")
                return TestDatasetProcessor()
            else:
                valid_types = ['training', 'test']
                raise ProcessorCreationException(
                    f"Unknown dataset type: '{dataset_type}'. Valid types: {valid_types}"
                )
        except Exception as e:
            logger.error(f"Error creating processor: {e}")
            if isinstance(e, ProcessorCreationException):
                raise
            else:
                raise ProcessorCreationException(f"Failed to create processor: {e}")