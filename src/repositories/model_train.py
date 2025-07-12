import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, recall_score
from xgboost import XGBClassifier
import warnings
import time
from datetime import datetime
import logging
import sys
import os
import joblib
warnings.filterwarnings('ignore')

# Exception imports with fallback logic similar to other repositories files
try:
    from exceptions.exceptions import DataLoadingException, DataSavingException, ClassificationException, ModelTrainingException
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from exceptions.exceptions import DataLoadingException, DataSavingException, ClassificationException, ModelTrainingException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Clase principal para el entrenamiento y evaluación de modelos de ML"""
    
    def __init__(self, data_train_path='../../data/training_data_features_selected.csv',
                 data_test_path='../../data/test_data_features_selected.csv',
                 y_test_path='../../data/y_test.csv',
                 models_output_dir='../../data/',
                 metrics_output_path='../../data/training_metrics.csv'):
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path
        self.y_test_path = y_test_path
        self.models_output_dir = models_output_dir
        self.metrics_output_path = metrics_output_path
        self.best_models = {}
        self.results = {}
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        
    def load_and_prepare_data(self):
        """Cargar y preparar los datos"""
        try:
            logger.info("Cargando datos...")
            data = pd.read_csv(self.data_train_path)
            data_test = pd.read_csv(self.data_test_path)
            y_test = pd.read_csv(self.y_test_path)
            
            logger.info(f"Datos de entrenamiento cargados: {data.shape}")
            logger.info(f"Datos de prueba cargados: {data_test.shape}")
            
            # Procesar etiquetas
            y_test['0'] = y_test['0'].replace({'new': 0, 'used': 1})
            y_test = y_test['0']
            
            # Separar características y etiquetas de entrenamiento
            self.X = data.drop('condition', axis=1)
            self.y = data['condition']
            self.X_test = data_test
            self.y_test = y_test
            
            logger.info(f"Características de entrenamiento: {self.X.columns.tolist()}, Total: {len(self.X.columns)}")
            logger.info(f"Características de prueba: {self.X_test.columns.tolist()}, Total: {len(self.X_test.columns)}")
            logger.info(f"Distribución de clases: {self.y.value_counts().to_dict()}")
            
        except FileNotFoundError as e:
            logger.error(f"Archivo no encontrado: {e}")
            raise DataLoadingException(f"Error al cargar archivos de datos: {e}") from e
        except Exception as e:
            logger.error(f"Error inesperado al cargar datos: {e}")
            raise DataLoadingException(f"Error inesperado al cargar datos: {e}") from e
    
    def save_models(self):
        """Guardar los modelos entrenados"""
        try:
            if not self.best_models:
                logger.warning("No hay modelos para guardar")
                return
            
            logger.info("Guardando modelos entrenados...")
            os.makedirs(self.models_output_dir, exist_ok=True)
            
            for name, model in self.best_models.items():
                model_path = os.path.join(self.models_output_dir, f'{name}_best_model.joblib')
                joblib.dump(model, model_path)
                logger.info(f"Modelo {name} guardado en {model_path}")
            
            logger.info("Todos los modelos han sido guardados exitosamente")
            
        except Exception as e:
            logger.error(f"Error al guardar modelos: {e}")
            raise DataSavingException(f"Error al guardar modelos: {e}") from e
    
    def save_metrics(self):
        """Guardar las métricas de ejecución en un archivo CSV"""
        try:
            if not self.results:
                logger.warning("No hay métricas para guardar")
                return
            
            logger.info("Guardando métricas de ejecución...")
            
            # Crear DataFrame con las métricas
            metrics_data = []
            for name, metrics in self.results.items():
                metrics_data.append({
                    'modelo': name,
                    'cv_score': metrics['cv_score'],
                    'test_accuracy': metrics['test_accuracy'],
                    'test_recall': metrics['test_recall'],
                    'diferencia_cv_test': metrics['cv_score'] - metrics['test_accuracy'],
                    'tiempo_entrenamiento_segundos': metrics['training_time'],
                    'total_fits': metrics['total_fits'],
                    'mejores_parametros': str(metrics['best_params']),
                    'fecha_entrenamiento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Ordenar por accuracy y luego por recall
            metrics_df = metrics_df.sort_values(['test_accuracy', 'test_recall'], ascending=[False, False])
            
            # Guardar CSV
            os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
            metrics_df.to_csv(self.metrics_output_path, index=False)
            logger.info(f"Métricas guardadas en {self.metrics_output_path}")
            
            return metrics_df
            
        except Exception as e:
            logger.error(f"Error al guardar métricas: {e}")
            raise DataSavingException(f"Error al guardar métricas: {e}") from e
    
    def select_best_model(self):
        """Seleccionar el mejor modelo basado en accuracy y recall"""
        try:
            if not self.results:
                logger.warning("No hay resultados para seleccionar el mejor modelo")
                return None, None
            
            # Ordenar modelos por accuracy (primaria) y recall (secundaria)
            sorted_models = sorted(
                self.results.items(),
                key=lambda x: (x[1]['test_accuracy'], x[1]['test_recall']),
                reverse=True
            )
            
            best_model_name, best_metrics = sorted_models[0]
            best_model = self.best_models[best_model_name]
            
            logger.info(f"Mejor modelo seleccionado: {best_model_name}")
            logger.info(f"  - Accuracy: {best_metrics['test_accuracy']:.4f}")
            logger.info(f"  - Recall: {best_metrics['test_recall']:.4f}")
            logger.info(f"  - CV Score: {best_metrics['cv_score']:.4f}")
            
            # Guardar el mejor modelo por separado
            best_model_path = os.path.join(self.models_output_dir, 'best_model_overall.joblib')
            joblib.dump(best_model, best_model_path)
            logger.info(f"Mejor modelo guardado por separado en {best_model_path}")
            
            return best_model_name, best_model
            
        except Exception as e:
            logger.error(f"Error al seleccionar el mejor modelo: {e}")
            raise ModelTrainingException(f"Error al seleccionar el mejor modelo: {e}") from e
    
    def train_and_evaluate_models(self, quick_mode=None):
        """Entrenar y evaluar modelos con validación cruzada"""
        try:
            # Configuración para seguimiento rápido (opcional)
            if quick_mode is None:
                quick_mode = input("¿Ejecutar en modo rápido? (reduce hiperparámetros) [y/N]: ").lower() == 'y'
            
            # Crear modelos
            if quick_mode:
                print("Modo rápido activado - usando menos hiperparámetros...")
                models = create_quick_models()
            else:
                models = create_optimized_models()
            
            for name, (model, params) in models.items():
                print(f'\n{"="*60}')
                print(f'Entrenando: {name}')
                print(f'Fecha/Hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                print(f'{"="*60}')
                
                # Calcular total de fits
                total_fits = calculate_total_fits(params, cv_folds=3)
                print(f'Total de combinaciones a probar: {total_fits // 3}')
                print(f'Total de fits (con CV): {total_fits}')
                
                try:
                    start_time = time.time()
                    logger.info(f"Iniciando entrenamiento para {name}...")
                    
                    # GridSearchCV con seguimiento
                    grid = GridSearchCV(
                        model, 
                        params, 
                        cv=3,
                        scoring='accuracy',
                        n_jobs=-1,
                        verbose=2,
                        error_score='raise',
                        return_train_score=True
                    )
                    
                    # Entrenar
                    grid.fit(self.X, self.y)
                    training_time = time.time() - start_time
                    logger.info(f"Entrenamiento completado para {name} en {training_time:.2f} segundos")
                    
                    # Guardar mejor modelo
                    self.best_models[name] = grid.best_estimator_
                    
                    # Predicciones
                    logger.info(f"Realizando predicciones para {name}...")
                    y_pred = grid.predict(self.X_test)
                    
                    # Métricas
                    test_accuracy = accuracy_score(self.y_test, y_pred)
                    test_recall = recall_score(self.y_test, y_pred, average='weighted')
                    
                    # Guardar resultados
                    self.results[name] = {
                        'cv_score': grid.best_score_,
                        'test_accuracy': test_accuracy,
                        'test_recall': test_recall,
                        'best_params': grid.best_params_,
                        'training_time': training_time,
                        'total_fits': total_fits
                    }
                    
                    logger.info(f'Resultados para {name}: CV Score: {grid.best_score_:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Recall: {test_recall:.4f}')
                    
                    print(f'\n{"="*40}')
                    print(f'RESULTADOS PARA {name}:')
                    print(f'{"="*40}')
                    print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')
                    print(f'Mejor CV Score: {grid.best_score_:.4f}')
                    print(f'Accuracy en Test: {test_accuracy:.4f}')
                    print(f'Recall en Test: {test_recall:.4f}')
                    print(f'Mejores parámetros: {grid.best_params_}')
                    
                    # Mostrar top 3 mejores configuraciones
                    print(f'\nTop 3 configuraciones:')
                    results_df = pd.DataFrame(grid.cv_results_)
                    top_3 = results_df.nlargest(3, 'mean_test_score')[['mean_test_score', 'params']]
                    for i, (_, row) in enumerate(top_3.iterrows(), 1):
                        print(f'{i}. Score: {row["mean_test_score"]:.4f}, Params: {row["params"]}')
                    
                    print(f'\nReporte de clasificación:')
                    print(classification_report(self.y_test, y_pred))
                    
                except Exception as e:
                    logger.error(f'Error entrenando {name}: {str(e)}')
                    raise ModelTrainingException(f'Error entrenando {name}: {str(e)}') from e
            
            return self.best_models, self.results
            
        except Exception as e:
            logger.error(f"Error en el entrenamiento de modelos: {e}")
            raise ModelTrainingException(f"Error en el entrenamiento de modelos: {e}") from e
    
    def run_full_pipeline(self, quick_mode=None):
        """Ejecutar el pipeline completo de entrenamiento"""
        try:
            logger.info("="*80)
            logger.info("PIPELINE DE MACHINE LEARNING OPTIMIZADO CON SEGUIMIENTO")
            logger.info("="*80)
            
            print("="*80)
            print("PIPELINE DE MACHINE LEARNING OPTIMIZADO CON SEGUIMIENTO")
            print("="*80)
            
            # Cargar datos
            self.load_and_prepare_data()
            
            print(f"\nInformación de los datos:")
            print(f"- Training set: {self.X.shape}")
            print(f"- Test set: {self.X_test.shape}")
            print(f"- Distribución de clases: {self.y.value_counts().to_dict()}")
            
            # Verificar si hay columna price
            if 'price' in self.X.columns:
                print(f"- Rango de precios: {self.X['price'].min():.2f} - {self.X['price'].max():.2f}")
                print(f"- Precio promedio: {self.X['price'].mean():.2f}")
            
            # Entrenar modelos
            self.train_and_evaluate_models(quick_mode)
            
            # Guardar modelos
            self.save_models()
            
            # Guardar métricas
            metrics_df = self.save_metrics()
            
            # Seleccionar y guardar el mejor modelo
            best_name, best_model = self.select_best_model()
            
            # Mostrar resumen
            if self.results:
                display_summary(self.results)
                logger.info("Entrenamiento completado exitosamente para todos los modelos")
            else:
                logger.warning("No se obtuvieron resultados del entrenamiento")
            
            return self.best_models, self.results, best_name, best_model
            
        except (DataLoadingException, ModelTrainingException, DataSavingException) as e:
            logger.error(f"Error en el pipeline de ML: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado en el pipeline de ML: {e}")
            raise ModelTrainingException(f"Error inesperado en el pipeline de ML: {e}") from e

# Función legacy mantenida para compatibilidad
def load_and_prepare_data(data_train_path='../../data/training_data_features_selected.csv',
                        data_test_path='../../data/test_data_features_selected.csv',
                        y_test_path='../../data/y_test.csv'):
    """Cargar y preparar los datos (función legacy)"""
    trainer = ModelTrainer(data_train_path, data_test_path, y_test_path)
    trainer.load_and_prepare_data()
    return trainer.X, trainer.y, trainer.X_test, trainer.y_test


def create_optimized_models():
    """Crear modelos con pipelines optimizados"""
    models = {
        'RandomForest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        ),
        
        # 'RandomForest_Scaled': (
        #     Pipeline([
        #         ('scaler', StandardScaler()),
        #         ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
        #     ]),
        #     {
        #         'rf__n_estimators': [100, 200, 500],
        #         'rf__max_depth': [10, 20, None],
        #         'rf__min_samples_split': [2, 5],
        #         'rf__min_samples_leaf': [1, 2]
        #     }
        # ),
        
        # 'LogisticRegression': (
        #     Pipeline([
        #         ('scaler', StandardScaler()),
        #         ('lr', LogisticRegression(random_state=42, max_iter=1000))
        #     ]),
        #     {
        #         'lr__C': [0.1, 1, 10, 100],
        #         'lr__penalty': ['l1', 'l2'],
        #         'lr__solver': ['liblinear', 'saga']
        #     }
        # ),
        
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'n_estimators': [200, 300, 500],
                'learning_rate': [0.1],
                'max_depth': [7,9],
                'subsample': [0.7,0.8],
                'min_samples_split': [2, 5]
            }
        ),
        
        # 'ExtraTrees': (
        #     ExtraTreesClassifier(random_state=42, n_jobs=-1),
        #     {
        #         'n_estimators': [100, 200, 300],
        #         'max_depth': [10, 20, None],
        #         'min_samples_split': [2, 5],
        #         'min_samples_leaf': [1, 2],
        #         'max_features': ['sqrt', 'log2']
        #     }
        # ),
        
        'LightGBM': (
            LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                objective='binary'
            ),
            {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        ),
        
        # 'MLP': (
        #     Pipeline([
        #         ('scaler', StandardScaler()),
        #         ('mlp', MLPClassifier(random_state=42, max_iter=1000, early_stopping=True))
        #     ]),
        #     {
        #         'mlp__hidden_layer_sizes': [(200, 100, 50), (256, 128, 64)],
        #         'mlp__activation': ['relu', 'tanh'],
        #         'mlp__alpha': [0.001, 0.01],
        #         'mlp__learning_rate': ['adaptive']
        #     }
        # ),
        
        # 'XGBoost': (
        #     XGBClassifier(
        #         random_state=42,
        #         eval_metric='logloss',
        #         n_jobs=-1,
        #         verbosity=0
        #     ),
        #     {
        #         'n_estimators': [100, 200],
        #         'max_depth': [3, 6, 9],
        #         'learning_rate': [0.01, 0.1, 0.2],
        #         'subsample': [0.8, 1.0],
        #         'colsample_bytree': [0.8, 1.0]
        #     }
        # ),
        
        # 'XGBoost_Scaled': (
        #     Pipeline([
        #         ('scaler', StandardScaler()),
        #         ('xgb', XGBClassifier(
        #             random_state=42,
        #             eval_metric='logloss',
        #             n_jobs=-1,
        #             verbosity=0
        #         ))
        #     ]),
        #     {
        #         'xgb__n_estimators': [100, 200],
        #         'xgb__max_depth': [3, 6, 9],
        #         'xgb__learning_rate': [0.01, 0.1, 0.2],
        #         'xgb__subsample': [0.8, 1.0],
        #         'xgb__colsample_bytree': [0.8, 1.0]
        #     }
        # )
    }
    
    return models

def calculate_total_fits(param_grid, cv_folds=3):
    """Calcular el número total de fits para el progreso"""
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    return total_combinations * cv_folds

def train_and_evaluate_models(X, y, X_test, y_test, models):
    """Entrenar y evaluar modelos con validación cruzada"""
    best_models = {}
    results = {}
    
    # Configuración para seguimiento rápido (opcional)
    quick_mode = input("¿Ejecutar en modo rápido? (reduce hiperparámetros) [y/N]: ").lower() == 'y'
    
    if quick_mode:
        print("Modo rápido activado - usando menos hiperparámetros...")
        models = create_quick_models()
    
    for name, (model, params) in models.items():
        print(f'\n{"="*60}')
        print(f'Entrenando: {name}')
        print(f'Fecha/Hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'{"="*60}')
        
        # Calcular total de fits
        total_fits = calculate_total_fits(params, cv_folds=3)
        print(f'Total de combinaciones a probar: {total_fits // 3}')
        print(f'Total de fits (con CV): {total_fits}')
        
        try:
            start_time = time.time()
            logger.info(f"Iniciando entrenamiento para {name}...")
            
            # GridSearchCV con seguimiento
            grid = GridSearchCV(
                model, 
                params, 
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2,
                error_score='raise',
                return_train_score=True
            )
            
            # Entrenar
            grid.fit(X, y)
            training_time = time.time() - start_time
            logger.info(f"Entrenamiento completado para {name} en {training_time:.2f} segundos")
            
            # Guardar mejor modelo
            best_models[name] = grid.best_estimator_
            
            # Predicciones
            logger.info(f"Realizando predicciones para {name}...")
            y_pred = grid.predict(X_test)
            
            # Métricas
            test_accuracy = accuracy_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred, average='weighted')
            
            # Guardar resultados
            results[name] = {
                'cv_score': grid.best_score_,
                'test_accuracy': test_accuracy,
                'test_recall': test_recall,
                'best_params': grid.best_params_,
                'training_time': training_time,
                'total_fits': total_fits
            }
            
            logger.info(f'Resultados para {name}: CV Score: {grid.best_score_:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Recall: {test_recall:.4f}')
            
            print(f'\n{"="*40}')
            print(f'RESULTADOS PARA {name}:')
            print(f'{"="*40}')
            print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')
            print(f'Mejor CV Score: {grid.best_score_:.4f}')
            print(f'Accuracy en Test: {test_accuracy:.4f}')
            print(f'Recall en Test: {test_recall:.4f}')
            print(f'Mejores parámetros: {grid.best_params_}')
            
            # Mostrar top 3 mejores configuraciones
            print(f'\nTop 3 configuraciones:')
            results_df = pd.DataFrame(grid.cv_results_)
            top_3 = results_df.nlargest(3, 'mean_test_score')[['mean_test_score', 'params']]
            for i, (_, row) in enumerate(top_3.iterrows(), 1):
                print(f'{i}. Score: {row["mean_test_score"]:.4f}, Params: {row["params"]}')
            
            print(f'\nReporte de clasificación:')
            print(classification_report(y_test, y_pred))
            
        except Exception as e:
            logger.error(f'Error entrenando {name}: {str(e)}')
            raise ModelTrainingException(f'Error entrenando {name}: {str(e)}') from e
    
    return best_models, results

def create_quick_models():
    """Versión rápida con menos hiperparámetros para pruebas"""
    models = {
        'RandomForest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                'n_estimators': [100],
                'max_depth': [10, None],
                'min_samples_split': [2]
            }
        ),
        
        'LogisticRegression': (
            Pipeline([
                ('scaler', StandardScaler()),
                ('lr', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            {
                'lr__C': [1, 10],
                'lr__penalty': ['l2'],
                'lr__solver': ['liblinear']
            }
        ),
        
        'GradientBoosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5]
            }
        ),
        
        'LightGBM': (
            LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                objective='binary'
            ),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [3, 5],
                'num_leaves': [31, 50]
            }
        ),
        
        'XGBoost': (
            XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0
            ),
            {
                'n_estimators': [100],
                'max_depth': [6],
                'learning_rate': [0.1]
            }
        )
    }
    return models

def display_summary(results):
    """Mostrar resumen de resultados"""
    print(f'\n{"="*80}')
    print('RESUMEN FINAL DE RESULTADOS')
    print(f'{"="*80}')
    
    # Crear DataFrame para comparación
    summary_data = []
    for name, metrics in results.items():
        summary_data.append({
            'Modelo': name,
            'CV Score': f"{metrics['cv_score']:.4f}",
            'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
            'Diferencia': f"{metrics['cv_score'] - metrics['test_accuracy']:.4f}",
            'Tiempo (s)': f"{metrics['training_time']:.1f}",
            'Total Fits': metrics['total_fits']
        })
    
    summary_df = pd.DataFrame(summary_data)
    # Ordenar por test accuracy
    summary_df = summary_df.sort_values('Test Accuracy', ascending=False)
    print(summary_df.to_string(index=False))
    
    # Mejor modelo
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f'\n🏆 Mejor modelo: {best_model[0]} con accuracy: {best_model[1]["test_accuracy"]:.4f}')
    
    # Modelo más rápido
    fastest_model = min(results.items(), key=lambda x: x[1]['training_time'])
    print(f'⚡ Modelo más rápido: {fastest_model[0]} en {fastest_model[1]["training_time"]:.1f}s')
    
    # Análisis por categorías de modelos
    tree_models = {k: v for k, v in results.items() if any(x in k for x in ['RandomForest', 'ExtraTrees', 'XGBoost', 'LightGBM', 'GradientBoosting'])}
    linear_models = {k: v for k, v in results.items() if any(x in k for x in ['LogisticRegression', 'MLP'])}
    
    if tree_models:
        print(f'\n🌳 Comparación de modelos basados en árboles:')
        for name, metrics in sorted(tree_models.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
            print(f'{name}: {metrics["test_accuracy"]:.4f} (tiempo: {metrics["training_time"]:.1f}s)')
        
        best_tree = max(tree_models.items(), key=lambda x: x[1]['test_accuracy'])
        fastest_tree = min(tree_models.items(), key=lambda x: x[1]['training_time'])
        print(f'🎯 Mejor árbol: {best_tree[0]} ({best_tree[1]["test_accuracy"]:.4f})')
        print(f'⚡ Árbol más rápido: {fastest_tree[0]} ({fastest_tree[1]["training_time"]:.1f}s)')
    
    if linear_models:
        print(f'\n📏 Comparación de modelos lineales:')
        for name, metrics in sorted(linear_models.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
            print(f'{name}: {metrics["test_accuracy"]:.4f} (tiempo: {metrics["training_time"]:.1f}s)')
    
    # Recomendaciones específicas para clasificación binaria
    print(f'\n💡 Recomendaciones para clasificación binaria:')
    if len(results) > 1:
        sorted_by_accuracy = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
        top_3 = sorted_by_accuracy[:3]
        
        print(f'- Top 3 modelos por accuracy:')
        for i, (name, metrics) in enumerate(top_3, 1):
            print(f'  {i}. {name}: {metrics["test_accuracy"]:.4f}')
        
        # Verificar si algún modelo de gradient boosting está en el top
        gb_models = [name for name, _ in top_3 if any(x in name for x in ['XGBoost', 'LightGBM', 'GradientBoosting'])]
        if gb_models:
            print(f'- Gradient boosting funciona bien: considera {gb_models[0]}')
        
        # Verificar overfitting
        for name, metrics in results.items():
            diff = metrics['cv_score'] - metrics['test_accuracy']
            if diff > 0.05:
                print(f'- {name} tiene posible overfitting (diff: {diff:.3f})')
            elif diff < -0.02:
                print(f'- {name} podría mejorar con más complejidad (diff: {diff:.3f})')
    
    return summary_df

def main():
    """Función principal"""
    try:
        # Crear instancia del trainer
        trainer = ModelTrainer(
            data_train_path='../../data/training_data_features_selected.csv',
            data_test_path='../../data/test_data_features_selected.csv',
            y_test_path='../../data/y_test.csv',
            models_output_dir='../../data/',
            metrics_output_path='../../data/training_metrics.csv'
        )
        
        # Ejecutar pipeline completo
        best_models, results, best_name, best_model = trainer.run_full_pipeline()
        
        return best_models, results, best_name, best_model
        
    except (DataLoadingException, ModelTrainingException, DataSavingException) as e:
        logger.error(f"Error en el pipeline de ML: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado en el pipeline de ML: {e}")
        raise ModelTrainingException(f"Error inesperado en el pipeline de ML: {e}") from e

if __name__ == "__main__":
    try:
        best_models, results, best_name, best_model = main()
        
        print(f"\n{'='*80}")
        print("RESUMEN FINAL")
        print(f"{'='*80}")
        print(f"Mejor modelo: {best_name}")
        print(f"Total de modelos entrenados: {len(best_models)}")
        print(f"Métricas guardadas en: ../../data/training_metrics.csv")
        print(f"Modelos guardados en: ../../data/")
        
    except (DataLoadingException, ModelTrainingException, DataSavingException) as e:
        logger.error(f"Error en el entrenamiento de modelos: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        sys.exit(1)