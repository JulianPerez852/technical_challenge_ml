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
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

class ProgressCallback:
    """Callback para mostrar progreso del GridSearchCV"""
    def __init__(self, total_fits):
        self.total_fits = total_fits
        self.current_fit = 0
        self.start_time = time.time()
    
    def __call__(self, *args, **kwargs):
        self.current_fit += 1
        elapsed = time.time() - self.start_time
        progress = (self.current_fit / self.total_fits) * 100
        eta = (elapsed / self.current_fit) * (self.total_fits - self.current_fit)
        
        print(f"\rProgreso: {progress:.1f}% ({self.current_fit}/{self.total_fits}) - "
              f"Tiempo: {elapsed:.1f}s - ETA: {eta:.1f}s", end="", flush=True)

def load_and_prepare_data():
    """Cargar y preparar los datos"""
    print("Cargando datos...")
    data = pd.read_csv('../data/training_data_features_selected.csv')
    # data.drop(columns=["title"], inplace=True)
    data_test = pd.read_csv('../data/test_data_features_selected.csv')
    # data_test.drop(columns=["title"], inplace=True)

    y_test = pd.read_csv("../data/y_test.csv")
        
    y_test['0'] = y_test['0'].replace({'new': 0, 'used': 1})
    y_test = y_test['0']
        
    X = data.drop('condition', axis=1)
    y = data['condition']

    print(X.columns,len(X.columns) )
    print(data_test.columns,len(data_test.columns) )

    return X, y, data_test, y_test


# def load_and_prepare_data():
#     """Cargar y preparar los datos"""

#     print("Cargando datos...")
#     data = pd.read_csv('data_train_pca.csv')
#     data_test = pd.read_csv('data_test_pca.csv')
#     y_test = pd.read_csv("../data/y_test.csv")

#     # Procesar etiquetas
#     y_test['0'] = y_test['0'].replace({'new': 0, 'used': 1})
#     y_test = y_test['0']

#     # Separar características y etiquetas de entrenamiento
#     X = data.drop('condition', axis=1)
#     y = data['condition']


    # return X, y, data_test, y_test

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
            
            # GridSearchCV con seguimiento
            grid = GridSearchCV(
                model, 
                params, 
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2,  # Aumentamos verbosity para más información
                error_score='raise',
                return_train_score=True
            )
            
            # Entrenar
            print(f"Iniciando entrenamiento...")
            grid.fit(X, y)
            
            training_time = time.time() - start_time
            
            # Guardar mejor modelo
            best_models[name] = grid.best_estimator_
            
            # Predicciones
            print(f"\nRealizando predicciones...")
            y_pred = grid.predict(X_test)
            
            # Métricas
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Guardar resultados
            results[name] = {
                'cv_score': grid.best_score_,
                'test_accuracy': test_accuracy,
                'best_params': grid.best_params_,
                'training_time': training_time,
                'total_fits': total_fits
            }
            
            print(f'\n{"="*40}')
            print(f'RESULTADOS PARA {name}:')
            print(f'{"="*40}')
            print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')
            print(f'Mejor CV Score: {grid.best_score_:.4f}')
            print(f'Accuracy en Test: {test_accuracy:.4f}')
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
            print(f'Error entrenando {name}: {str(e)}')
            continue
    
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
    print("="*80)
    print("PIPELINE DE MACHINE LEARNING OPTIMIZADO CON SEGUIMIENTO")
    print("="*80)
    
    # Cargar datos
    X, y, X_test, y_test = load_and_prepare_data()
    
    print(f"\nInformación de los datos:")
    print(f"- Training set: {X.shape}")
    print(f"- Test set: {X_test.shape}")
    print(f"- Distribución de clases: {y.value_counts().to_dict()}")
    
    # Verificar si hay columna price
    if 'price' in X.columns:
        print(f"- Rango de precios: {X['price'].min():.2f} - {X['price'].max():.2f}")
        print(f"- Precio promedio: {X['price'].mean():.2f}")
    
    # Crear modelos
    models = create_optimized_models()
    
    print(f"\nModelos a entrenar: {list(models.keys())}")
    
    # Entrenar y evaluar
    best_models, results = train_and_evaluate_models(X, y, X_test, y_test, models)
    
    # Mostrar resumen
    if results:
        display_summary(results)
    
    return best_models, results

if __name__ == "__main__":
    best_models, results = main()