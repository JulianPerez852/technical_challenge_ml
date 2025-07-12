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
import torch
import joblib

warnings.filterwarnings('ignore')

def check_gpu_availability():
    """Verificar disponibilidad de GPU para diferentes librerías"""
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    }
    
    # Verificar LightGBM GPU
    try:
        import lightgbm as lgb
        lgb_gpu = True
    except:
        lgb_gpu = False
    
    gpu_info['lightgbm_gpu'] = lgb_gpu
    
    # Verificar XGBoost GPU
    try:
        import xgboost as xgb
        xgb_gpu = True
    except:
        xgb_gpu = False
    
    gpu_info['xgboost_gpu'] = xgb_gpu
    
    return gpu_info

def print_gpu_info():
    """Mostrar información de GPU disponible"""
    gpu_info = check_gpu_availability()
    
    print("="*60)
    print("INFORMACIÓN DE GPU")
    print("="*60)
    print(f"CUDA disponible: {'✅' if gpu_info['cuda_available'] else '❌'}")
    if gpu_info['cuda_available']:
        print(f"Dispositivos CUDA: {gpu_info['cuda_device_count']}")
        print(f"GPU principal: {gpu_info['cuda_device_name']}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"LightGBM GPU: {'✅' if gpu_info['lightgbm_gpu'] else '❌'}")
    print(f"XGBoost GPU: {'✅' if gpu_info['xgboost_gpu'] else '❌'}")
    print("="*60)
    
    return gpu_info


def load_and_prepare_data():
    """Cargar y preparar los datos"""
    print("Cargando datos...")
    data = pd.read_csv('../data/training_data_pca.csv')
    data_test = pd.read_csv('../data/test_data_pca.csv')
    y_test = pd.read_csv("../data/y_test.csv")
        
    y_test['0'] = y_test['0'].replace({'new': 0, 'used': 1})
    y_test = y_test['0']
        
    X = data.drop('condition', axis=1)
    y = data['condition']

    print(f"Features de entrenamiento: {len(X.columns)}")
    print(f"Features de test: {len(data_test.columns)}")

    return X, y, data_test, y_test

def create_gpu_optimized_models(gpu_info):
    """Crear modelos optimizados para GPU cuando esté disponible"""
    models = {}
    
    # RandomForest (CPU optimizado)
    # models['RandomForest_CPU'] = (
    #     RandomForestClassifier(
    #         random_state=42, 
    #         n_jobs=-1,  # Usar todos los cores CPU
    #         verbose=1
    #     ),
    #     {
    #         'n_estimators': [200, 300, 500],
    #         'max_depth': [10, 15, 20, None],
    #         'min_samples_split': [2, 5],
    #         'min_samples_leaf': [1, 2],
    #         'max_features': ['sqrt', 'log2']
    #     }
    # )
    
    # LightGBM con GPU si está disponible
    if gpu_info['lightgbm_gpu'] and gpu_info['cuda_available']:
        print("🚀 Configurando LightGBM con GPU...")
        models['LightGBM_GPU'] = (
            LGBMClassifier(
                random_state=42,
                device='gpu',  # Usar GPU
                gpu_platform_id=0,
                gpu_device_id=0,
                verbosity=-1,
                objective='binary',
                boosting_type='gbdt',
                num_threads=1  # Para GPU, usar 1 thread
            ),
            {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        )
    else:
        print("⚠️ GPU no disponible para LightGBM, usando CPU...")
        models['LightGBM_CPU'] = (
            LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                objective='binary'
            ),
            {
                'n_estimators': [200, 300, 500],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7, -1],
                'num_leaves': [31, 63, 127],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
        )
    
    # XGBoost con GPU si está disponible
    if gpu_info['xgboost_gpu'] and gpu_info['cuda_available']:
        print("🚀 Configurando XGBoost con GPU...")
        models['XGBoost_GPU'] = (
            XGBClassifier(
                random_state=42,
                tree_method='gpu_hist',  # Usar GPU
                gpu_id=0,
                eval_metric='logloss',
                verbosity=1,
                n_jobs=1  # Para GPU, usar 1 job
            ),
            {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        )
    else:
        print("⚠️ GPU no disponible para XGBoost, usando CPU...")
        models['XGBoost_CPU'] = (
            XGBClassifier(
                random_state=42,
                tree_method='hist',  # Optimizado para CPU
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0
            ),
            {
                'n_estimators': [200, 300, 500],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
        )
    
    # Gradient Boosting (sklearn - solo CPU)
    models['GradientBoosting_CPU'] = (
        GradientBoostingClassifier(
            random_state=42,
            verbose=1
        ),
            {
                'n_estimators': [200, 300, 500],
                'learning_rate': [0.1],
                'max_depth': [7,9],
                'subsample': [0.7,0.8],
                'min_samples_split': [2, 5]
        }
    )
    
    # Logistic Regression (optimizado)
    # models['LogisticRegression_Optimized'] = (
    #     Pipeline([
    #         ('scaler', StandardScaler()),
    #         ('lr', LogisticRegression(
    #             random_state=42, 
    #             max_iter=2000,
    #             n_jobs=-1  # Paralelización
    #         ))
    #     ]),
    #     {
    #         'lr__C': [0.01, 0.1, 1, 10, 100],
    #         'lr__penalty': ['l1', 'l2', 'elasticnet'],
    #         'lr__solver': ['saga'],  # Solver que soporta todos los penalties
    #         'lr__l1_ratio': [0.1, 0.5, 0.7, 0.9]  # Solo para elasticnet
    #     }
    # )
    
    return models

def create_quick_gpu_models(gpu_info):
    """Versión rápida con GPU para pruebas"""
    models = {}
    
    # RandomForest rápido
    models['RandomForest_Quick'] = (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            'n_estimators': [100, 200],
            'max_depth': [10, None],
            'min_samples_split': [2]
        }
    )
    
    # LightGBM rápido con GPU si está disponible
    if gpu_info['lightgbm_gpu'] and gpu_info['cuda_available']:
        models['LightGBM_GPU_Quick'] = (
            LGBMClassifier(
                random_state=42,
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                verbosity=-1,
                objective='binary'
            ),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [5, 7],
                'num_leaves': [31, 63]
            }
        )
    else:
        models['LightGBM_CPU_Quick'] = (
            LGBMClassifier(
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                objective='binary'
            ),
            {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.2],
                'max_depth': [5, 7]
            }
        )
    
    # XGBoost rápido con GPU si está disponible
    if gpu_info['xgboost_gpu'] and gpu_info['cuda_available']:
        models['XGBoost_GPU_Quick'] = (
            XGBClassifier(
                random_state=42,
                tree_method='gpu_hist',
                gpu_id=0,
                eval_metric='logloss',
                verbosity=0
            ),
            {
                'n_estimators': [100, 200],
                'max_depth': [5, 7],
                'learning_rate': [0.1, 0.2]
            }
        )
    else:
        models['XGBoost_CPU_Quick'] = (
            XGBClassifier(
                random_state=42,
                tree_method='hist',
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0
            ),
            {
                'n_estimators': [100, 200],
                'max_depth': [5, 7],
                'learning_rate': [0.1, 0.2]
            }
        )
    
    return models

def calculate_total_combinations(param_grid):
    """Calcular el número total de combinaciones de hiperparámetros"""
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    return total_combinations

def train_single_model(name, model, params, X, y, X_test, y_test):
    """Entrenar un modelo individual - función para paralelización"""
    print(f'\n{"="*60}')
    print(f'🔥 Entrenando: {name}')
    if 'GPU' in name:
        print('⚡ MODO GPU ACTIVADO')
    else:
        print('💻 Modo CPU')
    print(f'🕐 Fecha/Hora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'{"="*60}')
    
    # Calcular total de combinaciones
    total_combinations = calculate_total_combinations(params)
    total_fits = total_combinations * 3  # 3-fold CV
    print(f'📊 Total de combinaciones a probar: {total_combinations}')
    print(f'🔄 Total de fits (con CV): {total_fits}')
    estimated_time = total_combinations * 2  # Estimación: 2 segundos por combinación
    print(f'⏱️ Tiempo estimado: ~{estimated_time/60:.1f} minutos')
    
    try:
        start_time = time.time()
        
        # GridSearchCV con configuración específica para GPU/CPU
        n_jobs = 1 if 'GPU' in name else -1
        verbose_level = 1 if 'GPU' in name else 2
        
        grid = GridSearchCV(
            model, 
            params, 
            cv=3,
            scoring='accuracy',
            n_jobs=n_jobs,
            verbose=verbose_level,
            error_score='raise',
            return_train_score=True
        )
        
        # Entrenar
        print(f"🚀 Iniciando entrenamiento...")
        if 'GPU' in name:
            print("⚡ Utilizando aceleración GPU...")
        else:
            print(f"💻 Utilizando {-1 if n_jobs == -1 else n_jobs} cores CPU...")
        
        grid.fit(X, y)
        
        training_time = time.time() - start_time
        
        # Predicciones
        print(f"\n🎯 Realizando predicciones...")
        pred_start = time.time()
        y_pred = grid.predict(X_test)
        prediction_time = time.time() - pred_start
        
        # Métricas
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Resultados
        result = {
            'model': grid.best_estimator_,
            'cv_score': grid.best_score_,
            'test_accuracy': test_accuracy,
            'best_params': grid.best_params_,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'total_combinations': total_combinations,
            'gpu_accelerated': 'GPU' in name,
            'grid_results': grid.cv_results_
        }
        
        print(f'\n{"="*40}')
        print(f'📈 RESULTADOS PARA {name}:')
        print(f'{"="*40}')
        print(f'⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos')
        print(f'⚡ Tiempo de predicción: {prediction_time:.4f} segundos')
        print(f'🎯 Mejor CV Score: {grid.best_score_:.4f}')
        print(f'📊 Accuracy en Test: {test_accuracy:.4f}')
        print(f'⚙️ Mejores parámetros: {grid.best_params_}')
        
        if 'GPU' in name:
            print(f'🚀 Aceleración GPU utilizada exitosamente!')
        
        # Mostrar top 3 mejores configuraciones
        print(f'\n🏆 Top 3 configuraciones:')
        results_df = pd.DataFrame(grid.cv_results_)
        top_3 = results_df.nlargest(3, 'mean_test_score')[['mean_test_score', 'params']]
        for i, (_, row) in enumerate(top_3.iterrows(), 1):
            print(f'{i}. Score: {row["mean_test_score"]:.4f}, Params: {row["params"]}')
        
        print(f'\n📋 Reporte de clasificación:')
        print(classification_report(y_test, y_pred))
        
        return name, result
        
    except Exception as e:
        print(f'❌ Error entrenando {name}: {str(e)}')
        return name, None

def train_models_sequential(models, X, y, X_test, y_test):
    """Entrenar modelos secuencialmente (para GPU)"""
    print("🔄 Entrenamiento secuencial (GPU)...")
    results = {}
    best_models = {}
    
    for name, (model, params) in models.items():
        result_name, result_data = train_single_model(name, model, params, X, y, X_test, y_test)
        if result_data is not None:
            best_models[result_name] = result_data['model']
            # Remover el modelo del resultado para evitar problemas de serialización
            result_data_clean = {k: v for k, v in result_data.items() if k != 'model'}
            results[result_name] = result_data_clean
    
    return best_models, results

def train_models_parallel(models, X, y, X_test, y_test):
    """Entrenar modelos en paralelo (para CPU)"""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    
    print(f"🚀 Entrenamiento paralelo (CPU) con {mp.cpu_count()} cores...")
    
    results = {}
    best_models = {}
    
    # Determinar número de workers (no más de 4 para evitar sobrecarga)
    max_workers = min(len(models), mp.cpu_count(), 4)
    print(f"👥 Usando {max_workers} workers paralelos...")
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Enviar tareas
            future_to_name = {
                executor.submit(train_single_model, name, model, params, X, y, X_test, y_test): name
                for name, (model, params) in models.items()
            }
            
            # Recoger resultados conforme se completan
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    result_name, result_data = future.result()
                    if result_data is not None:
                        best_models[result_name] = result_data['model']
                        # Remover el modelo del resultado
                        result_data_clean = {k: v for k, v in result_data.items() if k != 'model'}
                        results[result_name] = result_data_clean
                        print(f"✅ Completado: {result_name}")
                    else:
                        print(f"❌ Falló: {name}")
                except Exception as e:
                    print(f"❌ Error en worker para {name}: {str(e)}")
                    
    except Exception as e:
        print(f"❌ Error en paralelización: {str(e)}")
        print("🔄 Cambiando a modo secuencial...")
        return train_models_sequential(models, X, y, X_test, y_test)
    
    return best_models, results

def train_and_evaluate_models_hybrid(X, y, X_test, y_test, models):
    """Estrategia híbrida: GPU secuencial + CPU paralelo"""
    print("🚀 Iniciando estrategia híbrida de entrenamiento...")
    
    # Separar modelos por tipo
    gpu_models = {k: v for k, v in models.items() if 'GPU' in k}
    cpu_models = {k: v for k, v in models.items() if 'GPU' not in k}
    
    all_best_models = {}
    all_results = {}
    
    # 1. Entrenar modelos GPU secuencialmente (más rápidos)
    if gpu_models:
        print(f"\n🚀 FASE 1: Entrenando {len(gpu_models)} modelos GPU secuencialmente...")
        gpu_best_models, gpu_results = train_models_sequential(gpu_models, X, y, X_test, y_test)
        all_best_models.update(gpu_best_models)
        all_results.update(gpu_results)
        print("✅ Fase GPU completada!")
    
    # 2. Entrenar modelos CPU en paralelo
    if cpu_models:
        print(f"\n💻 FASE 2: Entrenando {len(cpu_models)} modelos CPU en paralelo...")
        cpu_best_models, cpu_results = train_models_parallel(cpu_models, X, y, X_test, y_test)
        all_best_models.update(cpu_best_models)
        all_results.update(cpu_results)
        print("✅ Fase CPU completada!")
    
    # 3. Si no hay separación GPU/CPU, usar método tradicional
    if not gpu_models and not cpu_models:
        print("🔄 No se detectó separación GPU/CPU, usando método secuencial...")
        return train_models_sequential(models, X, y, X_test, y_test)
    
    print(f"\n🎉 Entrenamiento híbrido completado!")
    print(f"📊 Total de modelos entrenados: {len(all_results)}")
    
    return all_best_models, all_results

def train_and_evaluate_models(X, y, X_test, y_test, models):
    """Función principal de entrenamiento con estrategia adaptativa"""
    
    # Configuración para seguimiento rápido
    quick_mode = input("¿Ejecutar en modo rápido? (reduce hiperparámetros) [y/N]: ").lower() == 'y'
    
    if quick_mode:
        print("🚀 Modo rápido activado - usando menos hiperparámetros...")
        gpu_info = check_gpu_availability()
        models = create_quick_gpu_models(gpu_info)
    
    # Detectar si hay modelos GPU y CPU
    gpu_models_exist = any('GPU' in name for name in models.keys())
    cpu_models_exist = any('GPU' not in name for name in models.keys())
    
    print(f"\n📋 Análisis de modelos:")
    print(f"   - Modelos GPU: {sum(1 for name in models.keys() if 'GPU' in name)}")
    print(f"   - Modelos CPU: {sum(1 for name in models.keys() if 'GPU' not in name)}")
    
    # Seleccionar estrategia
    if gpu_models_exist and cpu_models_exist:
        print("🎯 Estrategia: Híbrida (GPU secuencial + CPU paralelo)")
        return train_and_evaluate_models_hybrid(X, y, X_test, y_test, models)
    elif gpu_models_exist:
        print("🚀 Estrategia: Solo GPU (secuencial)")
        return train_models_sequential(models, X, y, X_test, y_test)
    else:
        print("💻 Estrategia: Solo CPU (paralelo)")
        return train_models_parallel(models, X, y, X_test, y_test)

def save_best_models(best_models, results):
    """Guardar los mejores modelos"""
    print("\n💾 Guardando mejores modelos...")
    
    # Encontrar el mejor modelo por accuracy
    if results:
        best_model_name = max(results.items(), key=lambda x: x[1]['test_accuracy'])[0]
        best_model = best_models[best_model_name]
        
        # Guardar mejor modelo
        joblib.dump(best_model, '../data/best_model_gpu.joblib')
        print(f"✅ Mejor modelo guardado: {best_model_name}")
        
        # Guardar información
        model_info = {
            'best_model_name': best_model_name,
            'test_accuracy': results[best_model_name]['test_accuracy'],
            'cv_score': results[best_model_name]['cv_score'],
            'best_params': results[best_model_name]['best_params'],
            'gpu_accelerated': results[best_model_name]['gpu_accelerated']
        }
        
        joblib.dump(model_info, '../data/best_model_info_gpu.joblib')
        print(f"✅ Información del modelo guardada")

def display_gpu_summary(results):
    """Mostrar resumen con información de GPU"""
    print(f'\n{"="*80}')
    print('🎯 RESUMEN FINAL DE RESULTADOS CON OPTIMIZACIÓN GPU')
    print(f'{"="*80}')
    
    # Crear DataFrame para comparación
    summary_data = []
    for name, metrics in results.items():
        gpu_status = "🚀 GPU" if metrics['gpu_accelerated'] else "💻 CPU"
        summary_data.append({
            'Modelo': name,
            'Aceleración': gpu_status,
            'CV Score': f"{metrics['cv_score']:.4f}",
            'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
            'Diferencia': f"{metrics['cv_score'] - metrics['test_accuracy']:.4f}",
            'Tiempo Entrenamiento (s)': f"{metrics['training_time']:.1f}",
            'Tiempo Predicción (s)': f"{metrics['prediction_time']:.4f}",
            'Combinaciones': metrics['total_combinations']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Test Accuracy', ascending=False)
    print(summary_df.to_string(index=False))
    
    # Análisis por aceleración
    gpu_models = {k: v for k, v in results.items() if v['gpu_accelerated']}
    cpu_models = {k: v for k, v in results.items() if not v['gpu_accelerated']}
    
    if gpu_models and cpu_models:
        print(f'\n⚡ Comparación GPU vs CPU:')
        
        # Mejor modelo GPU
        best_gpu = max(gpu_models.items(), key=lambda x: x[1]['test_accuracy'])
        print(f'🚀 Mejor GPU: {best_gpu[0]} - {best_gpu[1]["test_accuracy"]:.4f} ({best_gpu[1]["training_time"]:.1f}s)')
        
        # Mejor modelo CPU
        best_cpu = max(cpu_models.items(), key=lambda x: x[1]['test_accuracy'])
        print(f'💻 Mejor CPU: {best_cpu[0]} - {best_cpu[1]["test_accuracy"]:.4f} ({best_cpu[1]["training_time"]:.1f}s)')
        
        # Speedup promedio si hay modelos equivalentes
        gpu_times = [v['training_time'] for v in gpu_models.values()]
        cpu_times = [v['training_time'] for v in cpu_models.values()]
        
        if gpu_times and cpu_times:
            avg_gpu_time = np.mean(gpu_times)
            avg_cpu_time = np.mean(cpu_times)
            speedup = avg_cpu_time / avg_gpu_time
            print(f"📈 Speedup promedio GPU: {speedup:.2f}x")
    
    # Mejor modelo general
    best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
    acceleration = "🚀 GPU" if best_model[1]['gpu_accelerated'] else "💻 CPU"
    print(f'\n🏆 Mejor modelo: {best_model[0]} ({acceleration}) con accuracy: {best_model[1]["test_accuracy"]:.4f}')
    
    return summary_df

def main():
    """Función principal con optimización GPU"""
    print("="*80)
    print("🚀 PIPELINE DE MACHINE LEARNING CON OPTIMIZACIÓN GPU")
    print("="*80)
    
    # Verificar GPU
    gpu_info = print_gpu_info()
    
    # Cargar datos
    X, y, X_test, y_test = load_and_prepare_data()
    
    print(f"\n📊 Información de los datos:")
    print(f"- Training set: {X.shape}")
    print(f"- Test set: {X_test.shape}")
    print(f"- Distribución de clases: {y.value_counts().to_dict()}")
    
    # Crear modelos optimizados para GPU
    models = create_gpu_optimized_models(gpu_info)
    
    print(f"\n🤖 Modelos a entrenar: {list(models.keys())}")
    
    # Entrenar y evaluar
    best_models, results = train_and_evaluate_models(X, y, X_test, y_test, models)
    
    # Mostrar resumen
    if results:
        display_gpu_summary(results)
        save_best_models(best_models, results)
    
    return best_models, results

if __name__ == "__main__":
    best_models, results = main()