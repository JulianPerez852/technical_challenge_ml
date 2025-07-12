import pandas as pd
import numpy as np
import re
import os
import sys
import logging
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
warnings.filterwarnings('ignore')

try:
    from exceptions.exceptions import DataLoadingException, ClassificationException, DataAnalysisException, DataSavingException
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from exceptions.exceptions import DataLoadingException, ClassificationException, DataAnalysisException, DataSavingException

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class ModeloHibridoOptimizado:
    def __init__(self):
        logger.info("Inicializando ModeloHibridoOptimizado")
        # Configuración basada en el análisis real
        self.sentence_model = None
        self.tfidf_model = None
        self.scaler = StandardScaler()
        
        self.palabras_clave_new = [
            'amortiguador', 'fric', 'imprimible', 'embrague', 'llanta', 'espiral',
            'delantero', 'delanteros', 'pastilla', 'ventilado', 'nuevo', 'original',
            'kit', 'digital', 'para', 'con', 'oferta'
        ]
        
        self.palabras_clave_used = [
            'usado', 'usada', 'usados', 'poco', 'buen', 'estado', 'funcionando',
            'antiguo', 'antigua', 'vinilo', 'revista', 'lote', 'excelente',
            'impecable', 'año', 'historia', 'argentina'
        ]
        
        # CATEGORÍAS identificadas con pesos basados en %NEW
        self.categorias_productos = {
            'automotriz': {
                'palabras': ['auto', 'carro', 'vehiculo', 'motor', 'amortiguador', 'freno', 
                        'filtro', 'aceite', 'llanta', 'bateria', 'ford', 'fiat', 'renault',
                        'chevrolet', 'toyota', 'peugeot', 'volkswagen', 'repuesto'],
                'peso_new': 0.786  # 78.6% son NEW
            },
            'electronico': {
                'palabras': ['celular', 'smartphone', 'notebook', 'laptop', 'tablet', 'tv',
                            'televisor', 'monitor', 'auriculares', 'samsung', 'iphone', 'sony',
                            'lg', 'display', 'digital', 'cable', 'memoria', 'disco'],
                'peso_new': 0.615  # 61.5% son NEW
            },
            'musica': {
                'palabras': ['guitarra', 'piano', 'violin', 'bateria', 'microfono', 'amplificador',
                        'instrumento', 'musical', 'cd', 'vinilo', 'disco', 'album'],
                'peso_new': 0.754  # 75.4% son NEW
            },
            'hogar': {
                'palabras': ['mesa', 'silla', 'cama', 'sofa', 'armario', 'espejo', 'lampara',
                        'decoracion', 'cocina', 'baño', 'mueble', 'fundas', 'almohada',
                        'sabanas', 'cortinas'],
                'peso_new': 0.609  # 60.9% son NEW
            },
            'libro': {
                'palabras': ['libro', 'enciclopedia', 'manual', 'guia', 'diccionario', 'atlas',
                        'historia', 'novela', 'biografia', 'literatura', 'revista', 'comic'],
                'peso_new': 0.340  # 34.0% son NEW (más propenso a USED)
            },
            'ropa': {
                'palabras': ['zapatillas', 'zapatos', 'remera', 'pantalon', 'jean', 'camisa',
                            'vestido', 'buzo', 'campera', 'talle', 'nike', 'adidas', 'ropa'],
                'peso_new': 0.525  # 52.5% son NEW
            },
            'juguete': {
                'palabras': ['juguete', 'muneca', 'pelota', 'puzzle', 'rompecabezas', 'lego',
                            'barbie', 'disney', 'juego', 'infantil', 'niño', 'niña'],
                'peso_new': 0.587  # 58.7% son NEW
            },
            'deporte': {
                'palabras': ['bicicleta', 'pelota', 'futbol', 'tenis', 'gym', 'fitness',
                            'deportivo', 'ejercicio', 'running', 'natacion'],
                'peso_new': 0.544  # 54.4% son NEW
            }
        }
        
        # MARCAS principales identificadas (filtradas las preposiciones)
        self.marcas_principales = [
            'samsung', 'original', 'kit', 'digital', 'argentina', 'cuero', 'ford',
            'fiat', 'renault', 'sony', 'lg', 'nike', 'adidas', 'apple', 'iphone'
        ]
        
        # PATRONES REGEX optimizados
        self.patrones_regex = {
            'años_completos': re.compile(r'\b(19[5-9]\d|20[0-2]\d)\b'),  # 1950-2029
            'años_cortos': re.compile(r'\b(19|20)\b'),  # Solo 19 o 20
            'especificaciones_gb': re.compile(r'\b\d+\s*gb\b', re.IGNORECASE),
            'especificaciones_mb': re.compile(r'\b\d+\s*mb\b', re.IGNORECASE),
            'especificaciones_core': re.compile(r'\b(core\s*i\d|i[3-9])\b', re.IGNORECASE),
            'modelos_numericos': re.compile(r'\b\d{3,4}\b'),
            'versiones': re.compile(r'\bv\d+\b', re.IGNORECASE),
            'dimensiones': re.compile(r'\b\d+\s*(cm|mm|m|pulgadas|")\b', re.IGNORECASE),
            'palabras_mayusculas': re.compile(r'\b[A-Z]{2,}\b'),
            'urgencia': re.compile(r'\b(urgente|liquido|oferta|ganga|oportunidad)\b', re.IGNORECASE),
            'estado_explicito': re.compile(r'\b(nuevo|usado|seminuevo|impecable|excelente)\b', re.IGNORECASE),
            'precios': re.compile(r'\$\d+'),
            'colores': re.compile(r'\b(negro|blanco|azul|rojo|verde|amarillo|gris|rosa|violeta)\b', re.IGNORECASE)
        }
    
    def limpiar_texto(self, texto):
        """Limpieza avanzada del texto"""
        if pd.isna(texto):
            return ""
        
        texto = str(texto).lower()

        texto = re.sub(r'[^\w\s\-\.\(\)]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto)
        return texto.strip()
    
    def extraer_features_avanzadas(self, titulo):
        """Extrae features basadas en el análisis real del dataset"""
        titulo_original = titulo
        titulo_limpio = self.limpiar_texto(titulo)
        
        features = {}
        
        features['longitud_titulo'] = len(titulo_original)
        features['num_palabras'] = len(titulo_limpio.split())
        features['densidad_texto'] = len(titulo_original.replace(' ', '')) / max(len(titulo_original), 1)
        
        años_completos = self.patrones_regex['años_completos'].findall(titulo_original)
        años_cortos = self.patrones_regex['años_cortos'].findall(titulo_original)
        
        features['tiene_año_completo'] = len(años_completos) > 0
        features['num_años_completos'] = len(años_completos)
        features['tiene_año_corto'] = len(años_cortos) > 0
        
        if años_completos:
            año_mas_reciente = max([int(año) for año in años_completos])
            features['año_mas_reciente'] = año_mas_reciente
            features['antiguedad'] = 2024 - año_mas_reciente
            features['es_año_muy_antiguo'] = año_mas_reciente < 1990
            features['es_año_antiguo'] = año_mas_reciente < 2000
            features['es_año_reciente'] = año_mas_reciente >= 2020
        else:
            features['año_mas_reciente'] = 2024
            features['antiguedad'] = 0
            features['es_año_muy_antiguo'] = False
            features['es_año_antiguo'] = False
            features['es_año_reciente'] = False
        

        features['specs_gb'] = len(self.patrones_regex['especificaciones_gb'].findall(titulo_original))
        features['specs_mb'] = len(self.patrones_regex['especificaciones_mb'].findall(titulo_original))
        features['specs_core'] = len(self.patrones_regex['especificaciones_core'].findall(titulo_original))
        features['tiene_specs_tecnicas'] = features['specs_gb'] + features['specs_mb'] + features['specs_core'] > 0
        

        features['modelos_numericos'] = len(self.patrones_regex['modelos_numericos'].findall(titulo_original))
        features['versiones'] = len(self.patrones_regex['versiones'].findall(titulo_original))
        features['dimensiones'] = len(self.patrones_regex['dimensiones'].findall(titulo_original))
        

        features['score_palabras_new'] = sum(1 for palabra in self.palabras_clave_new if palabra in titulo_limpio)
        features['score_palabras_used'] = sum(1 for palabra in self.palabras_clave_used if palabra in titulo_limpio)
        features['ratio_new_used'] = features['score_palabras_new'] / max(features['score_palabras_used'], 0.1)
        features['dominancia_new'] = features['score_palabras_new'] > features['score_palabras_used']
        features['dominancia_used'] = features['score_palabras_used'] > features['score_palabras_new']
        

        features['score_categoria_weighted'] = 0
        for categoria, info in self.categorias_productos.items():
            count = sum(1 for palabra in info['palabras'] if palabra in titulo_limpio)
            features[f'es_{categoria}'] = count > 0
            features[f'count_{categoria}'] = count
            
            if count > 0:
                peso = info['peso_new'] if info['peso_new'] > 0.5 else (1 - info['peso_new'])
                features['score_categoria_weighted'] += count * peso
        

        features['es_marca_principal'] = any(marca in titulo_limpio for marca in self.marcas_principales)
        features['count_marcas'] = sum(1 for marca in self.marcas_principales if marca in titulo_limpio)
        

        features['tiene_exclamacion'] = '!' in titulo_original
        features['tiene_parentesis'] = '(' in titulo_original or ')' in titulo_original
        features['tiene_guiones'] = '-' in titulo_original
        features['num_guiones'] = titulo_original.count('-')
        
        features['palabras_mayusculas'] = len(self.patrones_regex['palabras_mayusculas'].findall(titulo_original))
        features['tiene_urgencia'] = len(self.patrones_regex['urgencia'].findall(titulo_original)) > 0
        features['tiene_estado_explicito'] = len(self.patrones_regex['estado_explicito'].findall(titulo_original)) > 0
        features['tiene_precio'] = len(self.patrones_regex['precios'].findall(titulo_original)) > 0
        features['tiene_color'] = len(self.patrones_regex['colores'].findall(titulo_original)) > 0
        

        features['es_electronico_nuevo'] = features['es_electronico'] and features['score_palabras_new'] > 0
        features['es_libro_antiguo'] = features['es_libro'] and (features['es_año_antiguo'] or features['score_palabras_used'] > 0)
        features['es_automotriz_nuevo'] = features['es_automotriz'] and features['score_palabras_new'] > 0
        

        indicadores_nuevo = ['nuevo', 'original', 'sellado', 'garantia', 'estrenar']
        indicadores_usado = ['usado', 'usada', 'estado', 'funcionando', 'poco', 'buen']
        
        features['indicadores_nuevo'] = sum(1 for ind in indicadores_nuevo if ind in titulo_limpio)
        features['indicadores_usado'] = sum(1 for ind in indicadores_usado if ind in titulo_limpio)
        features['balance_indicadores'] = features['indicadores_nuevo'] - features['indicadores_usado']
        
        return features
    
    def obtener_embeddings(self, textos):
        """Obtiene embeddings con fallback a TF-IDF"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            if self.sentence_model is None:
                print("Cargando modelo de embeddings...")
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            return self.sentence_model.encode(textos)
        else:
            if self.tfidf_model is None:
                print("Usando TF-IDF como alternativa a embeddings...")
                self.tfidf_model = TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True
                )
                return self.tfidf_model.fit_transform(textos).toarray()
            else:
                return self.tfidf_model.transform(textos).toarray()
    
    def preparar_datos_completos(self, df):
        """Prepara dataset completo con embeddings + features"""
        try:
            logger.info(f"Preparando datos para {len(df)} registros...")
            print(f"Preparando datos para {len(df)} registros...")
        

            textos_limpios = [self.limpiar_texto(titulo) for titulo in df['title']]

            embeddings = self.obtener_embeddings(textos_limpios)
            print(f"Embeddings shape: {embeddings.shape}")
            

            print("Extrayendo features avanzadas...")
            features_list = []
            for titulo in df['title']:
                features_list.append(self.extraer_features_avanzadas(titulo))
            
            features_df = pd.DataFrame(features_list)
            print(f"Features engineeradas: {features_df.shape[1]} features")

            if len(embeddings.shape) == 2:
                embeddings_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
            else:
                embeddings_df = pd.DataFrame(embeddings.reshape(len(embeddings), -1), 
                                        columns=[f'emb_{i}' for i in range(embeddings.shape[0])])
            
            datos_finales = pd.concat([embeddings_df.reset_index(drop=True), 
                                    features_df.reset_index(drop=True)], axis=1)
            
            print(f"Dataset final: {datos_finales.shape}")
            logger.info(f"Datos preparados exitosamente: {datos_finales.shape}")
            return datos_finales, features_df
        except Exception as e:
            logger.error(f"Error preparando datos: {str(e)}")
            raise DataAnalysisException(f"Error preparando datos: {str(e)}") from e
    
    def entrenar_modelo_optimizado(self, X, y):
        """Entrena ensemble de modelos optimizado"""
        try:
            logger.info("Iniciando entrenamiento de modelos optimizados")
            print("\nEntrenando modelos optimizados...")
            

            print("Validando tipos de datos...")
            

            if hasattr(X, 'values'):
                X = X.values
            X = np.array(X, dtype=np.float32)

            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            if y.dtype == 'object':
                y_encoded = le.fit_transform(y)
                print(f"Etiquetas convertidas: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            else:
                y_encoded = y
                le = None
            

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            print(f"Train: {len(X_train)}, Test: {len(X_test)}")
            print(f"Distribución train: {np.bincount(y_train)}")
            print(f"Distribución test: {np.bincount(y_test)}")

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            

            print("Verificando datos...")
            print(f"NaN en X_train: {np.isnan(X_train).sum()}")
            print(f"NaN en X_train_scaled: {np.isnan(X_train_scaled).sum()}")
            print(f"Inf en X_train_scaled: {np.isinf(X_train_scaled).sum()}")
            

            X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
            

            modelos = {
                'RandomForest': RandomForestClassifier(
                    n_estimators=200,  
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ),
                'LogisticRegression': LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    C=1.0,
                    class_weight='balanced',
                    solver='liblinear'  
                ),
                'MLPClassifier': MLPClassifier(
                    hidden_layer_sizes=(100, 50), 
                    max_iter=300, 
                    random_state=42,
                    alpha=0.01,  
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    learning_rate_init=0.001,
                    solver='adam'
                )
            }
            
            # Entrenar modelos individuales
            resultados = {}
            modelos_entrenados = {}
            
            for nombre, modelo in modelos.items():
                print(f"\nEntrenando {nombre}...")
                
                try:
                    if nombre in ['LogisticRegression', 'MLPClassifier']:
                        modelo.fit(X_train_scaled, y_train)
                        pred = modelo.predict(X_test_scaled)
                        proba = modelo.predict_proba(X_test_scaled)[:, 1]
                        score = modelo.score(X_test_scaled, y_test)
                    else:
                        modelo.fit(X_train, y_train)
                        pred = modelo.predict(X_test)
                        proba = modelo.predict_proba(X_test)[:, 1]
                        score = modelo.score(X_test, y_test)
                    
                    # Convertir predicciones de vuelta a etiquetas originales si es necesario
                    if le is not None:
                        pred_original = le.inverse_transform(pred)
                        y_test_original = le.inverse_transform(y_test)
                    else:
                        pred_original = pred
                        y_test_original = y_test
                    
                    auc = roc_auc_score(y_test, proba)
                    
                    resultados[nombre] = {
                        'accuracy': score,
                        'auc': auc,
                        'predictions': pred_original,
                        'probabilities': proba,
                        'y_test': y_test_original
                    }
                    
                    modelos_entrenados[nombre] = modelo
                    
                    print(f"  Accuracy: {score:.4f}")
                    print(f"  AUC: {auc:.4f}")
                    
                except Exception as e:
                    print(f"  Error en {nombre}: {str(e)}")
                    print(f"     Continuando sin este modelo...")
                    continue
            

            if not modelos_entrenados:
                raise Exception("No se pudo entrenar ningún modelo!")
            

            mejor_modelo_nombre = max(resultados.keys(), key=lambda x: resultados[x]['auc'])
            ensemble_final = modelos_entrenados[mejor_modelo_nombre]
            
            print(f"\nMejor modelo seleccionado: {mejor_modelo_nombre}")

            if mejor_modelo_nombre in ['LogisticRegression', 'MLPClassifier']:
                ensemble_pred = ensemble_final.predict(X_test_scaled)
                ensemble_proba = ensemble_final.predict_proba(X_test_scaled)[:, 1]
                ensemble_score = ensemble_final.score(X_test_scaled, y_test)
            else:
                ensemble_pred = ensemble_final.predict(X_test)
                ensemble_proba = ensemble_final.predict_proba(X_test)[:, 1]
                ensemble_score = ensemble_final.score(X_test, y_test)
            
            ensemble_auc = roc_auc_score(y_test, ensemble_proba)
            

            if le is not None:
                ensemble_pred_original = le.inverse_transform(ensemble_pred)
                y_test_original = le.inverse_transform(y_test)
            else:
                ensemble_pred_original = ensemble_pred
                y_test_original = y_test
            
            print(f"\nModelo Final - {mejor_modelo_nombre}:")
            print(f"  Accuracy: {ensemble_score:.4f}")
            print(f"  AUC: {ensemble_auc:.4f}")
            

            print(f"\nREPORTE FINAL:")
            print("=" * 50)
            print(classification_report(y_test_original, ensemble_pred_original))
            

            if mejor_modelo_nombre == 'RandomForest' and hasattr(ensemble_final, 'feature_importances_'):
                self.mostrar_feature_importance(ensemble_final, X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])])
            
            logger.info(f"Entrenamiento completado exitosamente - Modelo: {mejor_modelo_nombre}, Accuracy: {ensemble_score:.4f}, AUC: {ensemble_auc:.4f}")
            
            return {
                'modelo_final': ensemble_final,
                'scaler': self.scaler,
                'label_encoder': le,
                'mejor_modelo_nombre': mejor_modelo_nombre,
                'resultados': resultados,
                'X_test': X_test,
                'y_test': y_test_original,
                'accuracy': ensemble_score,
                'auc': ensemble_auc,
                'usa_scaling': mejor_modelo_nombre in ['LogisticRegression', 'MLPClassifier']
            }
        except Exception as e:
            logger.error(f"Error durante entrenamiento de modelos: {str(e)}")
            raise ClassificationException(f"Error durante entrenamiento de modelos: {str(e)}") from e
    
    def mostrar_feature_importance(self, modelo, feature_names):
        """Muestra las features más importantes"""
        if hasattr(modelo, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': modelo.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTOP 20 FEATURES MÁS IMPORTANTES:")
            print("-" * 60)
            for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
                print(f"{i+1:2d}. {row['feature']:<35} {row['importance']:.4f}")
    
    def predecir_nuevo_titulo(self, titulo, modelo_info):
        """Predice la condición de un título nuevo"""
        try:
            logger.info(f"Iniciando predicción para título: {titulo[:50]}...")
            print(f"\nPREDICCIÓN PARA: '{titulo}'")
            print("-" * 80)

            df_temp = pd.DataFrame({'title': [titulo]})
            X_temp, features_temp = self.preparar_datos_completos(df_temp)
            
            X_temp = np.array(X_temp, dtype=np.float32)
            X_temp = np.nan_to_num(X_temp, nan=0.0, posinf=1e6, neginf=-1e6)
            

            modelo = modelo_info['modelo_final']
            if modelo_info.get('usa_scaling', False):
                X_temp = modelo_info['scaler'].transform(X_temp)
            

            prediccion_encoded = modelo.predict(X_temp)[0]
            probabilidades = modelo.predict_proba(X_temp)[0]
            

            le = modelo_info.get('label_encoder')
            if le is not None:
                prediccion = le.inverse_transform([prediccion_encoded])[0]
            else:
                prediccion = prediccion_encoded
            
            print(f"Predicción: {prediccion.upper()}")
            print(f"Probabilidades: NEW={probabilidades[1]:.3f}, USED={probabilidades[0]:.3f}")
            print(f"Confianza: {max(probabilidades):.3f}")
            

            features_dict = features_temp.iloc[0].to_dict()
            features_importantes = {k: v for k, v in features_dict.items() 
                                if v > 0 and not k.startswith('count_') and not k.startswith('es_')}
            
            print(f"\n Features detectadas relevantes:")
            for feature, valor in list(features_importantes.items())[:10]:
                print(f"  • {feature}: {valor}")
            

            categorias_detectadas = [k.replace('es_', '') for k, v in features_dict.items() 
                                    if k.startswith('es_') and v == True]
            if categorias_detectadas:
                print(f"\nCategorías detectadas: {', '.join(categorias_detectadas)}")
            
            logger.info(f"Predicción completada: {prediccion} (confianza: {max(probabilidades):.3f})")
            return prediccion, probabilidades
        except Exception as e:
            logger.error(f"Error en predicción para título '{titulo[:50]}...': {str(e)}")
            raise ClassificationException(f"Error en predicción: {str(e)}") from e


def entrenar_modelo_completo(df):
    """Función principal para entrenar el modelo con tu dataset"""
    try:
        logger.info("Iniciando entrenamiento completo del modelo híbrido optimizado")
        print("INICIANDO ENTRENAMIENTO DEL MODELO HÍBRIDO OPTIMIZADO")
        print("=" * 80)
    

        if 'title' not in df.columns or 'condition' not in df.columns:
            logger.error("DataFrame debe contener columnas 'title' y 'condition'")
            raise DataLoadingException("El DataFrame debe tener columnas 'title' y 'condition'")
        
        logger.info(f"Dataset validado: {len(df)} registros con columnas requeridas")
    
        print(f"Dataset: {len(df)} registros")
        print(f"   - NEW: {len(df[df['condition'] == 'new'])}")
        print(f"   - USED: {len(df[df['condition'] == 'used'])}")
        
        # Inicializar modelo
        modelo = ModeloHibridoOptimizado()
        
        X, features_df = modelo.preparar_datos_completos(df)
        y = df['condition']
        

        resultado = modelo.entrenar_modelo_optimizado(X, y)
        
        print(f"\nENTRENAMIENTO COMPLETADO")
        print(f"Accuracy Final: {resultado['accuracy']:.4f}")
        print(f"AUC Score: {resultado['auc']:.4f}")
    
        resultado['modelo_processor'] = modelo
        
        logger.info(f"Entrenamiento completo finalizado exitosamente - Accuracy: {resultado['accuracy']:.4f}, AUC: {resultado['auc']:.4f}")
        return resultado
    except (DataLoadingException, DataAnalysisException, ClassificationException) as e:
        logger.error(f"Error en entrenamiento completo: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado en entrenamiento completo: {e}")
        raise ClassificationException(f"Error inesperado en entrenamiento completo: {e}") from e


def ejemplo_prediccion(modelo_info):
    """Ejemplos de predicción con el modelo entrenado"""
    try:
        logger.info("Iniciando ejemplos de predicción")
        modelo_processor = modelo_info['modelo_processor']
    
        titulos_ejemplo = [
            "Amortiguador Delantero Ford Focus 2015 Original Nuevo",
            "Libro Historia Argentina Antiguo 1967 Excelente Estado",
            "Samsung Galaxy S21 256GB Nuevo Sellado Garantía",
            "Vinilo Los Beatles Revolver 1966 Usado Buen Estado",
            "Kit Embrague Fiat Palio 2020 Original Para Mecánico"
        ]
        
        print("\EJEMPLOS DE PREDICCIÓN:")
        print("=" * 80)
        
        for titulo in titulos_ejemplo:
                prediccion, prob = modelo_processor.predecir_nuevo_titulo(titulo, modelo_info)
                print()
        
        logger.info("Ejemplos de predicción completados exitosamente")
    except Exception as e:
        logger.error(f"Error en ejemplos de predicción: {e}")
        raise ClassificationException(f"Error en ejemplos de predicción: {e}") from e

def save_results_to_file(modelo_entrenado, output_dir='../../data'):
    """Guarda los resultados del entrenamiento en un archivo de texto"""
    try:
        logger.info("Guardando resultados en archivo de texto")
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar nombre de archivo con timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f'title_model_training_results_{timestamp}.txt')
        
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DEL ENTRENAMIENTO - MODELO HÍBRIDO OPTIMIZADO\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Fecha de entrenamiento: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("INFORMACIÓN DEL MODELO:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mejor modelo seleccionado: {modelo_entrenado['mejor_modelo_nombre']}\n")
            f.write(f"Accuracy final: {modelo_entrenado['accuracy']:.4f}\n")
            f.write(f"AUC Score: {modelo_entrenado['auc']:.4f}\n")
            f.write(f"Usa scaling: {modelo_entrenado['usa_scaling']}\n\n")
            

            f.write("RESULTADOS DE MODELOS INDIVIDUALES:\n")
            f.write("-" * 40 + "\n")
            for nombre, resultados in modelo_entrenado['resultados'].items():
                f.write(f"{nombre}:\n")
                f.write(f"  Accuracy: {resultados['accuracy']:.4f}\n")
                f.write(f"  AUC: {resultados['auc']:.4f}\n\n")
            

            f.write("ARCHIVOS GENERADOS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"- Modelo final: modelo_final.joblib\n")
            f.write(f"- Scaler: modelo_scaler.joblib\n")
            if modelo_entrenado.get('label_encoder') is not None:
                f.write(f"- Label Encoder: modelo_encoder.joblib\n")
            f.write(f"- Resultados: {os.path.basename(results_file)}\n\n")
        
        logger.info(f"Resultados guardados exitosamente en: {results_file}")
        return results_file
    
    except Exception as e:
        logger.error(f"Error guardando resultados: {str(e)}")
        raise DataSavingException(f"Error guardando resultados: {str(e)}") from e

if __name__ == "__main__":
    try:
        logger.info("Iniciando script principal de entrenamiento")
        
        input_path = '../../data/x_train.csv'
        logger.info(f"Cargando dataset desde: {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Dataset cargado: {len(df)} registros")
        
        modelo_entrenado = entrenar_modelo_completo(df)
        ejemplo_prediccion(modelo_entrenado)
        
        output_dir = '../../data'
        logger.info(f"Guardando modelos en directorio: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(modelo_entrenado['modelo_final'], os.path.join(output_dir, 'modelo_final.joblib'))
        joblib.dump(modelo_entrenado['scaler'], os.path.join(output_dir, 'modelo_scaler.joblib'))
        if 'label_encoder' in modelo_entrenado and modelo_entrenado['label_encoder'] is not None:
            joblib.dump(modelo_entrenado['label_encoder'], os.path.join(output_dir, 'modelo_encoder.joblib'))
        
        logger.info("Modelos guardados exitosamente")
        
        results_file = save_results_to_file(modelo_entrenado, output_dir)
        
        logger.info("Script principal completado exitosamente")
        
    except (DataLoadingException, DataAnalysisException, ClassificationException, DataSavingException) as e:
        logger.error(f"Error en script principal: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error inesperado en script principal: {e}")
        print(f"Error inesperado: {e}")
        sys.exit(1)