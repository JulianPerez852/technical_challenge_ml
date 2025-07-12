# technical_challenge_ml
Repositorio para el reto técnico de Machine Learning.

## Paso a paso para ejecutar y entender la prueba

1. **Inicialización del proyecto**
   - Ejecuta `uv init` para inicializar el proyecto.
   - Luego ejecuta `uv sync` para crear el entorno virtual `.venv` y sincronizar dependencias.

2. **Estructura del código**
   - Se crearon los módulos `exceptions`, `legacy` y `repositories`:
     - `exceptions`: gestor de excepciones personalizado.
     - `legacy`: contiene el código original entregado.
     - `repositories`: contiene el nuevo código para tareas específicas, incluyendo el análisis de garantía.

3. **Exploración de datos**
   - Se creó el notebook `exploration.ipynb` para identificar y explorar el dataset.

4. **Clasificación de la columna 'warranty'**
   - Se desarrolló el archivo `warranty_handler.py` en `repositories`, donde se analiza la columna de garantía y se genera una columna adicional con la clasificación encontrada.
   - El análisis se realiza por lotes para optimizar el procesamiento.

5. **Instalación de dependencias para clasificación**
   - Es necesario instalar `torch` y `transformers` en el ambiente virtual, ya sea agregándolos al `pyproject.toml` o ejecutando:
     ```sh
     uv pip install torch transformers ........... (depende de tu hardware)
     ```

6. **Ejecución del análisis de garantía**
   - El procedimiento puede demorar incluso si tienes GPU, ya que el análisis por lotes de la columna es intensivo.
   - Para ejecutarlo, abre una consola en la carpeta `repositories` y ejecuta el script:
     ```sh
     uv run warranty_handler.py
     ```
   - Se recomienda ir por un café mientras se genera el resultado para los archivos de train y test.

---
Este flujo permite analizar la columna de garantía de manera distinta y obtener una clasificación automatizada.


El resultado son 2 dataset train y test en pandas con la variable de warranty_category

---

## Análisis de títulos con LLM (Ollama)

El siguiente paso fue tratar el análisis de títulos con un modelo de lenguaje grande (LLM). Para este caso utilicé el modelo `gemma3n:e4b`, ya que puedo utilizarlo en mi GPU sin problema y a una velocidad decente. Sin embargo, esa "velocidad decente" implicaba dejar mi equipo procesando lotes por 2 o 3 días seguidos, lo que hacía inviable completar el análisis en el tiempo disponible y no era el enfoque más adecuado para esta prueba.

Aun así, si se quiere probar el método, se deben seguir estos pasos:

1. Instalar Ollama siguiendo la documentación oficial.
2. Realizar `ollama pull gemma3n:e4b` para descargar el modelo, o correr directamente la línea `ollama run gemma3n:e4b` en otra consola.
3. Desde la carpeta `repositories`, ejecutar el archivo correspondiente al análisis de títulos (`uv run ollama_title_handler.py`).
4. Si se desea usar este método para todo el dataset, tener en cuenta que el procesamiento puede tomar varios días.

Esta aproximación permite extraer información semántica avanzada de los títulos, pero requiere recursos computacionales y tiempo considerable.

---

## Análisis automático de títulos (TitleAnalyzer)

Como alternativa al análisis con LLM, se desarrolló un sistema de análisis automático de títulos que utiliza técnicas de procesamiento de texto tradicionales para extraer características y patrones útiles para la clasificación de productos.

### Funcionalidades del TitleAnalyzer

El archivo `title_explorer.py` en `repositories` implementa la clase `TitleAnalyzer` que realiza un análisis comprehensivo de los títulos, incluyendo:

**Estadísticas básicas:**
- Longitud promedio y mediana de títulos por condición (nuevo/usado)
- Número promedio de palabras por título
- Distribución de características textuales

**Extracción de palabras clave:**
- Identifica las palabras más frecuentes para productos nuevos vs usados
- Calcula palabras distintivas con ratios de aparición significativos
- Filtra palabras cortas y poco informativas

**Detección de patrones numéricos:**
- Años de fabricación (19XX, 20XX)
- Especificaciones técnicas (GB, MB, MPX, Core i5/i7, RAM)
- Números de modelo (3-4 dígitos)
- Dimensiones y medidas

**Identificación automática de marcas:**
- Detecta palabras en mayúsculas y capitalizadas como candidatos a marcas
- Analiza distribución de marcas por condición del producto
- Filtra candidatos por frecuencia de aparición

**Categorización de productos:**
- Clasifica automáticamente en categorías: electrónicos, automotriz, libros, ropa, hogar, juguetes, deportes, música
- Utiliza diccionarios de palabras clave específicas por categoría
- Calcula estadísticas de distribución por condición

**Detección de patrones especiales:**
- Signos de exclamación, palabras en mayúsculas
- Precios, medidas, colores
- Palabras de urgencia (urgente, líquido, oferta)
- Indicadores de condición del producto

### Ejecución del análisis

Para ejecutar el análisis automático de títulos:
Recuerda estar en repositories
```sh
uv run title_explorer.py
```

**Resultado:**
El script genera un archivo JSON (`title_analysis_results.json`) que contiene:
- Palabras clave distintivas para productos nuevos y usados
- Lista de marcas identificadas automáticamente
- Categorías de productos detectadas
- Patrones regex útiles para feature engineering
- Estadísticas detalladas de todos los análisis
- Recomendaciones para uso en modelos de ML

Este análisis es significativamente más rápido que el enfoque con LLM y proporciona características estructuradas que pueden ser directamente utilizadas como features en modelos de machine learning para la clasificación de condición de productos.

---

## Entrenamiento de modelos híbridos para predicción de títulos

El archivo `title_model_trainer.py` en la carpeta `repositories` implementa el flujo completo para entrenar modelos de machine learning que predicen la condición del producto (nuevo/usado) a partir de los títulos, utilizando un enfoque híbrido basado en embeddings y features extraídas del análisis previo.

### Características principales
- **Modelo híbrido:** Utiliza embeddings (Sentence Transformers o TF-IDF) junto con features ingenieradas del análisis automático de títulos.
- **Entrenamiento de múltiples modelos:** Entrena tres modelos principales:
  - `modelo_final.joblib`: El modelo de ML seleccionado como el mejor (RandomForest, LogisticRegression o MLPClassifier).
  - `modelo_scaler.joblib`: El scaler utilizado para normalizar los datos.
  - `modelo_encoder.joblib`: El encoder de etiquetas (LabelEncoder) para transformar las clases.
- **Selección automática del mejor modelo:** El script evalúa accuracy y AUC para seleccionar el modelo final.
- **Reporte detallado:** Se genera un archivo de resultados en `data` con métricas de entrenamiento y detalles de los modelos.

### Ejecución

Para entrenar el modelo híbrido y guardar los artefactos:

Recuerda estar en la carpeta `repositories`:
```sh
uv run title_model_trainer.py
```

Los resultados y modelos generados se guardan en la carpeta `data`:
- `modelo_final.joblib`: Modelo final para predicción de títulos.
- `modelo_scaler.joblib`: Scaler para normalización.
- `modelo_encoder.joblib`: Encoder de etiquetas.
- `title_model_training_results_<timestamp>.txt`: Reporte de entrenamiento con métricas y detalles.

### Notas técnicas
- El modelo utiliza las features extraídas en el análisis automático (`title_explorer.py`) para mejorar la predicción.
- El enfoque híbrido permite combinar información semántica (embeddings) y estructurada (features textuales y categóricas).
- El script selecciona el mejor modelo según AUC y accuracy, y guarda todos los artefactos necesarios para reproducir la predicción en nuevos títulos.

---

## Procesamiento completo del dataset y preparación para modelos

Después de la predicción de títulos, se realiza un proceso exhaustivo de obtención y procesado de las variables que se van a utilizar en el entrenamiento de modelos. Este proceso fue explorado inicialmente en el notebook Jupyter `eda.ipynb` y posteriormente se convirtió en un script independiente llamado `process_datasets.py`.

### Funcionalidades del procesamiento de datos

El pipeline de procesamiento incluye las siguientes etapas principales:

**1. Extracción de características de campos complejos:**
- **Shipping**: Extrae `local_pick_up`, `free_shipping`, y `mode` de datos JSON anidados
- **Métodos de pago**: Procesa la lista de métodos de pago no-MercadoPago y crea variables dummy para cada tipo (Visa, MasterCard, Transferencia bancaria, Efectivo, etc.)
- **Variaciones de productos**: Calcula número de variaciones, cantidad disponible promedio por variación
- **Tags**: Crea variables dummy para tags importantes como `dragged_bids_and_visits`, `good_quality_thumbnail`, etc.

**2. Codificación de variables categóricas:**
- **Tipos de listado**: One-hot encoding para `bronze`, `silver`, `free`, `gold_special`, `gold`, `gold_premium`, `gold_pro`
- **Modos de compra**: Variables dummy para `buy_it_now`, `classified`, `auction`
- **Modos de envío**: Codificación para `custom`, `not_specified`, `me1`, `me2`

**3. Procesamiento geográfico:**
- Extracción del estado del vendedor desde `seller_address` (campo JSON complejo)
- Agrupación de estados en top 5 (Capital Federal, Buenos Aires, Santa Fe, Córdoba, Mendoza) + "Otros"
- One-hot encoding de los grupos de estados

**4. Procesamiento de garantías:**
- Clasificación de tipos de garantía en dos categorías principales:
  - `sin_garantia`: Sin garantía, Garantía de autenticidad/descripción, Garantía basada en reputación
  - `garantia_especifica`: Garantías oficiales, con plazo explícito, ilimitadas, etc.

**5. Predicción de títulos integrada:**
- Utiliza el modelo híbrido entrenado (`TitleProcessor`) para predecir condición desde títulos
- Agrega probabilidades de predicción como features adicionales (`probabilidad_new`, `probabilidad_used`)

**6. Balanceo de clases con SMOTE:**
- Aplica **SMOTENC** (SMOTE para features categóricas) para balancear la variable objetivo `condition`
- Maneja automáticamente las features categóricas binarias identificadas en el dataset
- Mejora significativamente el balance entre productos nuevos y usados

### Archivos generados en el procesamiento

El script `process_datasets.py` utiliza un patrón de factory (`DatasetProcessorFactory`) para procesar tanto datos de entrenamiento como de prueba:

**Entrada:**
- `datos_con_categoria.csv` (datos de entrenamiento con warranty categorizado)
- `datos_con_categoria_test.csv` (datos de prueba con warranty categorizado)

**Salida:**
- `training_data_features_selected.csv`: Dataset de entrenamiento completamente procesado
- `test_data_features_selected.csv`: Dataset de prueba procesado
- `data_clean.csv`: Dataset limpio antes del balanceo SMOTE

### Ejecución del procesamiento

Para ejecutar el procesamiento completo del dataset:

Desde la carpeta `repositories`:
```sh
uv run process_datasets.py
```

**Resultado:**
- Transforma ~50 columnas originales en más de 50 features procesadas
- Aplica balanceo SMOTE para mejorar el entrenamiento
- Genera datasets listos para entrenamiento de modelos de ML
- Incluye features de embeddings de títulos y características extraídas

### Flujo de datos completo

1. **Datos iniciales** → `x_train.csv`, `x_test.csv`
2. **Clasificación de garantías** → `datos_con_categoria.csv`, `datos_con_categoria_test.csv`
3. **Procesamiento inicial** → `process_datasets.py` → `training_data_processed.csv`, `test_data_processed.csv`
4. **Preprocesamiento avanzado** → `DatasetProcessor` → `training_data_features_selected.csv`, `test_data_features_selected.csv`
5. **Entrenamiento de modelos** → `model_train.py` → Modelos finales `.joblib` y métricas

Este procesamiento es fundamental para el éxito de los modelos de ML, ya que convierte datos crudos con campos JSON complejos en features estructuradas, balanceadas y optimizadas para algoritmos de aprendizaje automático.

---

## Preprocesador avanzado de datasets (DatasetProcessor)

Antes del entrenamiento de modelos, se aplica un preprocesador avanzado implementado en la clase `DatasetProcessor` que realiza la limpieza final de variables y extracción de características optimizadas para algoritmos de ML.

### Funcionalidades del DatasetProcessor

**1. Selección automática de características:**
- **SelectKBest**: Selección basada en puntuaciones F estadísticas
- **Feature importance de RandomForest**: Identifica variables más predictivas
- **Análisis de correlación**: Elimina características altamente correlacionadas
- **Filtrado de varianza**: Remueve características con baja variabilidad

**2. Reducción de dimensionalidad:**
- **PCA (Principal Component Analysis)**: Extrae componentes principales que explican 95% de la varianza
- **Escalado estándar**: Normalización de todas las features para algoritmos sensibles a escala
- **Preservación de información**: Mantiene el poder predictivo mientras reduce complejidad

**3. Análisis de calidad de datos:**
- Detección automática de valores faltantes
- Identificación de características redundantes
- Análisis de distribución de variables
- Reporte de correlaciones altas entre features

**4. Generación de datasets optimizados:**
- **Datasets con features seleccionadas**: Subconjunto óptimo de características originales
- **Datasets con PCA**: Versiones con dimensionalidad reducida
- **Múltiples configuraciones**: Permite experimentar con diferentes enfoques

### Archivos generados por DatasetProcessor

El preprocesador genera múltiples versiones del dataset:

**Features seleccionadas:**
- `training_data_features_selected.csv`: Dataset de entrenamiento con características optimizadas
- `test_data_features_selected.csv`: Dataset de prueba con las mismas características

**Versiones PCA (opcional):**
- `training_data_pca.csv`: Dataset de entrenamiento con PCA aplicado
- `test_data_pca.csv`: Dataset de prueba con PCA aplicado

**Metadatos:**
- `feature_selection_summary.json`: Reporte detallado del proceso de selección
- Información de varianza explicada, características eliminadas, correlaciones

### Ejecución del preprocesador

El DatasetProcessor se ejecuta a través de la función `generate_train_dataset()` en el archivo main:

```python
def generate_train_dataset():
    processor = DatasetProcessor(data_dir="../data", target_variance=0.95)
    summary = processor.run_full_analysis(
        generate_pca=True,
        generate_features=True
    )
```

**Configuraciones principales:**
- `target_variance=0.95`: PCA explica al menos 95% de la varianza
- `generate_pca=True`: Genera versiones con PCA
- `generate_features=True`: Genera versiones con feature selection

---

## Entrenamiento automático de modelos de Machine Learning

El último paso del pipeline es el entrenamiento de modelos de ML utilizando el script `model_train.py`, que implementa una clase `ModelTrainer` con funcionalidades avanzadas para entrenar, evaluar y seleccionar automáticamente el mejor modelo.

### Características principales del ModelTrainer

**1. Entrenamiento de múltiples modelos:**
- **RandomForest**: Clasificador de bosques aleatorios con hiperparámetros optimizados
- **GradientBoosting**: Clasificador de gradient boosting con configuración avanzada  
- **LightGBM**: Modelo de gradient boosting ligero y eficiente

**2. Optimización automática de hiperparámetros:**
- **GridSearchCV** con validación cruzada de 3 folds
- Búsqueda exhaustiva en grids de hiperparámetros predefinidos
- Modo rápido opcional para pruebas con hiperparámetros reducidos

**3. Selección inteligente del mejor modelo:**
- **Criterio principal**: Test Accuracy (precisión en conjunto de prueba)
- **Criterio secundario**: Test Recall (recall ponderado en conjunto de prueba)
- Selección automática basada en rendimiento combinado

**4. Guardado automático de modelos y métricas:**
- Cada modelo entrenado se guarda como `.joblib` individual
- El mejor modelo se guarda por separado como `best_model_overall.joblib`
- Archivo CSV detallado con todas las métricas de ejecución

### Métricas y evaluación

El sistema calcula y guarda las siguientes métricas para cada modelo:

- **CV Score**: Puntuación de validación cruzada durante entrenamiento
- **Test Accuracy**: Precisión en el conjunto de prueba
- **Test Recall**: Recall ponderado en el conjunto de prueba  
- **Tiempo de entrenamiento**: Duración del proceso de entrenamiento
- **Mejores hiperparámetros**: Configuración óptima encontrada
- **Total de fits**: Número total de entrenamientos realizados

### Archivos generados

El script genera automáticamente:

**Modelos individuales:**
- `RandomForest_best_model.joblib`
- `GradientBoosting_best_model.joblib`
- `LightGBM_best_model.joblib`

**Mejor modelo:**
- `best_model_overall.joblib`: El modelo con mejor rendimiento general

**Métricas detalladas:**
- `training_metrics.csv`: Tabla completa con todas las métricas, ordenada por accuracy y recall

### Ejecución del entrenamiento

Para ejecutar el entrenamiento completo de modelos:

Desde la carpeta `repositories`:
```sh
uv run model_train.py
```

**El script incluye:**
- Modo interactivo para elegir entrenamiento rápido o completo
- Progreso detallado con ETA para cada modelo
- Reportes completos de clasificación
- Top 3 mejores configuraciones para cada modelo
- Resumen final con recomendaciones específicas

### Manejo de errores y logging

- **Excepciones personalizadas**: `ModelTrainingException`, `DataLoadingException`, `DataSavingException`
- **Logging comprehensivo**: Seguimiento detallado de cada etapa del proceso
- **Recuperación de errores**: Continúa entrenamiento aunque fallen modelos individuales
- **Validación de datos**: Verificación automática de archivos de entrada

### Pipeline completo de ML

El flujo completo desde datos crudos hasta modelos listos para producción:

1. **Datos iniciales** → `x_train.csv`, `x_test.csv`
2. **Clasificación de garantías** → `warranty_handler.py` → `datos_con_categoria.csv`
3. **Feature engineering inicial** → `process_datasets.py` → `training_data_processed.csv`
4. **Preprocesamiento avanzado** → `DatasetProcessor` → `training_data_features_selected.csv`
5. **Entrenamiento de modelos** → `model_train.py` → Modelos `.joblib` + métricas
6. **Selección automática** → `best_model_overall.joblib` (listo para producción)

Este sistema automatizado garantiza la reproducibilidad, trazabilidad y selección objetiva del mejor modelo basado en métricas de rendimiento claras y criterios predefinidos.

---

## Archivo principal de orquestación (main.py)

El archivo `src/main.py` actúa como el orquestador principal de todo el pipeline de Machine Learning, proporcionando funciones organizadas para ejecutar cada etapa del proceso de manera controlada y secuencial.

### Funciones principales del main.py

**1. `create_data()`:**
- Carga el dataset original desde `MLA_100k.jsonlines`
- Realiza la división train/test de los datos
- Genera los archivos CSV iniciales: `x_train.csv`, `x_test.csv`, `y_test.csv`, `y_train.csv`
- Crea la estructura base de datos para el pipeline

**2. `process_warranty()`:**
- Ejecuta la clasificación automática de garantías usando `WarrantyClassifier`
- Procesa tanto datos de entrenamiento como de prueba
- Genera `datos_con_categoria.csv` y `datos_con_categoria_test.csv`
- Incluye manejo completo de excepciones y logging

**3. `process_explore_title()`:**
- Ejecuta el análisis automático de títulos usando `TitleAnalyzer`
- Genera insights y patrones de los títulos de productos
- Crea el archivo `title_analysis_results.json` con resultados detallados

**4. `process_train_title_model()`:**
- Entrena el modelo híbrido para predicción de títulos
- Utiliza embeddings y features extraídas del análisis de títulos
- Guarda los modelos entrenados: `modelo_final.joblib`, `modelo_scaler.joblib`, `modelo_encoder.joblib`

**5. `process_dataset()`:**
- Ejecuta el feature engineering inicial usando `process_datasets.py`
- Procesa campos JSON complejos y crea variables dummy
- Aplica SMOTE para balanceo de clases
- Genera `training_data_processed.csv` y `test_data_processed.csv`

**6. `generate_train_dataset()`:**
- Ejecuta el preprocesador avanzado `DatasetProcessor`
- Realiza selección de características y reducción de dimensionalidad
- Genera datasets optimizados: `training_data_features_selected.csv`
- Proporciona análisis detallado de calidad de features

**7. `train_model()`:**
- Ejecuta el entrenamiento completo de modelos ML usando `ModelTrainer`
- Entrena múltiples algoritmos con optimización de hiperparámetros
- Selecciona automáticamente el mejor modelo basado en accuracy y recall
- Genera todos los artefactos finales de modelos y métricas

### Ejecución del pipeline completo

El archivo main incluye una sección de ejecución que puede ejecutar todo el pipeline:

```python
if __name__ == "__main__":
    #create_data()           # 1. Crear datos iniciales
    #process_warranty()      # 2. Clasificar garantías
    #process_explore_title() # 3. Analizar títulos
    #process_train_title_model() # 4. Entrenar modelo de títulos
    process_dataset()       # 5. Feature engineering inicial
    generate_train_dataset() # 6. Preprocesamiento avanzado
    train_model()           # 7. Entrenar modelos finales
```

**Características del orquestador:**
- **Ejecución modular**: Cada función puede ejecutarse independientemente
- **Manejo de errores**: Logging y excepciones personalizadas en cada etapa
- **Flexibilidad**: Permite comentar/descomentar etapas según necesidades
- **Trazabilidad**: Logging detallado de cada paso del proceso
- **Validación**: Verificación de archivos de entrada y salida en cada etapa

### Uso recomendado

Para ejecutar el pipeline completo desde el inicio:
1. Descomentar todas las funciones en `main.py`
2. Ejecutar desde la carpeta `src`: `uv run main.py`
3. El sistema ejecutará automáticamente todas las etapas en secuencia

Para ejecutar etapas específicas:
- Comentar las funciones no deseadas
- Ejecutar solo las etapas necesarias
- Útil para desarrollo iterativo y debugging

El archivo main.py convierte un pipeline complejo de ML en un proceso ejecutable con un solo comando, manteniendo la flexibilidad para desarrollo y experimentación.

---

