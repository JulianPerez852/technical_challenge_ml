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

