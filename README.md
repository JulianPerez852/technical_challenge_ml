
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
Este flujo permite analizar la columna de garantía de manera distinta y obtener una clasificación automatizada usando modelos de lenguaje.
