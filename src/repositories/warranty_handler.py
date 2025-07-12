import pandas as pd
import torch
from transformers import pipeline
from tqdm.auto import tqdm
import os
import logging
import sys
import os
############ Esta linea se agrega si y solo si se esta ejecutando el script desde la carpeta de repositories, si es desde otro lado el import se debe manejar distinto.
############ Es probable que si quiero hacer pruebas unitarias deba cambiar este script.
try:
    from exceptions.exceptions import DataLoadingException, ClassificationException, DataSavingException
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from exceptions.exceptions import DataLoadingException, ClassificationException, DataSavingException
############



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

torch.set_num_threads(os.cpu_count())

class WarrantyClassifier:
    def __init__(self, input_path, output_path, batch_size=32):
        self.input_path = input_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.labels = [
            "Garantía oficial del fabricante / importador",
            "Garantía por defecto de fabricación",
            "Garantía con plazo explícito",
            "Garantía ilimitada / de por vida",
            "Garantía de satisfacción / devolución de dinero",
            "Garantía basada en reputación del vendedor",
            "Garantía de autenticidad / descripción",
            "Garantía de inspección en el punto de entrega",
            "Sin garantía"
        ]

        ######Estas clases las definí haciendo una revisión de variio del contenido de la columna warranty en el dataset,
        ######Las agrupé por lo que consideré util. 

        self._configure_threads()
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"{'GPU detectada' if self.device == 0 else 'No se detectó GPU'}. Se utilizará {'GPU' if self.device == 0 else 'CPU'} para la clasificación.")
        try:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=self.device,
                batch_size=self.batch_size
            )
            logger.info(f"Pipeline de clasificación inicializado con batch_size={self.batch_size}.")
        except Exception as e:
            logger.error("Error al inicializar el pipeline de clasificación.")
            raise ClassificationException("Error al inicializar el pipeline de clasificación.") from e

    def _configure_threads(self):
        os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
        os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
        torch.set_num_threads(os.cpu_count())
        logger.info(f"Configurados {os.cpu_count()} hilos de CPU para PyTorch y OpenMP/MKL.")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"DataFrame cargado con {len(self.df)} filas desde {self.input_path}.")
            self.df['warranty'] = self.df['warranty'].fillna('').astype(str)
        except Exception as e:
            logger.error(f"Error al cargar el archivo {self.input_path}.")
            raise DataLoadingException(f"Error al cargar el archivo {self.input_path}.") from e

    def classify(self):
        try:
            categories = []
            texts_to_classify = []
            for text in self.df['warranty']:
                if text.strip() == "":
                    texts_to_classify.append(None)
                else:
                    texts_to_classify.append(text)
            actual_texts_to_classify = [text for text in texts_to_classify if text is not None]
            results = []
            if actual_texts_to_classify:
                logger.info(f"Clasificando {len(actual_texts_to_classify)} textos con el modelo...")
                for i in tqdm(range(0, len(actual_texts_to_classify), self.batch_size), desc="Clasificando garantías por lotes", unit="batch"):
                    batch = actual_texts_to_classify[i:i + self.batch_size]
                    batch_results = self.classifier(batch, candidate_labels=self.labels)
                    results.extend(batch_results)
            else:
                logger.info("No hay textos para clasificar con el modelo.")
            result_idx = 0
            for text_or_none in texts_to_classify:
                if text_or_none is None:
                    categories.append("Sin garantía")
                else:
                    categories.append(results[result_idx]['labels'][0])
                    result_idx += 1
            self.df['warranty_category'] = categories
            logger.info("Clasificación completada y columna 'warranty_category' agregada.")
        except Exception as e:
            logger.error("Error durante la clasificación de garantías.")
            raise ClassificationException("Error durante la clasificación de garantías.") from e

    def save(self):
        try:
            self.df.to_csv(self.output_path, index=False)
            logger.info(f"¡Proceso completado! Archivo guardado como '{self.output_path}'")
        except Exception as e:
            logger.error(f"Error al guardar el archivo {self.output_path}.")
            raise DataSavingException(f"Error al guardar el archivo {self.output_path}.") from e

if __name__ == "__main__":
    try:
        input_path = '../../data/x_test.csv'
        output_path = '../../data/datos_con_categoria_test.csv'
        classifier = WarrantyClassifier(input_path, output_path, batch_size=32)
        classifier.load_data()
        classifier.classify()
        classifier.save()
    except (DataLoadingException, ClassificationException, DataSavingException) as e:
        logger.error(f"Error en el procesamiento de x_test: {e}")

    try:
        input_path = '../../data/x_train.csv'
        output_path = '../../data/datos_con_categoria.csv'
        classifier = WarrantyClassifier(input_path, output_path, batch_size=32)
        classifier.load_data()
        classifier.classify()
        classifier.save()
    except (DataLoadingException, ClassificationException, DataSavingException) as e:
        logger.error(f"Error en el procesamiento de x_train: {e}")