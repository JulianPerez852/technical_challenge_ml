import pandas as pd
from ollama import Client
import json
import re
from collections import Counter
import logging
import os
import sys
############ Esta linea se agrega si y solo si se esta ejecutando el script desde la carpeta de repositories, si es desde otro lado el import se debe manejar distinto.
############ Es probable que si quiero hacer pruebas unitarias deba cambiar este script.
try:
    from exceptions.exceptions import DataLoadingException, DataSavingException, OllamaAnalysisException
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from exceptions.exceptions import DataLoadingException, DataSavingException, OllamaAnalysisException
############


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OllamaTitleAnalyzer:
    def __init__(self, input_path, ollama_host, ollama_model):
        self.input_path = input_path
        self.ollama_host = ollama_host
        self.ollama_model = ollama_model
        self.client = Client(host=self.ollama_host)
        self.df = None
        self.results = {
            'keywords': [],
            'categories': [],
            'brands': [],
            'patterns': [],
            'product_types': [],
            'analysis_details': []
        }

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info(f"DataFrame cargado exitosamente desde {self.input_path}.")
        except FileNotFoundError:
            logger.warning(f"Archivo '{self.input_path}' no encontrado. Creando un DataFrame de ejemplo.")
            data = {
                'title': [
                    'Apple iPhone 15 Pro Max 256GB - New, Sealed',
                    'Sony PlayStation 5 Disc Edition - Used, Good Condition',
                    'Samsung Galaxy S24 Ultra 512GB - Brand New',
                    'Used Nintendo Switch OLED with 3 Games',
                    'Dell XPS 15 Laptop i7 16GB RAM 512GB SSD Refurbished',
                    'Bose QuietComfort 45 Headphones - New, Unopened Box',
                    'Used Canon EOS R5 Mirrorless Camera Body',
                    'Logitech MX Master 3S Mouse for Mac',
                    'Apple Watch Series 9 GPS + Cellular 45mm Midnight (Open Box)',
                    'Microsoft Surface Pro 9 i5 8GB RAM 256GB SSD (Used)'
                ],
                'condition': [
                    'new', 'used', 'new', 'used', 'new',
                    'new', 'used', 'new', 'new', 'used'
                ]
            }
            self.df = pd.DataFrame(data)
            logger.info("DataFrame de ejemplo creado, por si falla el load, para pruebas unitarias.")
        except Exception as e:
            logger.error(f"Error al cargar los datos: {e}")
            raise DataLoadingException(f"Error al cargar los datos: {e}")

    def analyze_title_with_ollama(self, title, condition):
        prompt = f"""
        Analiza el siguiente título de producto y su condición.
        Título: "{title}"
        Condición: "{condition}""" + """"

        Extrae la siguiente información en formato JSON. Si no encuentras una categoría o marca específica, usa "desconocido".
        {
            "keywords": ["palabra_clave1", "palabra_clave2", ...],
            "category": "categoría_principal_del_producto",
            "brand": "nombre_de_la_marca",
            "product_type": "tipo_específico_de_producto_ej_smartphone_consola_laptop",
            "condition_detail": "detalle_de_la_condicion_ej_sealed_good_condition_refurbished",
            "potential_patterns": ["patrón_identificado_ej_modelo_capacidad"]
        }
        """
        try:
            response = self.client.generate(model=self.ollama_model, prompt=prompt, format='json')
            try:
                parsed_response = json.loads(response['response'])
            except (json.JSONDecodeError, KeyError):
                json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
                if json_match:
                    parsed_response = json.loads(json_match.group(0))
                else:
                    logger.warning(f"No se pudo encontrar un JSON válido en la respuesta para '{title}'. Respuesta completa: {response['response']}")
                    return None
            return parsed_response
        except Exception as e:
            logger.error(f"Error al llamar a Ollama para el título '{title}': {e}")
            raise OllamaAnalysisException(f"Error al analizar el título '{title}': {e}")

    def process_titles(self):
        logger.info("Iniciando el procesamiento de títulos. Esto puede tomar un tiempo considerable...")
        for index, row in self.df.iterrows():
            title = row['title']
            condition = row['condition']
            logger.info(f"Procesando título {index+1}/{len(self.df)}: '{title}'")
            try:
                analysis = self.analyze_title_with_ollama(title, condition)
            except OllamaAnalysisException as e:
                logger.error(str(e))
                analysis = None
            if analysis:
                self.results['analysis_details'].append(analysis)
                if 'keywords' in analysis and isinstance(analysis['keywords'], list):
                    self.results['keywords'].extend(analysis['keywords'])
                if 'category' in analysis and analysis['category'] not in self.results['categories']:
                    self.results['categories'].append(analysis['category'])
                if 'brand' in analysis and analysis['brand'] not in self.results['brands']:
                    self.results['brands'].append(analysis['brand'])
                if 'product_type' in analysis and analysis['product_type'] not in self.results['product_types']:
                    self.results['product_types'].append(analysis['product_type'])
                if 'potential_patterns' in analysis and isinstance(analysis['potential_patterns'], list):
                    self.results['patterns'].extend(analysis['potential_patterns'])
            else:
                logger.warning(f"No se pudo analizar el título: '{title}'")
        logger.info("Análisis de títulos completado.")

    def consolidate_and_save_results(self):
        logger.info("Consolidando y guardando resultados...")
        self.results['keywords'] = list(set([k.lower() for k in self.results['keywords'] if k and k.lower() != 'desconocido']))
        self.results['categories'] = list(set([c.lower() for c in self.results['categories'] if c and c.lower() != 'desconocido']))
        self.results['brands'] = list(set([b.lower() for b in self.results['brands'] if b and b.lower() != 'desconocido']))
        self.results['product_types'] = list(set([pt.lower() for pt in self.results['product_types'] if pt and pt.lower() != 'desconocido']))
        self.results['patterns'] = list(set([p.lower() for p in self.results['patterns'] if p and p.lower() != 'desconocido']))

        keyword_counts = Counter(self.results['keywords'])
        category_counts = Counter([detail['category'] for detail in self.results['analysis_details'] if 'category' in detail])
        brand_counts = Counter([detail['brand'] for detail in self.results['analysis_details'] if 'brand' in detail])
        product_type_counts = Counter([detail['product_type'] for detail in self.results['analysis_details'] if 'product_type' in detail])

        try:
            with open('extracted_keywords.json', 'w', encoding='utf-8') as f:
                json.dump(dict(keyword_counts.most_common(50)), f, ensure_ascii=False, indent=4)
            logger.info("Palabras clave guardadas en 'extracted_keywords.json'")

            with open('extracted_categories.json', 'w', encoding='utf-8') as f:
                json.dump(dict(category_counts.most_common()), f, ensure_ascii=False, indent=4)
            logger.info("Categorías guardadas en 'extracted_categories.json'")

            with open('extracted_brands.json', 'w', encoding='utf-8') as f:
                json.dump(dict(brand_counts.most_common()), f, ensure_ascii=False, indent=4)
            logger.info("Marcas guardadas en 'extracted_brands.json'")

            with open('extracted_product_types.json', 'w', encoding='utf-8') as f:
                json.dump(dict(product_type_counts.most_common()), f, ensure_ascii=False, indent=4)
            logger.info("Tipos de producto guardados en 'extracted_product_types.json'")

            with open('extracted_patterns.json', 'w', encoding='utf-8') as f:
                json.dump(self.results['patterns'], f, ensure_ascii=False, indent=4)
            logger.info("Patrones identificados guardados en 'extracted_patterns.json'")

            with open('full_analysis_details.json', 'w', encoding='utf-8') as f:
                json.dump(self.results['analysis_details'], f, ensure_ascii=False, indent=4)
            logger.info("Detalles completos del análisis guardados en 'full_analysis_details.json'")
        except Exception as e:
            logger.error(f"Error al guardar los resultados: {e}")
            raise DataSavingException(f"Error al guardar los resultados: {e}")

if __name__ == "__main__":
    OLLAMA_HOST = 'http://localhost:11434'
    OLLAMA_MODEL = 'gemma3n:e4b'
    INPUT_PATH = '../../data/x_train.csv'
    try:
        analyzer = OllamaTitleAnalyzer(INPUT_PATH, OLLAMA_HOST, OLLAMA_MODEL)
        analyzer.load_data()
        analyzer.process_titles()
        analyzer.consolidate_and_save_results()
        logger.info("\nAnálisis finalizado y resultados guardados. ¡Estás listo para el siguiente paso!")
    except (DataLoadingException, OllamaAnalysisException, DataSavingException) as e:
        logger.error(f"Error en el procesamiento: {e}")