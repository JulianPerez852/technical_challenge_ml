from repositories.data_handler import DataHandler
from repositories.warranty_handler import WarrantyClassifier
from repositories.title_explorer import TitleAnalyzer
from repositories.title_model_trainer import *
from repositories.process_datasets import process_training_data, process_test_data
from repositories.dataset_processor import DatasetProcessor
from repositories.model_train import *
from exceptions.exceptions import *
from pathlib import Path
import logging
import joblib
from tqdm import tqdm
import pandas as pd
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_data():
    data_handler = DataHandler("../data/MLA_100k.jsonlines")

    pack_data = data_handler.build_dataset(n=-10000)
    x_train_df, y_train_df, x_test_df, y_test_df = data_handler.convert_to_dataframe(*pack_data)

    if not os.path.exists("../data"):
        os.makedirs("../data")

    x_train_df.to_csv("../data/x_train.csv", index=False)
    x_test_df.to_csv("../data/x_test.csv", index=False)
    y_test_df.to_csv("../data/y_test.csv", index=False)
    x_train_df.to_csv("../data/y_train.csv", index=False)

def process_warranty():
    try:
        input_path = '../data/x_test.csv'
        output_path = '../data/datos_con_categoria_test.csv'
        classifier = WarrantyClassifier(input_path, output_path, batch_size=32)
        classifier.load_data()
        classifier.classify()
        classifier.save()
    except (DataLoadingException, ClassificationException, DataSavingException) as e:
        logger.error(f"Error en el procesamiento de x_test: {e}")

    try:
        input_path = '../data/x_train.csv'
        output_path = '../data/datos_con_categoria.csv'
        classifier = WarrantyClassifier(input_path, output_path, batch_size=32)
        classifier.load_data()
        classifier.classify()
        classifier.save()
    except (DataLoadingException, ClassificationException, DataSavingException) as e:
        logger.error(f"Error en el procesamiento de x_train: {e}")

def process_explore_title():
    try:
        input_path = '../data/x_train.csv'
        output_path = '../data/title_analysis_results.json'
        
        analyzer = TitleAnalyzer(input_path, output_path)
        analyzer.load_data()
        results = analyzer.run_complete_analysis()
        analyzer.save_results(results)
        
    except (DataLoadingException, DataAnalysisException, DataSavingException) as e:
        logger.error(f"Error in title analysis: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def process_train_title_model():
    try:
        logger.info("Iniciando script principal de entrenamiento")
        
        input_path = '../data/x_train.csv'
        logger.info(f"Cargando dataset desde: {input_path}")
        
        df = pd.read_csv(input_path)
        logger.info(f"Dataset cargado: {len(df)} registros")
        
        modelo_entrenado = entrenar_modelo_completo(df)
        ejemplo_prediccion(modelo_entrenado)
        
        output_dir = '../data'
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

def process_dataset():
    try:
        base_path = Path("../data")
        
        training_input = base_path / "datos_con_categoria.csv"
        training_output = base_path / "training_data_processed.csv"
        
        if training_input.exists():
            training_data = process_training_data(str(training_input), str(training_output))
            print(f"Training data shape: {training_data.shape}")
            print(f"Training data condition distribution:")
            print(training_data['condition'].value_counts())
        else:
            logger.warning(f"Training file not found: {training_input}")
        
    # Procesamiento de la data de test
        test_input = base_path / "datos_con_categoria_test.csv"
        test_output = base_path / "test_data_processed.csv"
        
        if test_input.exists():
            test_data = process_test_data(str(test_input), str(test_output))
            print(f"Test data shape: {test_data.shape}")
        else:
            logger.warning(f"Test file not found: {test_input}")
            
    except Exception as e:
        logger.error(f"Error processing datasets: {e}")
        raise

def generate_train_dataset():
    try:
        processor = DatasetProcessor(data_dir="../data", target_variance=0.95)
        
        # Run full analysis
        summary = processor.run_full_analysis(
            generate_pca=True,
            generate_features=True
        )
        
        print("\n" + "="*60)
        print("DATASET PROCESSING COMPLETE")
        print("="*60)
        print(f"Original features: {summary['original_features']}")
        print(f"Selected features: {summary['selected_features']}")
        print(f"Feature reduction: {summary['feature_reduction_pct']:.1f}%")
        print(f"High correlation pairs: {summary['high_correlation_pairs']}")
        
        if 'pca_components' in summary:
            print(f"PCA components: {summary['pca_components']}")
            print(f"PCA variance explained: {summary['pca_variance_explained']:.4f}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

def train_model():
    try:
        # Crear instancia del trainer
        trainer = ModelTrainer(
            data_train_path='../data/training_data_features_selected.csv',
            data_test_path='../data/test_data_features_selected.csv',
            y_test_path='../data/y_test.csv',
            models_output_dir='../data/',
            metrics_output_path='../data/training_metrics.csv'
        )
        
        # Ejecutar pipeline completo
        best_models, results, best_name, best_model = trainer.run_full_pipeline()

        print(f"Mejor modelo: {best_name}")
        print(f"Total de modelos entrenados: {len(best_models)}")
        
        return best_models, results, best_name, best_model
        
    except (DataLoadingException, ModelTrainingException, DataSavingException) as e:
        logger.error(f"Error en el pipeline de ML: {e}")
        raise
    except Exception as e:
        logger.error(f"Error inesperado en el pipeline de ML: {e}")
        raise ModelTrainingException(f"Error inesperado en el pipeline de ML: {e}") from e

if __name__ == "__main__":
    #create_data()
    #process_warranty()
    #process_explore_title()
    # process_train_title_model()
    process_dataset()
    generate_train_dataset()
    train_model()
