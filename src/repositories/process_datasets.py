import pandas as pd
import logging
from pathlib import Path
from dataset_processor_factory import DatasetProcessorFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_training_data(input_path: str, output_path: str) -> None:
    """
    Process training dataset and save results
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed data
    """
    logger.info(f"Loading training data from {input_path}")
    data = pd.read_csv(input_path)
    
    # Create training processor using factory
    processor = DatasetProcessorFactory.create_processor('training')
    
    # Process the data
    processed_data = processor.process(data)
    
    # Save processed data
    processed_data.to_csv(output_path, index=False)
    logger.info(f"Training data saved to {output_path}")
    
    return processed_data


def process_test_data(input_path: str, output_path: str) -> None:
    """
    Process test dataset and save results
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed data
    """
    logger.info(f"Loading test data from {input_path}")
    data = pd.read_csv(input_path)
    
    # Create test processor using factory
    processor = DatasetProcessorFactory.create_processor('test')
    
    # Process the data
    processed_data = processor.process(data)
    
    # Save processed data
    processed_data.to_csv(output_path, index=False)
    logger.info(f"Test data saved to {output_path}")
    
    return processed_data


def main():
    """Main function to process both datasets"""
    try:
        # Define paths
        base_path = Path("../../data")
        
        # Process training data
        training_input = base_path / "datos_con_categoria.csv"
        training_output = base_path / "training_data_processed.csv"
        
        if training_input.exists():
            training_data = process_training_data(str(training_input), str(training_output))
            print(f"Training data shape: {training_data.shape}")
            print(f"Training data condition distribution:")
            print(training_data['condition'].value_counts())
        else:
            logger.warning(f"Training file not found: {training_input}")
        
        # Process test data
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


if __name__ == "__main__":
    main()