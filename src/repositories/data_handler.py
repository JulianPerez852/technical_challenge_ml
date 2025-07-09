"""
    This repository its used to handle data operations such as loading, preprocessing, and saving datasets.
"""
import json
import pandas as pd
from exceptions.exceptions import DataLoadingException

class DataHandler:
    def __init__(self, path: str):
        self.path = path
        
        
    def build_dataset(self, n: int = -10000) -> tuple:
        """
            Load the dataset from a JSON lines file and split it into training and testing sets.
            Args:
                n (int): The number of samples to include in the training set. If n is negative, it will take the last -n samples for testing.
            Returns:
                tuple: A tuple containing the training features (X_train), training labels (y_train),
                    testing features (X_test), and testing labels (y_test).
        """
        try:
            data = [json.loads(x) for x in open(self.path)]
            target = lambda x: x.get("condition")
            N = n
            x_train = data[:N]
            x_test = data[N:]
            y_train = [target(x) for x in x_train]
            y_test = [target(x) for x in x_test]
            for x in x_test:
                del x["condition"]
            return x_train, y_train, x_test, y_test
        except Exception as e:
            raise DataLoadingException(f"Failed to load dataset from {path}: {str(e)}") from e
    
    def convert_to_dataframe(self, x_train: list, y_train: list, x_test: list, y_test:list) -> tuple:
        """
            Convert the training and testing data into pandas DataFrames.
            Args:
                x_train (list): The training features.
                y_train (list): The training labels.
                x_test (list): The testing features.
                y_test (list): The testing labels.
            Returns:
                tuple: A tuple containing the training DataFrame (X_train_df), training labels Series (y_train_df),
                    testing DataFrame (X_test_df), and testing labels Series (y_test_df).
        """
        x_train_df = pd.DataFrame(x_train)
        y_train_df = pd.Series(y_train)
        x_test_df = pd.DataFrame(x_test)
        y_test_df = pd.Series(y_test)
        return x_train_df, y_train_df, x_test_df, y_test_df
