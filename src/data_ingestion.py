import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

import logging

logger= logging. getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s' )
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            logger.debug("test_size received")
        return test_size
    except FileNotFoundError:
        logger.error('File is Not Found')
        raise
    except KeyError as e:
        logger.error('keyerror')
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error')
        raise

# Path to the input CSV file
def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except FileNotFoundError:
        logger.error('File is not found')
        raise

# Define the path for saving the processed data
def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        # Create the directory if it does not exist
        os.makedirs(data_path, exist_ok=True)
        # Save the training and testing data to CSV files
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    except IOError as e:
        logger.error('Input-Output Error')
        raise

def main():
    try:
        test_size = load_params('params.yaml')
        df = read_data("c:/Users/Subha/OneDrive/Desktop/HR_capstone_dataset.csv")
        # Split the data into training and testing sets
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
    except Exception as e:
        logger.error('File unavailable')
        
if __name__ == "__main__":
    main()
