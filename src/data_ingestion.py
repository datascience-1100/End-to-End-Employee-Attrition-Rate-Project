import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

def load_params(params_path: str) -> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
        return test_size
    except FileNotFoundError:
        print(f"Error: The file '{params_path}' was not found.")
        raise
    except KeyError as e:
        print(f"Error: Key '{e}' not found in the YAML file.")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file. {e}")
        raise

# Path to the input CSV file
def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except FileNotFoundError:
        print(f"Error: The file '{url}' was not found.")
        raise
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{url}' is empty.")
        raise
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the CSV file '{url}'.")
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
        print(f"Error: Failed to save data to path '{data_path}'. {e}")
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
        print(f"An error occurred during the data processing pipeline: {e}")

if __name__ == "__main__":
    main()
