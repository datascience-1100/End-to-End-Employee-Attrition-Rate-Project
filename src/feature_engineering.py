import numpy as np
import pandas as pd
import os
import logging

# Set up logging
logger = logging.getLogger('data_processing')
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_processing.log')
file_handler.setLevel(logging.ERROR)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error("File is not found")
        raise

def ohe(df):
    try:
        df = pd.get_dummies(df, prefix=['salary', 'dept'], columns=['salary', 'department'], drop_first=False)
        logger.info("Successfully applied one-hot encoding")
        return df
    except Exception as e:
        logger.error("Could not apply ohe")
        raise

def save_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise

def process_data():
    try:
        # Define file paths
        train_file_path = './data/processed/train_processed.csv'
        test_file_path = './data/processed/test_processed.csv'
        data_path = os.path.join("data", "featured")
        
        # Load data
        train_data = load_data(train_file_path)
        test_data = load_data(test_file_path)
        
        # Apply one-hot encoding
        train_featured_data = ohe(train_data)
        test_featured_data = ohe(test_data)
        
        # Create the directory if it does not exist
        os.makedirs(data_path, exist_ok=True)
        
        # Save processed data
        save_data(train_featured_data, os.path.join(data_path, 'train_featured.csv'))
        save_data(test_featured_data, os.path.join(data_path, 'test_featured.csv'))
        
    except Exception as e:
        logger.error(f"An error occurred during data processing: {e}")
        raise

if __name__ == "__main__":
    process_data()
