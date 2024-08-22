import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
import logging

# Set up logging
logger = logging.getLogger('model_training')
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_training.log')
file_handler.setLevel(logging.ERROR)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path):
    try:
        logger.info(f"Loading parameters from {params_path}")
        params = yaml.safe_load(open(params_path, 'r'))
        max_iter = params['model_building']['max_iter']
        logger.info("Successfully loaded parameters")
        return max_iter
    except Exception as e:
        logger.error(f"Error loading parameters: {e}")
        raise

def load_data(file_path):
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def train_model(X, y, max_iter):
    try:
        logger.info("Training logistic regression model")
        clf = LogisticRegression(random_state=42, max_iter=max_iter).fit(X, y)
        logger.info("Model training completed successfully")
        return clf
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def save_model(model, model_path):
    try:
        logger.info(f"Saving model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def process_model_training():
    try:
        # Define file paths
        params_path = 'params.yaml'
        train_file_path = './data/featured/train_featured.csv'
        model_path = 'model.pkl'

        # Load parameters
        max_iter = load_params(params_path)
        
        # Load training data
        train_data = load_data(train_file_path)
        
        # Select the features and outcome variable
        X_train = train_data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure', 
                              'work_accident', 'promotion_last_5years', 'salary_high', 'salary_low', 'salary_medium',
                              'dept_IT', 'dept_RandD', 'dept_accounting', 'dept_hr', 'dept_management', 'dept_marketing',
                              'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical']]
        y_train = train_data['left']
        
        # Train the model
        clf = train_model(X_train, y_train, max_iter)
        
        # Save the trained model
        save_model(clf, model_path)
        
    except Exception as e:
        logger.error(f"An error occurred during model training process: {e}")
        raise

if __name__ == "__main__":
    process_model_training()
