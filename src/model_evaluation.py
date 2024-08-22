import numpy as np
import pandas as pd
import os
import json
import pickle
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Set up logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.ERROR)

# Create formatter and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path):
    try:
        logger.info(f"Loading model from {model_path}")
        model = pickle.load(open(model_path, 'rb'))
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def load_data(file_path):
    try:
        logger.info(f"Loading test data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info("Test data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        logger.info("Evaluating the model on the test set")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        logger.info("Model evaluation completed successfully")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def save_metrics(metrics, file_path):
    try:
        logger.info(f"Saving evaluation metrics to {file_path}")
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info(f"Metrics saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics to {file_path}: {e}")
        raise

def process_model_evaluation():
    try:
        # Define file paths
        model_path = 'model.pkl'
        test_file_path = './data/featured/test_featured.csv'
        metrics_file_path = 'metrics.json'
        
        # Load the model and test data
        clf = load_model(model_path)
        test_data = load_data(test_file_path)
        
        # Select the features and outcome variable
        X_test = test_data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure', 
                            'work_accident', 'promotion_last_5years', 'salary_high', 'salary_low', 'salary_medium',
                            'dept_IT', 'dept_RandD', 'dept_accounting', 'dept_hr', 'dept_management', 'dept_marketing',
                            'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical']]
        y_test = test_data['left']
        
        # Evaluate the model
        metrics = evaluate_model(clf, X_test, y_test)
        
        # Save evaluation metrics
        save_metrics(metrics, metrics_file_path)
        
    except Exception as e:
        logger.error(f"An error occurred during the model evaluation process: {e}")
        raise

if __name__ == "__main__":
    process_model_evaluation()
