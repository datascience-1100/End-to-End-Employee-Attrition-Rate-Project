import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
        params = yaml.safe_load(open(params_path, 'r'))
        decision_tree_max_depth= params['model_building']['decision_tree_max_depth']
        max_iter = params['model_building']['max_iter']
        logger.info("Successfully loaded parameters")
        return max_iter, decision_tree_max_depth
    
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

def train_logistic_regression_model(X, y, max_iter):
    try:
        logger.info("Training logistic regression model")
        clf = LogisticRegression(random_state=42, max_iter=max_iter).fit(X, y)        
        logger.info("Model training completed successfully")
        return clf
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
    
def train_decision_tree(X, y, decision_tree_max_depth):
    try:
        decision_tree_clf = DecisionTreeClassifier(max_depth=decision_tree_max_depth, random_state=42).fit(X, y)
        logger.info("Model decision training completed successfully")

        return decision_tree_clf
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def save_model(model, model_path):
    try:
        logger.info(f"Saving model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved successfully to {model}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def process_model_training():
    try:
        # Define file paths
        params_path = 'params.yaml'
        train_file_path = './data/featured/train_featured.csv'
        logistic_model_path = 'logistic_regression_model.pkl'
        decision_tree_model_path = 'decision_tree_model.pkl'
        
        # Load parameters
        max_iter, decision_tree_max_depth = load_params(params_path)

        # Load training data
        train_data = load_data(train_file_path)
        
        # Select the features and outcome variable
        X_train = train_data.drop(columns=['left'])
        y_train = train_data['left']
        
        # Train the model
        clf = train_logistic_regression_model(X_train, y_train, max_iter)
        decision_tree_clf = train_decision_tree(X_train, y_train, decision_tree_max_depth)
        
        # Save the trained model
        save_model(clf, logistic_model_path )
        save_model(decision_tree_clf, decision_tree_model_path)
        
    except Exception as e:
        logger.error(f"An error occurred during model training process: {e}")
        raise

if __name__ == "__main__":
    process_model_training()