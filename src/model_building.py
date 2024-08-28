import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

os.environ["DAGSHUB_PAT"] = "2e48be432fcecb47f9c9b2133e2468fb685366fa"

dagshub_token = os.getenv("DAGSHUB_PAT")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "datascience-1100"
repo_name = "End-to-End-Employee-Attrition-Rate-Project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

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
        decision_tree_max_depth = params['model_building']['decision_tree_max_depth']
        max_iter = params['model_building']['max_iter']
        learning_rate = params['model_building']['learning_rate']
        max_depth = params['model_building']['max_depth']
        n_estimators = params['model_building']['n_estimators']
        svm_c = params['model_building']['svm_c']
        logger.info("Successfully loaded parameters")
        return max_iter, decision_tree_max_depth, learning_rate, max_depth, n_estimators, svm_c
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

def compare_models(X, y, logistic_param_grid, decision_tree_param_grid, xgboost_param_grid, svm_param_grid):
    try:
        # Logistic Regression Experiment
        mlflow.set_experiment('logistic_regression_experiment')
        with mlflow.start_run():
            logistic_model = GridSearchCV(LogisticRegression(random_state=42), param_grid=logistic_param_grid, cv=5, n_jobs=-1)
            logger.info("Performing cross-validation for Logistic Regression")
            logistic_scores = cross_val_score(logistic_model, X, y, cv=5, scoring='accuracy')
            logger.info(f"Logistic Regression Cross-Validation Accuracy: {logistic_scores.mean()} ± {logistic_scores.std()}")
            logistic_model.fit(X, y)

        # Decision Tree Experiment
        mlflow.set_experiment('decision_tree_experiment')
        with mlflow.start_run():
            decision_tree_model = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=decision_tree_param_grid, cv=5, n_jobs=-1)
            logger.info("Performing cross-validation for Decision Tree")
            decision_tree_scores = cross_val_score(decision_tree_model, X, y, cv=5, scoring='accuracy')
            logger.info(f"Decision Tree Cross-Validation Accuracy: {decision_tree_scores.mean()} ± {decision_tree_scores.std()}")
            decision_tree_model.fit(X, y)

        # XGBoost Experiment
        mlflow.set_experiment('xgboost_experiment')
        with mlflow.start_run():
            xgboost_model = GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), param_grid=xgboost_param_grid, cv=5, n_jobs=-1)
            logger.info("Performing cross-validation for XGBoost")
            xgboost_scores = cross_val_score(xgboost_model, X, y, cv=5, scoring='accuracy')
            logger.info(f"XGBoost Cross-Validation Accuracy: {xgboost_scores.mean()} ± {xgboost_scores.std()}")
            xgboost_model.fit(X, y)

        # SVM Experiment
        mlflow.set_experiment('svm_experiment')
        with mlflow.start_run():
            svm_model = GridSearchCV(SVC(random_state=42), param_grid=svm_param_grid, cv=5, n_jobs=-1)
            logger.info("Performing cross-validation for SVM")
            svm_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')
            logger.info(f"SVM Cross-Validation Accuracy: {svm_scores.mean()} ± {svm_scores.std()}")
            svm_model.fit(X, y)

        logger.info("Model comparison completed successfully")

        return (logistic_model.best_estimator_, decision_tree_model.best_estimator_, 
                xgboost_model.best_estimator_, svm_model.best_estimator_)

    except Exception as e:
        logger.error(f"Error comparing models: {e}")
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
        logistic_model_path = 'model/logistic_regression_model.pkl'
        decision_tree_model_path = 'model/decision_tree_model.pkl'
        xgboost_model_path = 'model/xgboost_model.pkl'
        svm_model_path = 'model/svm_model.pkl'
        
        # Load parameters
        max_iter, decision_tree_max_depth, learning_rate, max_depth, n_estimators, svm_c = load_params(params_path)

        # Load training data
        train_data = load_data(train_file_path)
        
        # Select the features and outcome variable
        X_train = train_data.drop(columns=['left'])
        y_train = train_data['left']

        # Define parameter grids for GridSearchCV
        logistic_param_grid = {'max_iter': [max_iter]}
        decision_tree_param_grid = {'max_depth': [decision_tree_max_depth]}
        xgboost_param_grid = {
            'learning_rate': [learning_rate],
            'max_depth': [max_depth],
            'n_estimators': [n_estimators]
        }
        svm_param_grid = {'C': [svm_c]}
        
        # Compare the models using cross-validation
        best_logistic_model, best_decision_tree_model, best_xgboost_model, best_svm_model = compare_models(
            X_train, y_train, logistic_param_grid, decision_tree_param_grid, xgboost_param_grid, svm_param_grid
        )
        
        # Save the trained models
        save_model(best_logistic_model, logistic_model_path)
        save_model(best_decision_tree_model, decision_tree_model_path)
        save_model(best_xgboost_model, xgboost_model_path)
        save_model(best_svm_model, svm_model_path)

    except Exception as e:
        logger.error(f"An error occurred during the model training process: {e}")
        raise

if __name__ == "__main__":
    process_model_training()
