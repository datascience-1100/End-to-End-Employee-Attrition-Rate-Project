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
        with mlflow.start_run(run_name="Model_Comparison"):
            # Define the models with GridSearchCV
            logistic_model = GridSearchCV(LogisticRegression(random_state=42), param_grid=logistic_param_grid, cv=5, n_jobs=-1)
            decision_tree_model = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=decision_tree_param_grid, cv=5, n_jobs=-1)
            xgboost_model = GridSearchCV(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), param_grid=xgboost_param_grid, cv=5, n_jobs=-1)
            svm_model = GridSearchCV(SVC(random_state=42), param_grid=svm_param_grid, cv=5, n_jobs=-1)
            
            # Perform cross-validation for Logistic Regression
            logger.info("Performing cross-validation for Logistic Regression")
            logistic_scores = cross_val_score(logistic_model, X, y, cv=5, scoring='accuracy')
            logger.info(f"Logistic Regression Cross-Validation Accuracy: {logistic_scores.mean()} ± {logistic_scores.std()}")

            # Perform cross-validation for Decision Tree
            logger.info("Performing cross-validation for Decision Tree")
            decision_tree_scores = cross_val_score(decision_tree_model, X, y, cv=5, scoring='accuracy')
            logger.info(f"Decision Tree Cross-Validation Accuracy: {decision_tree_scores.mean()} ± {decision_tree_scores.std()}")

            # Perform cross-validation for XGBoost
            logger.info("Performing cross-validation for XGBoost")
            xgboost_scores = cross_val_score(xgboost_model, X, y, cv=5, scoring='accuracy')
            logger.info(f"XGBoost Cross-Validation Accuracy: {xgboost_scores.mean()} ± {xgboost_scores.std()}")

            # Perform cross-validation for SVM
            logger.info("Performing cross-validation for SVM")
            svm_scores = cross_val_score(svm_model, X, y, cv=5, scoring='accuracy')
            logger.info(f"SVM Cross-Validation Accuracy: {svm_scores.mean()} ± {svm_scores.std()}")

            # Fit the models on the entire dataset
            logger.info("Fitting Logistic Regression model on entire dataset")
            logistic_model.fit(X, y)
            logger.info("Fitting Decision Tree model on entire dataset")
            decision_tree_model.fit(X, y)
            logger.info("Fitting XGBoost model on entire dataset")
            xgboost_model.fit(X, y)
            logger.info("Fitting SVM model on entire dataset")
            svm_model.fit(X, y)
            
            # Log the best parameters and scores for each model
            mlflow.log_param("logistic_best_params", logistic_model.best_params_)
            mlflow.log_param("decision_tree_best_params", decision_tree_model.best_params_)
            mlflow.log_param("xgboost_best_params", xgboost_model.best_params_)
            mlflow.log_param("svm_best_params", svm_model.best_params_)
            mlflow.log_metric("logistic_cv_accuracy", logistic_scores.mean())
            mlflow.log_metric("decision_tree_cv_accuracy", decision_tree_scores.mean())
            mlflow.log_metric("xgboost_cv_accuracy", xgboost_scores.mean())
            mlflow.log_metric("svm_cv_accuracy", svm_scores.mean())
            
            # Log the models
            mlflow.sklearn.log_model(logistic_model.best_estimator_, "logistic_regression_model")
            mlflow.sklearn.log_model(decision_tree_model.best_estimator_, "decision_tree_model")
            mlflow.sklearn.log_model(xgboost_model.best_estimator_, "xgboost_model")
            mlflow.sklearn.log_model(svm_model.best_estimator_, "svm_model")
            
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
        logistic_model_path = 'logistic_regression_model.pkl'
        decision_tree_model_path = 'decision_tree_model.pkl'
        xgboost_model_path = 'xgboost_model.pkl'
        svm_model_path = 'svm_model.pkl'
        
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
