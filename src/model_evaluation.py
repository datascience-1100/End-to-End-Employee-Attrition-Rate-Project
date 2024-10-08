import numpy as np
import pandas as pd
import json
import pickle
import logging
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
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

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")
    else:
        logger.info(f"Directory already exists: {directory_path}")

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
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def evaluate_model(model, X_test, y_test, model_name, experiment_name):
    try:
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            logger.info(f"Evaluating the {model_name} model on the test set")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
                      
            # Log confusion matrix as an artifact
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10,7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.title(f"{model_name} Confusion Matrix")
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')

            # Ensure directory exists
            conf_matrix_dir = "outputs/confusion_matrix"
            create_directory(conf_matrix_dir)
            
            # Save confusion matrix plot
            conf_matrix_path = f"{conf_matrix_dir}/{model_name}_confusion_matrix.png"
            plt.savefig(conf_matrix_path)
            
            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Ensure directory exists
            report_dir = "outputs/classification_report"
            create_directory(report_dir)
            
            report_path = f"{report_dir}/{model_name}_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=3)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

def process_model_evaluation():
    try:
        # Define file paths
        logistic_model_path = 'model/logistic_regression_model.pkl'
        decision_tree_model_path = 'model/decision_tree_model.pkl'
        xgboost_model_path = 'model/xgboost_model.pkl'
        svm_model_path = 'model/svm_model.pkl'
        test_file_path = './data/featured/test_featured.csv'
        
        # Load the models and test data
        clf = load_model(logistic_model_path)
        decision_tree_clf = load_model(decision_tree_model_path)
        xgboost_clf = load_model(xgboost_model_path)
        svm_clf = load_model(svm_model_path)

        test_data = load_data(test_file_path)
        
        # Select the features and outcome variable
        X_test = test_data.drop(columns=['left'])
        y_test = test_data['left']
        
        # Evaluate the models
        metrics_logistic = evaluate_model(clf, X_test, y_test, 'Logistic_Regression', 'Logistic_Regression_Evaluation')
        metrics_decision_tree = evaluate_model(decision_tree_clf, X_test, y_test, 'Decision_Tree', 'Decision_Tree_Evaluation')
        metrics_xgboost = evaluate_model(xgboost_clf, X_test, y_test, 'XGBoost', 'XGBoost_Evaluation')
        metrics_svm = evaluate_model(svm_clf, X_test, y_test, 'SVM', 'SVM_Evaluation')
        
        # Save evaluation metrics locally
        metrics_dir = "outputs"
        create_directory(metrics_dir)
        
        metrics_file_path_logistic = f'{metrics_dir}/metrics_logistic.json'
        with open(metrics_file_path_logistic, 'w') as file:
            json.dump(metrics_logistic, file, indent=4)
        logger.info("Logistic Regression metrics saved")

        metrics_file_path_decision_tree = f'{metrics_dir}/metrics_decision_tree.json'
        with open(metrics_file_path_decision_tree, 'w') as file:
            json.dump(metrics_decision_tree, file, indent=4)  
        logger.info("Decision Tree metrics saved")

        metrics_file_path_xgboost = f'{metrics_dir}/metrics_xgboost.json'
        with open(metrics_file_path_xgboost, 'w') as file:
            json.dump(metrics_xgboost, file, indent=4)  
        logger.info("XGBoost metrics saved")

        metrics_file_path_svm = f'{metrics_dir}/metrics_svm.json'
        with open(metrics_file_path_svm, 'w') as file:
            json.dump(metrics_svm, file, indent=4)  
        logger.info("SVM metrics saved")

    except Exception as e:
        logger.error(f"An error occurred during the model evaluation process: {e}")
        raise

if __name__ == "__main__":
    process_model_evaluation()
