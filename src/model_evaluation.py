import numpy as np
import pandas as pd
import os
import json
import pickle
import logging
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

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
    models = load_models('models.pkl')
    clf = models['logistic_regression']
    decision_tree_clf = models['decision_tree']

def load_data(file_path):
    try:
        logger.info(f"Loading test data from {file_path}")
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        with mlflow.start_run():
            logger.info("Evaluating the model on the test set")
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log confusion matrix as an artifact
            conf_matrix = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10,7))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            
            # Save confusion matrix plot
            conf_matrix_path = "confusion_matrix.png"
            plt.savefig(conf_matrix_path)
            plt.close()  # Close the plot to avoid memory issues
            
            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_path = "classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
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
        logistic_model_path = 'logistic_regression_model.pkl'
        decision_tree_model_path ='decision_tree_model.pkl'
        test_file_path = './data/featured/test_featured.csv'
        
        # Load the model and test data
        clf = load_model(logistic_model_path)
        decision_tree_clf = load_model(decision_tree_model_path)

        test_data = load_data(test_file_path)
        
        # Select the features and outcome variable
        X_test = test_data.drop(columns=['left'])
        y_test = test_data['left']
        
        # Evaluate the model
        metrics = evaluate_model(clf, X_test, y_test)
        metrics1 = evaluate_model(decision_tree_clf , X_test, y_test)
       
        
        # Optionally, save evaluation metrics locally (if needed)
        metrics_file_path = 'metrics.json'
        with open(metrics_file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info("Metrics is created")

                
        metrics_file_path = 'metrics1.json'
        with open(metrics_file_path, 'w') as file:
            json.dump(metrics1, file, indent=4)  
        logger.info("Metrics1 is created")

        
    except Exception as e:
        logger.error(f"An error occurred during the model evaluation process: {e}")
        raise

if __name__ == "__main__":
    process_model_evaluation()
