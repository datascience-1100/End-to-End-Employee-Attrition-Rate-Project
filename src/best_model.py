import pickle
import json
import logging
import mlflow
from mlflow import log_metric, log_param, log_artifact

# mlflow with Dagshub
import dagshub
dagshub.init(repo_owner='datascience-1100', repo_name='End-to-End-Employee-Attrition-Rate-Project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/datascience-1100/End-to-End-Employee-Attrition-Rate-Project.mlflow")

# Set up logging
logger = logging.getLogger('best_model')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('best_model.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_metrics(file_path):
    try:
        with open(file_path, 'r') as file:
            metrics = json.load(file)
        return metrics
    except Exception as e:
        logger.error(f"Error loading metrics from {file_path}: {e}")
        raise

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def save_best_model(best_model_name, run_id):
    model_path = f"model/{best_model_name.lower()}_model.pkl"
    best_model_path = "best_model.pkl"
    
    try:
        best_model = load_model(model_path)
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"Best model saved as {best_model_path}")
        
        # Log the best model artifact to MLflow
        mlflow.log_artifact(best_model_path)
                
    except Exception as e:
        logger.error(f"Error saving the best model: {e}")
        raise

def save_run_id(run_id, file_path="run_id.json"):
    try:
        with open(file_path, 'w') as f:
            json.dump({"run_id": run_id}, f)
        logger.info(f"Run ID {run_id} saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving run ID to {file_path}: {e}")
        raise

def process_best_model_selection():
    try:
        # Start an MLflow run
        with mlflow.start_run(run_name="Best_Model_Selection") as run:
            run_id = run.info.run_id
            
            # Save the run_id for later use
            save_run_id(run_id)
            
            # Load evaluation metrics
            metrics_logistic = load_metrics('outputs/metrics_logistic.json')
            metrics_decision_tree = load_metrics('outputs/metrics_decision_tree.json')
            metrics_xgboost = load_metrics('outputs/metrics_xgboost.json')
            metrics_svm = load_metrics('outputs/metrics_svm.json')
            logger.info("Evaluation metrics loaded")
            
            # Determine the best model based on F1 score
            models_metrics = {
                'Logistic_Regression': metrics_logistic,
                'Decision_Tree': metrics_decision_tree,
                'XGBoost': metrics_xgboost,
                'SVM': metrics_svm
            }
            logger.info('Model metrics loaded')
            
            best_model_name = max(models_metrics, key=lambda k: models_metrics[k]['f1_score'])    
            best_model_f1_score = models_metrics[best_model_name]['f1_score']
            logger.info(f"Best model is {best_model_name} with F1 score: {best_model_f1_score}")
            
            # Log the metrics and the name of the best model to MLflow
            log_param("best_model", best_model_name)
            log_metric("best_model_f1_score", best_model_f1_score)
            
            # Save the best model and register it using the run_id
            save_best_model(best_model_name, run_id)

    except Exception as e:
        logger.error(f"An error occurred during the best model selection process: {e}")
        raise

if __name__ == "__main__":
    process_best_model_selection()
