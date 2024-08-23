import pickle
import json
import logging

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

def save_best_model(best_model_name):
    model_path = f"model/{best_model_name.lower()}_model.pkl"
    best_model_path = "best_model.pkl"
    
    try:
        best_model = load_model(model_path)
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)
        logger.info(f"Best model saved as {best_model_path}")
    except Exception as e:
        logger.error(f"Error saving the best model: {e}")
        raise

def process_best_model_selection():
    try:
        # Load evaluation metrics
        metrics_logistic = load_metrics('outputs/metrics_logistic.json')
        metrics_decision_tree = load_metrics('outputs/metrics_decision_tree.json')
        metrics_xgboost = load_metrics('outputs/metrics_xgboost.json')
        metrics_svm = load_metrics('outputs/metrics_svm.json')
        logger.info("evaluation metrics loaded")
        
        # Determine the best model based on F1 score
        models_metrics = {
            'Logistic_Regression': metrics_logistic,
            'Decision_Tree': metrics_decision_tree,
            'XGBoost': metrics_xgboost,
            'SVM': metrics_svm
        }
        logger.info('model_metrics loaded')
        best_model_name = max(models_metrics, key=lambda k: models_metrics[k]['f1_score'])    
        logger.info(f"Best model is {best_model_name} with F1 score: {models_metrics[best_model_name]['f1_score']}")
        
        # Save the best model
        save_best_model(best_model_name)

    except Exception as e:
        logger.error(f"An error occurred during the best model selection process: {e}")
        raise

if __name__ == "__main__":
    process_best_model_selection()
