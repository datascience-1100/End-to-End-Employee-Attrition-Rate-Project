import mlflow
import pickle
import json
import logging
import dagshub
from mlflow import log_metric, log_param

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
logger = logging.getLogger('register_model')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('register_model.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def register_model(run_id, model_name, model_path):
    
    try:
        model_uri = f"runs:/{run_id}/{model_path}"
        result = mlflow.register_model(model_uri, model_name)
        
        logger.info(f"Model registered with name: {model_name}, URI: {model_uri}")
        logger.info(f"Model version: {result.version}")

        # Save the model version to a file
        with open('registered_model_version.txt', 'w') as version_file:
            version_file.write(result.version)
        
        return result.version

    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise

def load_model(model_path):
   
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def save_model_artifact(model_path, artifact_path="model_artifact.pkl"):
   
    try:
        model = load_model(model_path)
        with open(artifact_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model artifact saved as {artifact_path}")
    except Exception as e:
        logger.error(f"Error saving model artifact: {e}")
        raise

def log_model_artifact(artifact_path):
    
    try:
        mlflow.log_artifact(artifact_path)
        logger.info(f"Model artifact logged to MLflow: {artifact_path}")
    except Exception as e:
        logger.error(f"Error logging model artifact to MLflow: {e}")
        raise

def main():
    model_path = "best_model.pkl"
    artifact_path = "model_artifact.pkl"
    model_name = "final-model"

    try:
        # Start an MLflow run
        with mlflow.start_run(run_name="Model_Registration") as run:
            run_id = run.info.run_id
            
            # Save and log the model artifact
            save_model_artifact(model_path, artifact_path)
            log_model_artifact(artifact_path)
            
            # Register the model
            model_version = register_model(run_id, model_name, artifact_path)
            logger.info(f"Model registered successfully with version: {model_version}")

    except Exception as e:
        logger.error(f"An error occurred during model registration: {e}")
        raise

if __name__ == "__main__":
    main()
