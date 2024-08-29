from flask import Flask, request, render_template
import mlflow
import pandas as pd
import os
import logging

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Set up MLflow tracking URI
dagshub_url = "dagshub.com"
repo_owner = "datascience-1100"
repo_name = "End-to-End-Employee-Attrition-Rate-Project"
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# Load the model from MLflow Model Registry
model_name = "final-model"
model_version = "1"  # Specify the correct version

try:
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri=model_uri)
    logger.info(f"Model loaded successfully from {model_uri}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

app = Flask(__name__)

# Define model columns
model_columns = [
    'satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours',
    'tenure', 'work_accident', 'promotion_last_5years', 'salary_high', 'salary_low', 'salary_medium',
    'dept_IT', 'dept_RandD', 'dept_accounting',
    'dept_hr', 'dept_management', 'dept_marketing',
    'dept_product_mng', 'dept_sales', 'dept_support',
    'dept_technical']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Extract features from the form
        input_data = {
            'satisfaction_level': float(request.form['satisfaction_level']),
            'last_evaluation': float(request.form['last_evaluation']),
            'number_project': int(request.form['number_project']),
            'average_monthly_hours': float(request.form['average_monthly_hours']),
            'time_spend_company': int(request.form['time_spend_company']),
            'work_accident': int(request.form['work_accident']),
            'promotion_last_5years': int(request.form['promotion_last_5years']),
            'department': request.form['department'],
            'salary': request.form['salary']
        }

        # Create DataFrame from the inputs
        input_df = pd.DataFrame([input_data])

        # Apply one-hot encoding
        input_df = pd.get_dummies(input_df, columns=['salary', 'department'], drop_first=False)

        # Rename one-hot encoded columns to match the model's expectations
        input_df.columns = input_df.columns.str.replace('department_', 'dept_')

        # Add missing columns with default values
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match the model's expectations
        input_df = input_df[model_columns]

        # Predict
        prediction = model.predict(input_df)
        output = 'left' if prediction[0] == 1 else 'Not Left'
        
        return render_template('index.html', prediction_text=f'Prediction: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True, port=8000)
