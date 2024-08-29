from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import mlflow
import dagshub
import os

from dotenv import load_dotenv

model_path = 'best_model.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
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
