import numpy as np
import pandas as pd
import os
import json

# For metrics and helpful functions
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import pickle

clf= pickle.load(open('model.pkl', 'rb'))
test_data=pd.read_csv('./data/featured/test_featured.csv')

# Select the features to be used in your model
X_test= test_data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure', 'work_accident', 'promotion_last_5years', 'salary_high', 'salary_low', 'salary_medium' , 'dept_IT', 'dept_RandD', 'dept_accounting', 'dept_hr', 'dept_management', 'dept_marketing', 'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical']]
# Isolate the outcome variable
y_test = test_data['left']

# Use the logistic regression model to get predictions on the test set
y_pred = clf.predict(X_test)

# Create classification report for logistic regression model
accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred)
recall=recall_score(y_test, y_pred)


metrics_dict={
    'accuracy': accuracy,
    'precision':precision,
    'recall':recall
}

with open('metrics.json', 'w') as file:
 json.dump(metrics_dict, file, indent=4)