import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
import yaml

max_iter=yaml.safe_load(open('params.yaml', 'r'))['model_building']['max_iter']



train_data=pd.read_csv('./data/featured/train_featured.csv')

# Select the features to be used in your model
X_train = train_data[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure', 'work_accident', 'promotion_last_5years', 'salary_high', 'salary_low', 'salary_medium' , 'dept_IT', 'dept_RandD', 'dept_accounting', 'dept_hr', 'dept_management', 'dept_marketing', 'dept_product_mng', 'dept_sales', 'dept_support', 'dept_technical']]
# Isolate the outcome variable
y_train = train_data['left']

# Construct a logistic regression model and fit it to the training dataset
clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)

pickle.dump(clf, open('model.pkl', 'wb'))

