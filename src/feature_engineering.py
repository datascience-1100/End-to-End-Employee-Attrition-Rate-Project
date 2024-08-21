import numpy as np
import pandas as pd
import os

train_data=pd.read_csv('./data/processed/train_processed.csv')
test_data= pd.read_csv('./data/processed/test_processed.csv')

# One-hot encode the categorical variables as needed and save resulting dataframe in a new variable
def ohe(df):
  df = pd.get_dummies(df, prefix=['salary', 'dept'], columns = ['salary', 'department'], drop_first=False)
  return df

train_featured_data= ohe(train_data)
test_featured_data= ohe(test_data)

# Define the path for saving the processed data
data_path = os.path.join("data", "featured")

# Create the directory if it does not exist
os.makedirs(data_path, exist_ok=True)

# Save the training and testing data to CSV files
train_featured_data.to_csv(os.path.join(data_path, 'train_featured.csv'))
test_featured_data.to_csv(os.path.join(data_path, 'test_featured.csv'))


