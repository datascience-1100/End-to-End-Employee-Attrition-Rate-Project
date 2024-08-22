import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

def load_params(params_path):
   test_size=yaml.safe_load(open(params_path, 'r'))['data_ingestion']['test_size']
   return test_size

# Path to the input CSV file
def read_data(url):
   df = pd.read_csv(url)
   return df

# Define the path for saving the processed data
def save_data(data_path, train_data,test_data):
    # Create the directory if it does not exist
    os.makedirs(data_path, exist_ok=True)
    # Save the training and testing data to CSV files
    train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    
def main():
    test_size=load_params('params.yaml')
    df=read_data("c:/Users/Subha/OneDrive/Desktop/HR_capstone_dataset.csv")
    # Split the data into training and testing sets
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    data_path = os.path.join("data", "raw")
    save_data(data_path, train_data,test_data)
    
if __name__=="__main__":
    main()