import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Path to the input CSV file
input_file_path = "c:/Users/Subha/OneDrive/Desktop/HR_capstone_dataset.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(input_file_path)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.4, random_state=42)

# Define the path for saving the processed data
data_path = os.path.join("data", "raw")

# Create the directory if it does not exist
os.makedirs(data_path, exist_ok=True)

# Save the training and testing data to CSV files
train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)