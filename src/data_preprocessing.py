import numpy as np
import pandas as pd
import os

train_data=pd.read_csv('./data/raw/train.csv')
test_data= pd.read_csv('./data/raw/test.csv')

print('Ok')

def rename_columns(df):
    
  df = df.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})
  return df


# Drop duplicates and save resulting dataframe in a new variable as needed
def remove_duplicates(df):
 df = df.drop_duplicates(keep='first')
 return df


def remove_outliers(df):
# Compute the 25th percentile value in `tenure`
    percentile25 = df['tenure'].quantile(0.25)

    # Compute the 75th percentile value in `tenure`
    percentile75 = df['tenure'].quantile(0.75)

    # Compute the interquartile range in `tenure`
    iqr = percentile75 - percentile25

    # Define the upper limit and lower limit for non-outlier values in `tenure`
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)

    # Identify subset of data containing outliers in `tenure`
    outliers = df[(df['tenure'] > upper_limit) | (df['tenure'] < lower_limit)]

    # Count how many rows in the data contain outliers in `tenure`
    print("Number of rows in the data containing outliers in `tenure`:", len(outliers))
    
    return(df)

def process_data(df):
    df = rename_columns(df)
    df = remove_outliers(df)
    df = remove_duplicates(df)
    return df

    
    
train_processed_data= process_data(train_data)
test_processed_data= process_data(test_data)

# Define the path for saving the processed data
data_path = os.path.join("data", "processed")

# Create the directory if it does not exist
os.makedirs(data_path, exist_ok=True)

# Save the training and testing data to CSV files
train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'))
test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'))


