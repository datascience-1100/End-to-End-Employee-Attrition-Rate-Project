import numpy as np
import pandas as pd
import os
import logging

logger= logging. getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler= logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler=logging.FileHandler('errors.log')
file_handler.setLevel('ERROR')

formatter=logging.Formatter('%(asctime)s - %(name)s -%(levelname)s - %(message)s' )
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Function to read CSV files
def read_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        logger.error('File is Not Found')
        raise
    except pd.errors.EmptyDataError:
        logger.error('File is Empty')
        raise


# Function to rename columns
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_mapping = {
        'Work_accident': 'work_accident',
        'average_montly_hours': 'average_monthly_hours',
        'time_spend_company': 'tenure',
        'Department': 'department'
    }
    return df.rename(columns=column_mapping)

# Function to drop duplicates
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(keep='first')

# Function to compute IQR and define outlier limits
def compute_iqr_limits(df: pd.DataFrame, column: str) -> tuple:
    percentile25 = df[column].quantile(0.25)
    percentile75 = df[column].quantile(0.75)
    iqr = percentile75 - percentile25
    upper_limit = percentile75 + 1.5 * iqr
    lower_limit = percentile25 - 1.5 * iqr
    return lower_limit, upper_limit

# Function to remove outliers based on the IQR method
def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    lower_limit, upper_limit = compute_iqr_limits(df, column)
    return df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]

# Function to process the DataFrame by chaining the previous functions
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    df = rename_columns(df)
    df = remove_outliers(df, 'tenure')
    df = remove_duplicates(df)
    return df

# Function to save processed data to CSV
def save_data(df: pd.DataFrame, data_path: str, filename: str) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        file_path = os.path.join(data_path, filename)
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    except IOError as e:
        print(f"Error: Failed to save data to '{file_path}'. {e}")
        raise

# Main function to execute the entire pipeline
def main() -> None:
    train_data = read_csv('./data/raw/train.csv')
    test_data = read_csv('./data/raw/test.csv')

    train_processed_data = process_data(train_data)
    test_processed_data = process_data(test_data)

    data_path = os.path.join("data", "processed")

    save_data(train_processed_data, data_path, 'train_processed.csv')
    save_data(test_processed_data, data_path, 'test_processed.csv')

if __name__ == "__main__":
    main()
