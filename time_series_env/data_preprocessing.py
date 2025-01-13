import os
import pandas as pd
import numpy as np

def load_dataset(data_root_path: str, data_name: str, index_col: str) -> pd.DataFrame:
    """Loads the dataset used in training the model."""
    dataset_path = os.path.join(data_root_path, data_name)
    df = pd.read_csv(dataset_path, index_col=index_col, parse_dates=[index_col])
    df.sort_index(inplace=True)
    df = df.astype('float32').ffill()
    return df

class FrequencyError(Exception):
    """Custom exception for non-uniform frequency in the dataset."""
    pass

def check_frequency_uniformity(df: pd.DataFrame) -> bool:
    """Checks if the frequency of the dataset is uniform."""
    time_diffs = df.index.to_series().diff()
    is_uniform = (time_diffs.iloc[1:] == time_diffs.mode()[0]).all()
    if not is_uniform:
        raise FrequencyError('The frequency of the time series data is not uniform. Please check your data.')
    return is_uniform

def add_time_specs(df: pd.DataFrame) -> pd.DataFrame:
    """Embeds cyclical time features into the DataFrame based on its index."""
    hours = df.index.hour
    months = df.index.month
    days_of_week = df.index.dayofweek

    df['sin_hour'] = np.sin(2 * np.pi * hours / 24)
    df['cos_hour'] = np.cos(2 * np.pi * hours / 24)
    df['sin_month'] = np.sin(2 * np.pi * (months - 1) / 12)
    df['cos_month'] = np.cos(2 * np.pi * (months - 1) / 12)
    df['sin_day_of_week'] = np.sin(2 * np.pi * days_of_week / 7)
    df['cos_day_of_week'] = np.cos(2 * np.pi * days_of_week / 7)

    return df
