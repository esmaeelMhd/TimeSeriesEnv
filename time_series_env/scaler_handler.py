import os
import sys
import numpy as np
import pandas as pd
import joblib
import pickle 
import sklearn
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from dataclasses import dataclass, fields
from typing import Optional, Union, Any

from sklearn.preprocessing import MinMaxScaler

@dataclass(eq=False)
class ScalerHandler:
    """
    Manages scaler operations for preprocessing in machine learning models.

    This class can either load a pre-existing scaler from disk or create and fit a new one
    using the provided data.

    Attributes:
        scaler_root_path (str): The root directory for storing scalers.
        scaler_folder (str): The name of the scaler(s) folder.
        scale_time_f (bool): Indicates whether the time feature should be scaled.
        load_scaler (bool): Determines whether to load an existing scaler from disk.
        data_scaler_name (str): Name of the scaler used for data features.
        time_scaler_name (str): Name of the scaler used for time features.
        in_features (int): Number of input features to scale.
        data_scaler (Optional[MinMaxScaler]): The scaler for data features, loaded or newly created.
        time_scaler (Optional[MinMaxScaler]): The scaler for time features, loaded or newly created.
        df (Optional[pd.DataFrame]): The DataFrame to fit the scaler if creating a new one.
    """

    load_scaler: bool
    scaler_root_path: str
    scale_time_f: bool
    data_scaler_name: str
    time_scaler_name: str
    scaler_folder: str = ''
    data_scaler: Optional[Any] = None
    time_scaler: Optional[Any] = None
    in_features: Optional[int] = None
    df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        """Initializes scalers by loading them from disk or fitting them to the provided DataFrame."""
        if self.data_scaler is None:
            if self.load_scaler:
                self._load_scalers()
            else:
                self._initialize_scalers()
    
    
    @staticmethod
    def from_namespace(args, **kwargs) -> 'ScalerHandler':
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(ScalerHandler))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        filtered_args.update(kwargs)
        return ScalerHandler(**filtered_args)

    def _load_scalers(self) -> MinMaxScaler:
        """Load the scaler from disk based on the model type and configuration."""
        scaler_path = os.path.join(self.scaler_root_path, self.scaler_folder)
        try:
            self.data_scaler = self._load_model_specific_scalers(scaler_path, self.data_scaler_name)
            if self.scale_time_f and self.time_scaler is None:
                self.time_scaler = self._load_model_specific_scalers(scaler_path, self.time_scaler_name)
        except FileNotFoundError as e:
            print(f"Scaler file not found: {e}", file=sys.stderr)
            sys.exit(1)
    
    def _initialize_scalers(self) -> None:
        """Creates and fits new scalers based on the provided DataFrame."""
        if self.df is None or self.in_features is None:
            raise ValueError('A DataFrame and in_features must be provided when load_scaler is set to False.')

        self.data_scaler = self.data_scaler or MinMaxScaler()
        self.data_scaler.fit(self.df.iloc[:, :self.in_features])

        if self.scale_time_f and self.time_scaler is None:
            if len(self.df.columns) <= self.in_features:
                raise ValueError('The DataFrame must include time features when scale_time_f is True.')
            self.time_scaler = self.time_scaler or MinMaxScaler()
            self.time_scaler.fit(self.df.iloc[:, -self.in_features:])

    def _load_model_specific_scalers(self, scaler_path: str, scaler_file: str) -> MinMaxScaler:
        """Loads scalers specific to the model type and scaling settings."""
        if '.gz' in scaler_file:
            scaler = joblib.load(os.path.join(scaler_path, scaler_file))
            return scaler
        elif '.pkl' in scaler_file:
            with open(os.path.join(scaler_path, scaler_file), 'rb') as f:
                scaler = pickle.load(f)
            return scaler
        else:
            raise ValueError(f"Scaler file type is unknown for {scaler_file}, use pickle (.pkl) or joblib (.gz) files.")

    def scale_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Applies the appropriate scalers to the input data.

        Parameters:
        data (pd.DataFrame or np.ndarray): The input data to scale.

        Returns:
        np.ndarray: The scaled data.
        """
        data = self._validate_and_prepare_data(data)

        # Apply data scaler
        num_features = self.data_scaler.n_features_in_
        data[:, :num_features] = self.data_scaler.transform(data[:, :num_features])

        # Apply time scaler if available
        if self.scale_time_f:
            time_features = self.time_scaler.n_features_in_
            data[:, -time_features:] = self.time_scaler.transform(data[:, -time_features:])

        return data

    def _validate_and_prepare_data(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Ensures input data is a numpy array and reshapes it as necessary."""
        if isinstance(data, pd.DataFrame):
            data = data.values
        elif not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a pandas DataFrame or numpy array.")

        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 3:
            data = data[0]  # Assume first dimension is irrelevant for scaling
        elif data.ndim != 2:
            raise ValueError("Input data must be 1D, 2D, or 3D.")

        return data
    
    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        """
        Reverses the scaling of data according to the model configuration.

        Parameters:
        arr (np.ndarray): The scaled data to inverse transform.

        Returns:
        np.ndarray: The inverse transformed data.
        """
        arr, original_shape = self._prepare_inverse_transform(arr)

        num_features = self.data_scaler.n_features_in_
        temp_arr = np.zeros((arr.shape[0], num_features))
        temp_arr[:, -arr.shape[1]:] = arr
        temp_arr[:, :num_features] = self.data_scaler.inverse_transform(temp_arr[:, :num_features])
        arr = temp_arr[:, -arr.shape[1]:]

        if self.scale_time_f:
            time_features = self.time_scaler.n_features_in_
            arr[:, -time_features:] = self.time_scaler.inverse_transform(arr[:, -time_features:])

        return arr.reshape(original_shape)

    def _prepare_inverse_transform(self, arr: np.ndarray) -> tuple:
        """Prepares the array for inverse transformation by reshaping and copying it."""
        original_shape = arr.shape
        arr = arr.reshape(-1, arr.shape[-1])  # Ensure 2D shape for transformation
        return np.array(arr), original_shape

