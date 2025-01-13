import os
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, fields
from typing import Union, Any

#from data_providers import former_data

@dataclass(eq=False)
class Predictor:
    """
    A class for making predictions using any model that processes sequences of shape 
    (seq_len, in_features) and outputs predictions of shape (pred_len, out_features).

    This class handles model initialization, data preparation, and prediction.

    Attributes:
        device (str): Device to run the model on ('cpu' or 'cuda').
        scale_data (bool): Flag indicating whether to scale the data.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        model (torch.nn.Module): The model for prediction.
        scaler_handler: Object handling data scaling operations.
    """

    device: str
    scale_data: bool
    in_features: int
    out_features: int
    model: Any = None
    scaler_handler: Any = None

    def __post_init__(self):
        if self.model == None:
            raise ValueError('A model must be provided which gets input with the shape of (seq_len, in_features)' +\
                             'and predicts the output with the shape of (pred_len, out_features).')
        if self.scale_data and self.scaler_handler == None:
            raise ValueError('A scaler_handler must be provided when scale_data = True.')
            
    @staticmethod
    def from_namespace(args, **kwargs) -> 'Predictor':
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(Predictor))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        filtered_args.update(kwargs)
        return Predictor(**filtered_args)
        
    def create_forecast_data(self, arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Prepare forecast data for model prediction.

        Args:
            arr (Union[np.ndarray, torch.Tensor]): Array of input data.

        Returns:
            torch.Tensor: Tensor of forecast data ready for prediction.
        """
        X_forecast = np.array(arr).reshape(1, -1, arr.shape[-1])
        return torch.Tensor(X_forecast).to(self.device)

    def predict(self, df_forecast: Union[pd.DataFrame, np.ndarray], do_scale: bool = True) -> np.ndarray:    
        """
        Generate and return one-step predictions for the given data.

        Args:
            df_forecast (Union[pd.DataFrame, np.ndarray]): DataFrame or array containing the forecast data.
            do_scale (bool): Whether to scale the input data.

        Returns:
            np.ndarray: Array of predicted values with shape (pred_len, out_features).
        """
        if self.scale_data and do_scale:
            df_forecast = self.scaler_handler.scale_data(df_forecast)

        X_forecast = self.create_forecast_data(df_forecast)
        
        # Do the prediction
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_forecast).cpu().numpy()
        
        predictions = predictions[0] if predictions.ndim > 2 else predictions
        predictions = predictions.reshape(-1, self.out_features)

        if self.scale_data and do_scale:
            predictions = self.scaler_handler.inverse_transform(predictions)

        return predictions

# === formers =================================================================
@dataclass(eq=False)
class TransformerPredictor:
    """
    A class for making predictions using transformer-based models.

    This class handles model initialization, data preparation, and prediction for 
    transformer or attention-based models.

    Attributes:
        device (str): Device to run the model on ('cpu' or 'cuda').
        model: The transformer model used for prediction.
        df_raw: Raw DataFrame containing the input data.
        scaler_handler_path: Path to the scaler_handler used for data normalization.
        args: Arguments containing model configurations and settings.
    """

    device: str
    model: Any
    df_raw: pd.DataFrame
    scaler_handler: Any
    args: Any

    def __post_init__(self):
        """Initializes the model and handles multi-GPU setup if necessary."""
        self.device = torch.device(self.device)
        self.use_multi_gpu = torch.cuda.is_available() and self.device.type == 'cuda' and torch.cuda.device_count() > 1

        if self.use_multi_gpu:
            self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.to(self.device)

    def _get_data(self, df_predict: pd.DataFrame) -> tuple:
        """
        Prepares the data for prediction.

        Args:
            df_predict: DataFrame containing the data to be predicted.

        Returns:
            tuple: Dataset and DataLoader objects.
        """
        return # former_data(args=self.args, flag='pred', df_predict=df_predict, 
                           # scaler_handler=self.scaler_handler, df_raw=self.df_raw, df_stamp=None)

    def _load_model_checkpoint(self, setting: str) -> None:
        """Loads a pre-trained model checkpoint."""
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def _run_prediction(self, pred_loader: Any) -> np.ndarray:
        """Runs the prediction process and returns the predictions."""
        preds = []
        self.model.eval()

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in pred_loader:
                batch_x, batch_x_mark, dec_inp = self._prepare_input(batch_x, batch_y, batch_x_mark)
                outputs = self._model_forward(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                preds.append(outputs.cpu().numpy())

        return np.concatenate(preds, axis=0)

    def _prepare_input(self, batch_x: torch.Tensor, batch_y: torch.Tensor, batch_x_mark: torch.Tensor) -> tuple:
        """
        Prepares the input data for the transformer model.

        Args:
            batch_x: Batch of input data.
            batch_y: Batch of output data.
            batch_x_mark: Batch of input time markers.

        Returns:
            tuple: Prepared input data for the transformer model.
        """
        batch_x = batch_x.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        dec_inp = torch.zeros([batch_y.size(0), self.args.pred_len, batch_y.size(2)]).float().to(self.device)
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        return batch_x, batch_x_mark, dec_inp

    def _model_forward(self, batch_x: torch.Tensor, batch_x_mark: torch.Tensor, dec_inp: torch.Tensor, batch_y_mark: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer model.

        Args:
            batch_x: Batch of input data.
            batch_x_mark: Batch of input time markers.
            dec_inp: Decoder input.
            batch_y_mark: Batch of output time markers.

        Returns:
            torch.Tensor: Model outputs.
        """
        if 'Linear' in self.args.model:
            return self.model(batch_x)
        elif self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        else:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    
    def predict(self, df_predict: pd.DataFrame, setting: str, load: bool = False) -> np.ndarray:
        """
        Generate predictions using the transformer model.

        Args:
            df_predict: DataFrame containing the prediction data.
            setting: String indicating the setting for saving results.
            load: Boolean indicating whether to load a pre-trained model.

        Returns:
            np.ndarray: Array of predicted values.
        """
        pred_data, pred_loader = self._get_data(df_predict)

        if load:
            self._load_model_checkpoint(setting)

        return self._run_prediction(pred_loader)