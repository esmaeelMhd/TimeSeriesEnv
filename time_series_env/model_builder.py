import os
from dataclasses import dataclass, fields
from typing import Any, Optional
import torch
import torch.nn as nn

from time_series_env.time_series_data import TimeSeriesData
from time_series_env.scaler_handler import ScalerHandler
from time_series_env.predictors import Predictor

@dataclass(eq=False)
class ModelBuilder:
    """
    A class to build and manage machine learning models with customizable configurations.

    Attributes:
        Data (type): The data handling class.
        ScalerHandler (type): The scaler handling class.
        data (Any): The data instance.
        model (Any): The machine learning model instance.
        scaler_handler (Any): The scaler handler instance.
        predictor (Any): The predictor instance.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        load_model (bool): Flag indicating whether to load a pre-trained model.
        chkpt_root_path (str): Path where model checkpoints are stored.
        chkpt_folder (str): Name of the model configuration.
        chkpt_name (str): Filename of the model training checkpoint.
        scale_data (bool): Flag indicating whether to scale the data.
        scaler_root_path (str): Path where scalers are stored.
        scale_time_f (str): Indicates if the time feature is scaled.
        load_scaler (bool): Flag indicating whether to load a pre-saved scaler.
        device (str): Device to be used for model computation ('cpu' or 'cuda').
        use_multi_gpu (bool): Flag indicating whether to use multiple GPUs.
        device_ids (str): Comma-separated string of GPU device IDs to use.
    """
    data: TimeSeriesData = None
    model: Any = None
    scaler_handler: ScalerHandler = None
    predictor: Any = None
    
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    load_model: bool = True
    chkpt_root_path: str = 'checkpoints'
    chkpt_folder: str = ''
    chkpt_name: str = ''
    model_type: Optional[int] = None

    scale_data: bool = False
    scaler_root_path: str = './scalers'
    scale_time_f: str = 'unscaled'
    load_scaler: bool = False
    
    device: str = 'cuda'
    use_multi_gpu: bool = False
    device_ids: str = '0,1'

    def __post_init__(self):
        """Post-initialization to set up the device, validate configurations, and prepare the model."""
        if not self.model:
            raise ValueError("A model architecture must be provided.")
        
        self.data = self.Data() if self.data is None else self.data
        self.df = self.data.df

        self.in_features = self.in_features or (
            len(self.data.act_vars) + len(self.data.exog_vars) + len(self.data.target_vars)
        )

        if self.scale_data:
            self._initialize_scaler_handler()

        self.model = self._load_model()
        self.create_predictor()

    @staticmethod
    def from_namespace(args, **kwargs) -> 'ModelBuilder':
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(ModelBuilder))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        filtered_args.update(kwargs)
        return ModelBuilder(**filtered_args)

    def _initialize_scaler_handler(self) -> None:
        """Initializes the scaler handler based on the provided settings."""
        if not self.scaler_handler:
            self.scaler_handler = self.ScalerHandler(
                load_scaler=self.load_scaler,
                scaler_root_path=self.scaler_root_path,
                scale_time_f=self.scale_time_f,
                data_scaler_name=getattr(self, 'data_scaler_name', ''),
                time_scaler_name=getattr(self, 'time_scaler_name', ''),
                scaler_folder=self.chkpt_folder,
                in_features=self.in_features,
                df=self.df if not self.load_scaler else None
            )

    def _build_model(self) -> None:
        """Initializes the model and handles multi-GPU setup if necessary."""
        self.device = torch.device(self.device)
        self.use_multi_gpu = torch.cuda.is_available() and self.device.type == 'cuda' and torch.cuda.device_count() > 1

        if self.use_multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        self.model = self.model.to(self.device)

    def _load_model_checkpoint(self) -> None:
        """
        Loads the model state from a checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file is not found.
            RuntimeError: For any other exceptions that occur during loading.
        """
        model_path = os.path.join(self.chkpt_root_path, self.chkpt_folder)
        checkpoint_file = os.path.join(model_path, self.chkpt_name)
        
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

        try:
            self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
        except Exception as e:
            raise RuntimeError(f"Failed to load the model checkpoint: {str(e)}")

    def _load_model(self) -> Any:
        """
        Loads the model from a checkpoint and sets it up for inference or further training.

        Returns:
            Any: The loaded model.
        """
        self._build_model()
        if self.load_model:
            self._load_model_checkpoint()
        # self.model = self._optimize_model_with_jit(self.model)
        self.model.eval()
        return self.model
    
    def _optimize_model_with_jit(self, model):
        """
        Checks if the given model is a PyTorch model, and if so, optimizes it using torch.jit.script.
        
        Args:
            model: The model to be checked and possibly optimized.
            
        Returns:
            The optimized model if it is a PyTorch model, otherwise returns the original model.
        """
        # Check if the model is an instance of torch.nn.Module
        if isinstance(model, nn.Module):
            print("Detected PyTorch model. Applying TorchScript optimization...")
            try:
                # Optimize the model using TorchScript
                scripted_model = torch.jit.script(model)
                return scripted_model
            except Exception as e:
                print(f"Failed to apply TorchScript: {e}")
                return model  # Return the original model if scripting fails
        else:
            print("The provided model is not a PyTorch model. Skipping TorchScript optimization.")
            return model
    
    def create_predictor(self) -> Any:
        """
        Initializes the predictor based on the model type.

        Returns:
            Any: The initialized predictor object.
        """
        if not self.predictor:
            self.predictor = Predictor(
                device=self.device,
                scale_data=self.scale_data,
                in_features=self.in_features,
                out_features=self.out_features or self.in_features,
                model=self.model,
                scaler_handler=self.scaler_handler
            )
        return self.predictor
