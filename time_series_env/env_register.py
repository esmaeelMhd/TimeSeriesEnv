from dataclasses import dataclass, field
from typing import Any

from gymnasium.envs.registration import register

from .env import TimeSeriesEnv
from .env_util import filter_kwargs, get_class_attributes
from .time_series_data import TimeSeriesData
from .scaler_handler import ScalerHandler
from .config_util import load_config

#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))

# Define the default environment ID
DEFAULT_ENV_ID = 'TimeSeriesEnv-v1'

@dataclass(eq=False)
class CustomEnvRegistrar:
    """
    A class responsible for registering and creating a custom environment.
    
    Attributes:
    - env_id: The ID of the environment to register.
    - model_kwargs: Keyword arguments for the model builder.
    - env_kwargs: Keyword arguments for the environment.
    - reward_kwargs: Keyword arguments for the reward function.
    """
    env_id: str
    data: Any = None
    model_builder: Any = None
    predictor: Any = None
    reward_function: Any = None
    env_kwargs: dict = field(default_factory=dict)


    def __post_init__(self):
        """Automatically register the environment after initialization."""
        self._register_env()

    def _register_env(self):
        """Registers the environment with the provided entry point."""
        register(
            id=self.env_id,
            entry_point=self._entry_point,
            kwargs=self.env_kwargs,
        )

    def _entry_point(self, **kwargs):
        """
        Entry point for the environment creation.
        This function initializes the environment with the model, data, and reward function.
        """
        # Merge kwargs with the predefined env_kwargs
        all_kwargs = {**self.env_kwargs, **kwargs}

        # Create and return the TimeSeriesEnv with the specified components and kwargs
        env = TimeSeriesEnv(
            data=self.data,
            model_builder=self.model_builder,
            predictor=self.predictor,
            reward_function=self.reward_function,
            **all_kwargs  # Pass all kwargs (merged)
        )

        return env
    
def env_register(
    env_id=DEFAULT_ENV_ID, 
    config=None, 
    data=None, 
    model=None, 
    model_builder=None,
    predictor=None,
    scaler_handler=None,
    reward_function=None,
    config_file=None,
    **kwargs
    ):
    """
    Registers and creates the environment using configuration.
    
    Parameters:
    - env_id: The ID of the environment to register.
    - config: A dictionary containing configuration for model, reward, env, and data.
    - data: Data object, can be passed separately.
    - model: Model object, can be passed separately.
    - model_builder: Model builder function, can be passed separately.
    - config_file: Path to a YAML config file. If provided, it overrides the `config` dictionary.
    - kwargs: Any other keyword arguments passed manually.
    
    Usage:
    - Manual mode: Provide all arguments manually.
    - Config file mode: Provide `config_file` path to a YAML file.
    - Dictionary mode: Provide config dictionaries directly.
    - Individual arguments mode: Provide variables directly (e.g., env_param1=value, model_param1=value).
    """
    if model is None and model_builder is None and predictor is None:
        raise ValueError('A model, model_builder, or predictor must be provided.')
    
    if reward_function is None:
        raise ValueError('A reward_function must be provided.')
    
    # If config_file is provided, load the config from the file
    if config_file:
        config = load_config(config_file)
    
    # If no config is provided, initialize an empty one
    if config is None:
        config = {}

    # Load the configuration sections from config or initialize empty
    data_kwargs = config.get('data_config', {})
    scaler_kwargs = config.get('scaler_config', {})
    env_kwargs = config.get('env_config', {})

    # Handle kwargs (manual variables passed directly)
    data_kwargs.update(filter_kwargs(kwargs, get_class_attributes(TimeSeriesData)))
    scaler_kwargs.update(filter_kwargs(kwargs, get_class_attributes(ScalerHandler)))
    env_kwargs.update(filter_kwargs(kwargs, get_class_attributes(TimeSeriesEnv)))

    # Update env_kwargs with the final data_config
    env_kwargs.update({'data_config': data_kwargs})
    
    if data is None:
        data = TimeSeriesData(**data_kwargs)

    if model_builder is None and predictor is None:
        from .model_builder import ModelBuilder

        model_builder_kwargs = config.get('model_config', {})
        model_builder_kwargs.update(filter_kwargs(kwargs, get_class_attributes(ModelBuilder)))

        if scaler_handler is None and model_builder_kwargs.get('scale_data', False):
            scaler_handler = ScalerHandler(**scaler_kwargs)
            
        model_builder = ModelBuilder(**model_builder_kwargs,
                                     data=data,
                                     model=model,
                                     scaler_handler=scaler_handler
                                     )

    # Register and create the environment
    CustomEnvRegistrar(env_id=env_id, data=data, model_builder=model_builder,
                       predictor=predictor, reward_function=reward_function,
                       env_kwargs=env_kwargs)

