import inspect
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")

from models import LSTMModel

from load_my_args import load_model_args, load_reward_function
from time_series_env.config_util import load_config, parse_args
from time_series_env.env_register import env_register
from time_series_env.time_series_data import TimeSeriesData
from time_series_env.env_util import filter_kwargs, get_class_attributes
from time_series_env.model_builder import ModelBuilder, ScalerHandler


def build_model(class_name='LSTM', model_args_dict=None):
    model_dict = {
        'LSTM': LSTMModel
    }

    model_class = model_dict[class_name]
    init_signature = inspect.signature(model_class.__init__)
    init_params = init_signature.parameters.keys()
    
    # Filter kwargs to only include keys that are parameters of the __init__ method
    filtered_kwargs = {k: v for k, v in model_args_dict.items() if k in init_params}
    
    # Initiate the class with filtered kwargs
    model = model_class(**filtered_kwargs)

    return model

def register_env(env_id):
    # Load configuration and parse arguments
    # args = parse_args()
    config = load_config('config.yaml')
    
    # Override config values with command-line arguments if provided
    # for key, value in args.items():
    #     if value is not None:
    #         config[key] = value
    
    model_args_dict = load_model_args()
    model = build_model('LSTM', model_args_dict)
    
    for key, value in model_args_dict.items():
        if value is not None:
            config['model_config'][key] = value
    
    reward_kwargs = config['reward_config']
    reward_function = load_reward_function(reward_kwargs=reward_kwargs)
    
    data_kwargs = config['data_config']
    data = TimeSeriesData(**data_kwargs)
    
    model_builder_kwargs = config['model_config']
    scaler_kwargs = config['scaler_config']
    scaler_kwargs['scaler_folder'] = model_builder_kwargs['model_name']
    
    if model_builder_kwargs['scale_data']:
        scaler_handler = ScalerHandler(**scaler_kwargs)
    else:
        scaler_handler = None
        
    model_builder_kwargs = filter_kwargs(model_builder_kwargs, get_class_attributes(ModelBuilder))
    model_builder = ModelBuilder(**model_builder_kwargs,
                                 data=data,
                                 model=model,
                                 scaler_handler=scaler_handler
                                 )

    # Registering the environment using the configuration file
    env_register(env_id=env_id, config=config, data=data, model=model, 
                 model_builder=model_builder, scaler_handler=scaler_handler, 
                 reward_function=reward_function)