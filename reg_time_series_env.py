import unittest
import inspect
import torch
import gymnasium as gym
import os
import sys
from argparse import Namespace
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .models import LSTMModel
from .linear_models import DLinear, NLinear
from .former_models import Informer, Transformer, Autoformer

from .load_my_args import load_model_args, load_reward_function
from .time_series_env.env import TimeSeriesEnv
from .time_series_env.config_util import load_config, parse_args
from .time_series_env.env_register import env_register
from .time_series_env.time_series_data import TimeSeriesData
from .time_series_env.env_util import filter_kwargs, get_class_attributes
from .time_series_env.model_builder import ModelBuilder, ScalerHandler



def build_model(class_name='LSTM', model_args_dict=None):
    model_dict = {
        'LSTM': LSTMModel,
        'Autoformer': Autoformer,
        'Transformer': Transformer,
        'Informer': Informer,
        'DLinear': DLinear,
        'NLinear': NLinear
    }

    model_class = model_dict[class_name]
    init_signature = inspect.signature(model_class.__init__)
    init_params = init_signature.parameters.keys()
    
    # Filter kwargs to only include keys that are parameters of the __init__ method
    filtered_kwargs = {k: v for k, v in model_args_dict.items() if k in init_params}
    
    # Initiate the class with filtered kwargs
    model = model_class(**filtered_kwargs)

    return model

def test_env(env_id='TestEnv-v1'):
    # Create the environment
    env = gym.make(env_id)
    
    agent_env_args = Namespace(
        obs_history=10,
        experiment=1,
        min_el=10,
        max_el=360,
        const_el=100,
        norm_values=True,
        title='test',
        agent_name='test_env_register')
    
    env.set_agent_env_args(agent_env_args)
    
    # Use the environment
    state, _ = env.reset()
    done = False
    while not done:
        action = env.choose_real_action()
        state, reward, done, _, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
    
    env.close()

def build_and_register(id, env_args=None):
    # Load configuration and parse arguments
    config = load_config('./envs/config.yaml')
    
    args_dict = vars(env_args)  # Convert Namespace object to a dictionary
     
    # Update the configuration with values from args_dict
    for key, value in args_dict.items():
        if value is not None:
            for config_part in config.keys():
                for variable in config[config_part].keys():
                    if key == variable:
                        config[config_part][key] = value
    
    model_args_dict = load_model_args(args_path='./envs/args', 
                                      scaler_root_path='./envs/scalers', 
                                      folder_name=config['model_config']['chkpt_folder'])
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
    env_register(env_id=id, config=config, data=data, model=model, 
                 model_builder=model_builder, scaler_handler=scaler_handler, 
                 reward_function=reward_function) 
    
def main():            
    # Env initialization method
    build_and_register(id='PhosphorusEnv-v1')
    # test_env(env_id='TestEnv-v1')
    

if __name__ == "__main__":
    # unittest.main()
    # test_different_configs()
    main()
