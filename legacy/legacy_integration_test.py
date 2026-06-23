import unittest
import inspect
import torch
import gymnasium as gym
from argparse import Namespace
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")

from models import LSTMModel
from linear_models import DLinear, NLinear
from former_models import Informer, Transformer, Autoformer

from load_my_args import load_model_args, load_reward_function
from time_series_env.env import TimeSeriesEnv
from time_series_env.config_util import load_config, parse_args
from time_series_env.env_register import env_register
from time_series_env.time_series_data import TimeSeriesData
from time_series_env.env_util import filter_kwargs, get_class_attributes
from time_series_env.model_builder import ModelBuilder, ScalerHandler

class TestTimeSeriesEnv(unittest.TestCase):
    def test_env_initialization(self):
        config = load_config('config.yaml')
        env = TimeSeriesEnv(config)
        self.assertIsNotNone(env)

    def test_env_step(self):
        config = load_config('config.yaml')
        env = TimeSeriesEnv(config)
        state = env.reset()
        action = env.choose_real_action()
        state, reward, done, trunc, info = env.step(action)
        self.assertFalse(done)

    def test_multi_gpu(self):
        config = load_config('config.yaml')
        config['use_gpu'] = True
        env = TimeSeriesEnv(config)
        self.assertTrue(env.use_multi_gpu or torch.cuda.device_count() <= 1)

def validate_config(config):
    required_keys = ['use_gpu', 'env_kwargs', 'model_kwargs']
    for key in required_keys:
        if key not in config:
            raise ValueError(f'Missing required config key: {key}')
    
    if config['use_gpu'] and not torch.cuda.is_available():
        raise ValueError('use_gpu is set to True, but no CUDA-enabled GPU is available.')

def test_different_configs():
    configs = [
        {'use_gpu': True, 'env_kwargs': {'seq_len': 10}, 'model_kwargs': {'model_type': 'LSTM'}},
        {'use_gpu': False, 'env_kwargs': {'seq_len': 20}, 'model_kwargs': {'model_type': 'GRU'}},
        {'use_gpu': True, 'env_kwargs': {'seq_len': 5}, 'model_kwargs': {'model_type': 'Transformer'}}
    ]
    
    for config in configs:
        env = TimeSeriesEnv(config)
        state = env.reset()
        done = False
        while not done:
            action = env.choose_real_action()
            state, reward, done, _, info = env.step(action)
        env.close()

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
    
    agent_args = Namespace(
        experiment=1,
        min_el=10,
        max_el=360,
        const_el=100,
        title='test',
        agent_name='test_env_register',
        results_root_path='./results')
    
    env.set_agent_args(agent_args)
    
    # Use the environment
    state, _ = env.reset()
    done = False
    while not done:
        action = env.choose_actual_action()
        state, reward, done, _, info = env.step(action)
        # print(f"State: {state}, Reward: {reward}, Done: {done}")
    
    env.close()
    
def main():            
    def method_one():
        # Load configuration and parse arguments
        args = parse_args()
        config = load_config(args['config'])
        
        # Override config values with command-line arguments if provided
        for key, value in args.items():
            if value is not None:
                config[key] = value
        
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
        env_register(env_id='PhosphorusEnv-v1', config=config, data=data, model=model, 
                     model_builder=model_builder, scaler_handler=scaler_handler, 
                     reward_function=reward_function)
    
    def method_two():
        # Load configuration and parse arguments
        args = parse_args()
        config = load_config(args['config'])
        
        # Override config values with command-line arguments if provided
        for key, value in args.items():
            if value is not None:
                   config[key] = value
        
        model_args_dict = load_model_args()
        model = build_model('LSTM', model_args_dict)
        
        for key, value in model_args_dict.items():
            if value is not None:
                config['model_config'][key] = value
        
        reward_kwargs = config['reward_config']
        reward_function = load_reward_function(reward_kwargs=reward_kwargs)
            
        # Registering the environment using the configuration file
        env_register(env_id='PhosphorusEnv-v1', config=config, model=model, reward_function=reward_function)
    
    def method_three():
        pass
    
    # Env initialization method
    method_one()
    #method_two()
            
    test_env(env_id='PhosphorusEnv-v1')
    
    # unittest.main()

    # Start training or evaluation process
    # env.train() or env.evaluate()

if __name__ == "__main__":
    # unittest.main()
    # test_different_configs()
    main()
