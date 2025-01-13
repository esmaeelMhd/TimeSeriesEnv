"""
Created on April 26, 2024

Author:
    Esmaeel Mohammadi
    Email: esm@kruger.dk, esmo@bio.aau.dk
    GitHub: https://github.com/esmaeelMhd

Description:
    This script serves as the entry point for managing and registering environments for Deep Reinforcement Learning (DRL) models.
    It supports the creation and registration of custom environments with specified configurations and reward functions.

    The script performs the following tasks:
        1. Parses command-line arguments to extract environment, model, and reward function configurations.
        2. Dynamically loads the specified reward function module and function.
        3. Registers the environment with the provided configurations.

Usage:
    To register a new environment, run the following command:
        python -m your_module_name register-env <env_id> <env_args>

    Example:
        python -m your_module_name register-env my_env env_param1=value1 env_param2=value2 model_param1=value1 reward_module.reward_function
"""

import sys
import importlib

from env_register import env_register
from .util import partial_from_args

def parse_args(args):
    """
    Parses command-line arguments to extract environment, model, and reward function configurations.

    Args:
        args (list): List of command-line arguments.

    Returns:
        tuple: A tuple containing dictionaries for environment kwargs, model kwargs, reward function spec, and reward function kwargs.
    """
    env_kwargs = {}
    model_kwargs = {}
    reward_function_kwargs = {}
    reward_function_spec = None

    for arg in args:
        key, value = arg.split('=')
        if key.startswith('env_'):
            env_kwargs[key[4:]] = value
        elif key.startswith('model_'):
            model_kwargs[key[6:]] = value
        elif key.startswith('reward_'):
            reward_function_kwargs[key[7:]] = value
        else:
            reward_function_spec = key

    return env_kwargs, model_kwargs, reward_function_spec, reward_function_kwargs

def load_reward_function(module_name, function_name):
    """
    Dynamically loads the specified reward function from the given module.

    Args:
        module_name (str): The name of the module containing the reward function.
        function_name (str): The name of the reward function to load.

    Returns:
        function: The loaded reward function.
    """
    module = importlib.import_module(module_name)
    return getattr(module, function_name)

def main():
    """
    Main function to handle the command-line interface for registering environments.
    """
    _, cmd, *args = sys.argv

    if cmd == "register-env":
        env_id = args[0]
        env_args = args[1:]
        env_kwargs, model_kwargs, reward_function_spec, reward_function_kwargs = parse_args(env_args)

        reward_function = None
        if reward_function_spec and reward_function_spec != "None":
            module_name, function_name = reward_function_spec.rsplit('.', 1)
            reward_function = load_reward_function(module_name, function_name)

        env_register(env_id, reward_function=reward_function, env_kwargs=env_kwargs, model_kwargs=model_kwargs, reward_function_kwargs=reward_function_kwargs)
    else:
        raise AttributeError("Undefined command: " + cmd)

if __name__ == "__main__":
    main()