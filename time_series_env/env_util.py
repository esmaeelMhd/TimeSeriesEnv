import os
import sys
import pickle
import functools
import inspect
from importlib import import_module
from typing import TypeVar, Union, Type, Dict

import numpy as np
import pandas as pd
import yaml

# Type Variable
T = TypeVar('T')  # helps with type inference in some editors

# === args ====================================================================        
def load_args(args_path, folder_name):
    """Loads the args file of the model."""
    file_path = os.path.join(args_path, folder_name)
    try:
        with open(os.path.join(file_path, 'args.pkl'), 'rb') as file:
            args = pickle.load(file)
    except OSError as e:
        print(f"Unable to open {file_path}: {e}", file=sys.stderr)
        args=None
    
    return args

def args_mapping(args, map_dict):
    # Assign variables based on the mapping
    for possible_name, standardized_name in map_dict.items():
        if hasattr(args, possible_name):
            # Assign to a local variable dynamically
            setattr(args, standardized_name, getattr(args, possible_name))
    
    return args

class Args:
    def __init__(self, **kwargs):
        # Loop through each key-value pair in kwargs and set them as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

# === partial =================================================================
def default():
    raise ValueError("This is a dummy function and not meant to be called.")


def partial(func: Type[T] = default, *args, **kwargs) -> Union[T, Type[T]]:
    """Like `functools.partial`, except if used as a keyword argument for another `partial` and no function is supplied.
     Then, the outer `partial` will insert the appropriate default value as the function. """

    if func is not default:
        for k, v in kwargs.items():
            if isinstance(v, functools.partial) and v.func is default:
                kwargs[k] = partial(inspect.signature(func).parameters[k].default, *v.args, **v.keywords)
    return functools.partial(func, *args, **kwargs)


FKEY = '+'

def partial_to_dict(p: functools.partial, version="3"):
    assert not p.args, "So far only keyword arguments are supported, here"
    fields = {k: v.default for k, v in inspect.signature(p.func).parameters.items()}
    fields = {k: v for k, v in fields.items() if v is not inspect.Parameter.empty}
    diff = p.keywords.keys() - fields.keys()
    assert not diff, f"There are invalid keywords present: {diff}"
    fields.update(p.keywords)
    nested = {k: partial_to_dict(partial(v), version="") for k, v in fields.items() if callable(v)}
    simple = {k: v for k, v in fields.items() if k not in nested}
    output = {FKEY: p.func.__module__ + ":" + p.func.__qualname__, **simple, **nested}
    return dict(output, __format_version__=version) if version else output


def partial_from_dict(d: dict):
    d = d.copy()
    assert d.pop("__format_version__", "3") == "3"
    d = {k: partial_from_dict(v) if isinstance(v, dict) and FKEY in v else v for k, v in d.items()}
    func = get_class_or_function(d.pop(FKEY) or "dcac_python.util:default")
    return partial(func, **d)


def get_class_or_function(func):
    module, name = func.split(":")
    return getattr(import_module(module), name)


def partial_from_args(func: Union[str, callable], kwargs: Dict[str, str]):
    # print(func, kwargs)  # useful to visualize the parsing process
    func = get_class_or_function(func) if isinstance(func, str) else func
    keys = {k.split('.')[0] for k in kwargs}
    keywords = {}
    for key in keys:
        params = inspect.signature(func).parameters
        assert key in params, f"'{key}' is not a valid parameter of {func}. Valid parameters are {tuple(params.keys())}."
        param = params[key]
        value = kwargs.get(key, param.default)
        if param.annotation is type:
            sub_keywords = {k.split('.', 1)[1]: v for k, v in kwargs.items() if k.startswith(key + '.')}
            keywords[key] = partial_from_args(value, sub_keywords)
        elif param.annotation is bool:
            keywords[key] = bool(eval(value))  # because bool('False') will evaluate to True (it's a non-empty string).
        else:
            keywords[key] = param.annotation(value)
    return partial(func, **keywords)

def get_class_attributes(cls):
    return [param.name for param in inspect.signature(cls).parameters.values()]

def filter_kwargs(kwargs, valid_keys):
    return {k: v for k, v in kwargs.items() if k in valid_keys}

# === dataset =================================================================
def load_dataset(data_root_path, data_name):
    """Loads the dataset used in training the model."""
    dataset_path = os.path.join(data_root_path, data_name)
    df = pd.read_csv(dataset_path, index_col=["date"], parse_dates=["date"])
    df.sort_index(inplace=True)
    df = df.astype('float32').ffill()
    
    return df

def add_time_specs(df):
    """Embeds cyclical time features into the DataFrame based on its index."""
    def generate_cyclical_features(df, col_name, period, start_num=0):
        sin_col = np.sin(2 * np.pi * (df[col_name] - start_num) / period)
        cos_col = np.cos(2 * np.pi * (df[col_name] - start_num) / period)
        df[f'sin_{col_name}'] = sin_col
        df[f'cos_{col_name}'] = cos_col
        return df.drop(columns=[col_name])
    
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df = generate_cyclical_features(df, 'hour', 24)
    df = generate_cyclical_features(df, 'month', 12, 1)  # Starts at 1
    df = generate_cyclical_features(df, 'day_of_week', 7)
    
    return df


    


