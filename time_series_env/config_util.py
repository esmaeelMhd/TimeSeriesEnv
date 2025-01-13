import yaml
import argparse

# Utility function to convert an object to a dictionary
def convert_to_dict(obj):
    return vars(obj) if hasattr(obj, '__dict__') else obj

# === yaml ====================================================================        
def load_config(config_path: str = 'config.yaml') -> dict:
    """Loads configuration parameters from a YAML file."""
    if config_path is None:
        raise ValueError("A configuration file path must be provided.")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# === argparse ================================================================        
def parse_args() -> dict:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Time Series Environment Configuration")

    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--use_gpu', type=bool, help='Whether to use GPU or not')
    parser.add_argument('--data_root_path', type=str, help='Root path of the data')
    parser.add_argument('--data_name', type=str, help='Name of the data file')
    parser.add_argument('--index_col', type=str, help='Index column name in the data file')

    args = parser.parse_args()

    return vars(args)