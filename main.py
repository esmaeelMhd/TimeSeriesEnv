from time_series_env.env_register import register_env_from_config
 
def main():
    # Path to the YAML configuration file
    config_file_path = './time_series_env/config.yaml'  
    # Registering the environment using the configuration file
    register_env_from_config(config_path=config_file_path)
    
if __name__ == "__main__":
    main()