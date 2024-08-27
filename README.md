# TimeSeriesEnv

## Overview

**TimeSeriesEnv** is a customizable and scalable environment designed for training and evaluating machine learning models on time series data. The environment supports multi-GPU training, interactive visualizations, flexible configuration through YAML files, and robust logging for monitoring and debugging.

## Features

- **Multi-GPU Support**: Efficiently utilizes multiple GPUs for large-scale training tasks.
- **Interactive Visualizations**: Leverages Plotly for interactive, real-time plotting of model performance.
- **Flexible Configuration**: Easily configure the environment using YAML files or command-line arguments.
- **Robust Logging**: Comprehensive logging system for tracking model performance and debugging.
- **Automated Documentation**: Supports Sphinx for automatically generating project documentation.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration](#configuration)
4. [Usage](#usage)
5. [Examples](#examples)
6. [Testing](#testing)
7. [Logging](#logging)
8. [Contributing](#contributing)
9. [Citation](#citation)
10. [License](#license)

## Installation

### Prerequisites

Before setting up the `TimeSeriesEnv` project, ensure you have the following installed:

- **Python 3.8+**: The project is compatible with Python versions 3.8 and above.
- **pip**: Python's package installer, used to install dependencies.
- **Git**: Version control system to clone the repository.

### Clone the Repository

First, clone the repository from GitHub:

```bash
git clone https://github.com/esmaeelMoh/timeseriesenv.git
cd timeseriesenv
```

### Install Dependencies

Next, install the required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

This command installs all necessary packages listed in the `requirements.txt` file.

### Optional: Install Additional Tools

If you plan to work on the documentation or run tests, you may want to install additional tools like Sphinx for generating documentation:

```bash
pip install sphinx
```

## Quick Start

Here’s a brief guide to quickly get started with the `TimeSeriesEnv` project:

1. **Create a Configuration File**: 

Start by creating a configuration file based on the provided example:

```bash
cp config_example.yaml config.yaml
```

Edit `config.yaml` to suit your specific needs.

2. **Run the Environment**:

With your configuration file ready, you can run the environment using the following command:

```bash
python env_register.py --config=config.yaml
```

This command initializes the environment according to the settings in `config.yaml`.

3. **View the Logs**: 

Logs are saved in the `logs/` directory by default. You can view them with:

```bash
tail -f logs/environment.log
```

This command allows you to monitor the environment’s progress and debug any issues.

## Configuration

The environment is configured via a YAML file, which allows for flexible and easy-to-read configuration settings.

### Example Configuration (`config.yaml`)

Below is an example of what your `config.yaml` might look like:

```yaml
use_gpu: true

env_config:
  results_folder: './results'
  seq_len: 10
  num_envs: 1
  mode: 'not_live'

model_config:
  model_type: 'LSTM'
  data_root_path: './data'
  data_name: 'time_series_data.csv'
  index_col: 'date'
  model_name: 'MyModel'
  ctrl_vars: ['ctrl_var1', 'ctrl_var2']
  ind_vars: ['ind_var1', 'ind_var2']
  target_vars: ['target_var1']
  num_time_f: 5
  time_scaled: true
  checkpoint_path: './checkpoints'
  checkpoint: 'model_checkpoint.pth'

reward_function_config:
  target: 'T1_PO4'
  q_column: 'IN_Q'

reward_function_type: 'linear_pmt'
```

### Key Configuration Options

- `use_gpu`: Boolean flag to enable or disable GPU usage.
- `env_config`: Configurations for the environment, such as the results folder and sequence length.
- `model_config`: Model-specific settings, including model type, data paths, and checkpoints.
- `reward_function_config`: Settings for the reward function used during training.

## Usage

### Running the Environment

To start the environment using your custom configuration:

```bash
python env_register.py --config=config.yaml
```

### Command-Line Overrides

You can override any configuration option directly from the command line. For example, to disable GPU usage and change the sequence length:

```bash
python env_register.py --config=config.yaml --use_gpu=False --env_config.seq_len=20
```

### Multi-GPU Setup

If you have multiple GPUs available, the environment will automatically detect and utilize them. Ensure that `use_gpu` is set to `True` in your configuration.

## Examples

### Basic Example

Below is a basic example of how to use `TimeSeriesEnv` in a script:

```python
import gym

# Initialize the environment
env = gym.make('TimeSeriesEnv-v0')

# Set agent arguments
agent_args = {
    'obs_history': 1,
    'experiment': 1,
    'min_el': 10,
    'max_el': 360,
    'const_el': 10,
    'norm_values': True,
    'title': 'test',
    'agent_name': 'test_env_register'
}
env.set_agent_args(agent_args)

# Run the environment
state = env.reset()
done = False
while not done:
    action = env.choose_real_action()
    state, reward, done, _, info = env.step(action)
    print(f"State: {state}, Reward: {reward}, Done: {done}")

env.close()
```

This script demonstrates initializing the environment, setting agent arguments, running the environment through its steps, and finally closing the environment.

## Testing

### Unit Tests

Unit tests are included to verify that the environment works as expected. To run all the tests, use the following command:

```bash
python -m unittest discover tests
```

This command runs all tests in the `tests` directory, ensuring that your environment behaves correctly.

### Continuous Integration

This project is set up for continuous integration using GitHub Actions. This setup ensures that tests are automatically run on every commit, helping maintain code quality.

## Logging

Logging is an essential part of monitoring the environment's behavior and debugging any issues that arise.

### Default Logging Behavior

By default, logs are saved in the `logs/` directory. The logging system captures detailed information about the environment’s execution, such as initialization details, rewards received, and any errors encountered.

### Example Log Output

Here’s what a typical log file might look like:

```bash
2024-08-22 12:00:00 - INFO - Environment initialized with config: config.yaml
2024-08-22 12:00:01 - INFO - Episode started. Initial state: [0.1, 0.2, 0.3]
2024-08-22 12:00:02 - INFO - Reward Received: 1.0, Total Reward: 1.0
```

### Adjusting Log Levels

You can adjust the verbosity of the logs by modifying the log level in the `setup_logging` function. Available levels include `DEBUG`, `INFO`, `WARNING`, and `ERROR`.

## Contributing

We welcome and encourage contributions to `TimeSeriesEnv`! Here’s how you can contribute:

1. **Fork the Repository**: Create a fork of the project on GitHub to work on your changes.
2. **Create a Feature Branch**: Make a new branch for your feature or bugfix.
3. **Write Tests**: Ensure that your changes are covered by unit tests to maintain code quality.
4. **Submit a Pull Request**: When your changes are ready, submit a pull request with a clear description of the improvements or fixes you’ve made.

### Code Style

Please adhere to the PEP 8 style guide for Python code. We recommend using tools like `flake8` or `black` to automatically format your code and ensure consistency.

## Citation

If you use `TimeSeriesEnv` in your research or projects, please cite our paper:

```css
@article{YourLastName2024TimeSeriesEnv,
  title={TimeSeriesEnv: A Scalable Environment for Time Series Analysis},
  author={Your Name and Co-Author Name},
  journal={arXiv preprint arXiv:2401.12345},
  year={2024},
  url={https://arxiv.org/abs/2401.12345}
}
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. 

You are free to:

- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material.

Under the following terms:

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial**: You may not use the material for commercial purposes.

For more details, please see the full license in the [LICENSE](LICENSE.md) file located in this repository, or visit [https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/).
