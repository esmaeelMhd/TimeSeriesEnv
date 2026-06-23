# TimeSeriesEnv

TimeSeriesEnv is a Gymnasium-style reinforcement learning environment for control and evaluation on time-series data.

The environment sits between an agent and a forecasting model. At each step, the agent supplies action variables, the predictor estimates the next target variables from the recent sequence, and a custom reward function scores the result. This makes the project useful for testing control policies against learned sequence models and historical time-series data.

## What It Provides

- A continuous-control `TimeSeriesEnv` built on Gymnasium.
- Configurable action, exogenous, target, observation, and time-feature variables.
- CSV-based time-series loading with sorted datetime indexes and frequency checks.
- Optional simple, cyclic, spline, or one-hot time features.
- Pluggable predictors, PyTorch models, checkpoints, and scaler handling.
- Custom reward functions through the `BaseRewardFunction` interface.
- Episode controls for fixed or random starts and lengths.
- Optional logging, plotting, and early truncation.

## Project Layout

```text
.
|-- models.py                         # Example LSTM and encoder-decoder LSTM models
|-- setup.py                          # Package metadata and base dependencies
|-- test.py                           # Legacy integration sketch
|-- time_series_env/
|   |-- env.py                        # Core Gymnasium environment
|   |-- env_register.py               # Gymnasium environment registration helper
|   |-- time_series_data.py           # Dataset loading and variable grouping
|   |-- model_builder.py              # Model, checkpoint, scaler, and predictor wiring
|   |-- predictors.py                 # Generic model predictor wrappers
|   |-- scaler_handler.py             # MinMaxScaler load/fit/transform helpers
|   |-- base_reward.py                # Reward-function base class
|   |-- data_util.py                  # Time-feature utilities
|   |-- early_trunc.py                # Early truncation helper
|   `-- visualization.py              # Matplotlib plotting helpers
`-- LICENCE                           # Creative Commons BY-NC 4.0 license text
```

## Installation

```bash
git clone https://github.com/esmaeelMhd/TimeSeriesEnv.git
cd TimeSeriesEnv

python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the package in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The current `setup.py` lists the base dependencies, but the code also imports a few packages that are not declared there yet. Install them explicitly if needed:

```bash
python -m pip install gymnasium pyyaml scikit-learn joblib matplotlib
```

## Data Requirements

`TimeSeriesData` expects a CSV file with:

- a datetime column named `date`;
- numeric action, exogenous, target, and observation columns;
- a mostly uniform timestamp frequency;
- missing numeric values that can be safely forward-filled.

Example:

```csv
date,control_1,weather_1,target_1
2024-01-01 00:00:00,0.20,12.5,5.1
2024-01-01 01:00:00,0.25,12.8,5.0
2024-01-01 02:00:00,0.23,12.6,4.9
```

## Configuration Shape

Most projects will keep their environment, data, model, scaler, and reward settings in YAML. The registration helper expects these top-level sections:

```yaml
data_config:
  data_root_path: ./data
  data_name: time_series.csv
  index_col: date
  time_f: true
  has_time_f: false
  num_time_f: 0
  time_f_list: [hour, month, day_of_week]
  time_f_type: cyclic
  act_vars: [control_1]
  exog_vars: [weather_1]
  target_vars: [target_1]
  obs_vars: [target_vars, exog_vars, time_vars]

model_config:
  load_model: false
  scale_data: false
  chkpt_root_path: ./checkpoints
  chkpt_folder: ""
  chkpt_name: ""
  device: cpu

scaler_config:
  load_scaler: false
  scaler_root_path: ./scalers
  scaler_folder: ""
  scale_time_f: false
  data_scaler_name: data_scaler.pkl
  time_scaler_name: time_scaler.pkl

env_config:
  use_gpu: false
  mode: not_live
  seq_len: 24
  const_el: 168
  min_el: 24
  max_el: 168
  results_root_path: ./results
  min_max_bounds: true
  do_logging: false

reward_config:
  target: target_1
```

## Minimal Usage Pattern

You need three pieces before creating an environment:

1. a `TimeSeriesData` instance or equivalent data object;
2. a predictor, model, or `ModelBuilder` with a `predict(sequence)` method;
3. a reward function derived from `BaseRewardFunction`.

```python
import numpy as np

from time_series_env.base_reward import BaseRewardFunction
from time_series_env.env import TimeSeriesEnv
from time_series_env.time_series_data import TimeSeriesData


class LastValuePredictor:
    def predict(self, sequence):
        # Return one prediction for one target variable.
        return np.array([[sequence[-1, -1]]], dtype=np.float32)


class NegativeTargetReward(BaseRewardFunction):
    def calculate_reward(self, source="predicted"):
        if source == "actual":
            values = self.env.df[self.env.target_vars[0]]
            self.actual_rewards = -values.abs()
            return self.actual_rewards

        return -abs(float(self.env.get_targets()[0]))


data = TimeSeriesData(
    data_root_path="./data",
    data_name="time_series.csv",
    index_col="date",
    time_f=True,
    has_time_f=False,
    num_time_f=0,
    time_f_list=["hour", "month", "day_of_week"],
    time_f_type="cyclic",
    act_vars=["control_1"],
    exog_vars=["weather_1"],
    target_vars=["target_1"],
    obs_vars=["target_vars", "exog_vars", "time_vars"],
)

env = TimeSeriesEnv(
    use_gpu=False,
    data=data,
    predictor=LastValuePredictor(),
    reward_function=NegativeTargetReward(),
    seq_len=24,
    const_el=168,
    min_el=24,
    max_el=168,
    mode="print_status",
)

obs, info = env.reset()
done = False

while not done:
    action = env.choose_actual_action()
    obs, reward, done, truncated, info = env.step(action)

env.close()
```

## Registering With Gymnasium

Use `env_register` when you want to create the environment through `gymnasium.make`:

```python
import gymnasium as gym

from time_series_env.env_register import env_register


env_register(
    env_id="TimeSeriesEnv-v1",
    config=config,
    data=data,
    model=model,
    model_builder=model_builder,
    scaler_handler=scaler_handler,
    reward_function=reward_function,
)

env = gym.make("TimeSeriesEnv-v1")
```

## Repository Status Notes

- This checkout does not include sample data, checkpoints, or a ready-to-run `config.yaml`.
- `test.py`, `reg_env.py`, and `reg_time_series_env.py` reference project-specific modules such as `load_my_args`, `linear_models`, and `former_models` that are not tracked in this repository.
- `main.py` references `register_env_from_config`, but the current `time_series_env/env_register.py` exposes `env_register`.
- `time_series_env/__init__.py` imports `data_providers`, which is not present as a source file in this checkout.

Treat those files as legacy integration sketches until the missing local modules are restored or the examples are updated.

## Development

Useful checks while working on the project:

```bash
python -m compileall .
python -m unittest test.py
```

The unittest command currently depends on the missing local modules noted above.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. See [LICENCE](LICENCE) for the full text.
