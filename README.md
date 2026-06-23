# TimeSeriesEnv

TimeSeriesEnv is a Gymnasium-style reinforcement learning environment for time-series control problems.

The environment sits between an agent and a forecasting model. At each step, the agent supplies control actions, the predictor estimates the next target state from recent history, and a custom reward function scores the result. This gives you a reusable simulation loop for evaluating reinforcement learning policies against learned time-series dynamics.

## Why This Project Matters

This repository demonstrates practical ML engineering skills that are useful beyond a single experiment:

- custom Gymnasium environment design;
- time-series data preparation and feature engineering;
- clean separation between environment dynamics, predictors, and rewards;
- support for lightweight predictors as well as PyTorch model integration;
- reproducible examples and smoke tests that run without private data.

## Architecture

```text
historical CSV
    |
    v
TimeSeriesData ---> TimeSeriesEnv <--- agent action
                         |
                         v
                    predictor.predict(sequence)
                         |
                         v
                next target + reward function
                         |
                         v
               observation, reward, done, info
```

## What It Provides

- Continuous-control `TimeSeriesEnv` built on Gymnasium.
- Configurable action, exogenous, target, observation, and time-feature variables.
- CSV loading with sorted datetime indexes and frequency checks.
- Optional simple, cyclic, spline, or one-hot time features.
- Pluggable predictors, reward functions, scalers, and PyTorch model builders.
- Episode controls for fixed or random starts and lengths.
- Optional logging, plotting, and early truncation.
- A public synthetic-data example and smoke tests.

## Project Layout

```text
.
|-- examples/
|   `-- minimal_env.py                # End-to-end runnable synthetic example
|-- legacy/                           # Archived project-specific integration sketches
|-- tests/
|   `-- test_smoke.py                 # Import, reset/step, and registration tests
|-- time_series_env/
|   |-- env.py                        # Core Gymnasium environment
|   |-- env_register.py               # Gymnasium registration helper
|   |-- time_series_data.py           # Dataset loading and variable grouping
|   |-- model_builder.py              # Optional PyTorch model/checkpoint wiring
|   |-- predictors.py                 # Predictor wrappers
|   |-- scaler_handler.py             # MinMaxScaler helpers
|   |-- base_reward.py                # Reward-function base class
|   |-- data_util.py                  # Time-feature utilities
|   |-- early_trunc.py                # Early truncation helper
|   `-- visualization.py              # Matplotlib plotting helpers
|-- requirements.txt
|-- setup.py
|-- pyproject.toml
`-- LICENCE
```

## Installation

TimeSeriesEnv supports Python 3.9 and newer.

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

Install the core package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

For PyTorch model/checkpoint integration:

```bash
python -m pip install -e ".[torch]"
```

## Quick Start

Run the public synthetic-data example:

```bash
python examples/minimal_env.py
```

Expected output:

```text
Ran 12 steps using synthetic_timeseries.csv.
Final observation shape: (2,)
Total reward: ...
```

Run the smoke tests:

```bash
python -m unittest discover -s tests
```

## Data Requirements

`TimeSeriesData` expects a CSV file with:

- a datetime index column, usually `date`;
- numeric action, exogenous, target, and observation columns;
- a uniform timestamp frequency;
- missing numeric values that can be safely forward-filled.

Example:

```csv
date,control,outside_temp,target
2024-01-01 00:00:00,5.50,8.00,10.29
2024-01-01 01:00:00,5.48,8.40,10.35
2024-01-01 02:00:00,5.43,8.79,10.40
```

## Configuration Shape

Projects can keep environment, data, model, scaler, and reward settings in YAML. The registration helper expects these top-level sections:

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
  act_vars: [control]
  exog_vars: [outside_temp]
  target_vars: [target]
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
```

## Minimal Usage Pattern

You need three pieces before creating an environment:

1. a `TimeSeriesData` instance or compatible data object;
2. a predictor, model, or `ModelBuilder` with a `predict(sequence)` method;
3. a reward function derived from `BaseRewardFunction`.

```python
import numpy as np

from time_series_env import BaseRewardFunction, TimeSeriesData, TimeSeriesEnv


class PersistencePredictor:
    def predict(self, sequence):
        return np.array([[sequence[-1, -1]]], dtype=np.float32)


class NegativeTargetReward(BaseRewardFunction):
    def calculate_reward(self, source="predicted"):
        if source == "actual":
            self.actual_rewards = -self.env.df[self.env.target_vars[0]].abs()
            return self.actual_rewards

        return -abs(float(self.env.get_targets()[0]))


data = TimeSeriesData(
    data_root_path="./data",
    data_name="time_series.csv",
    index_col="date",
    time_f=False,
    has_time_f=False,
    num_time_f=0,
    act_vars=["control"],
    exog_vars=["outside_temp"],
    target_vars=["target"],
    obs_vars=["target", "outside_temp"],
)

env = TimeSeriesEnv(
    use_gpu=False,
    data=data,
    predictor=PersistencePredictor(),
    reward_function=NegativeTargetReward(),
    seq_len=24,
    const_el=168,
    min_el=24,
    max_el=168,
)
```

## Registering With Gymnasium

Use `env_register` when you want to create the environment through `gymnasium.make`:

```python
import gymnasium as gym

from time_series_env.env_register import env_register


env_register(
    env_id="TimeSeriesEnv-v1",
    data=data,
    predictor=predictor,
    reward_function=reward_function,
    config={"env_config": {"use_gpu": False, "seq_len": 24}},
)

env = gym.make("TimeSeriesEnv-v1")
```

## Legacy Scripts

Older project-specific integration scripts are archived under `legacy/`. They are retained for context but are not part of the public runnable path because they reference local configuration, private data, or untracked checkpoint utilities.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. See [LICENCE](LICENCE) for the full text.
