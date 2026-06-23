"""Run a minimal TimeSeriesEnv episode with synthetic data."""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from time_series_env import BaseRewardFunction, TimeSeriesData, TimeSeriesEnv


class PersistencePredictor:
    """Tiny predictor used only for the public smoke example."""

    def predict(self, sequence):
        action = sequence[-1, 0]
        outside_temp = sequence[-1, 1]
        target = sequence[-1, 2]
        next_target = 0.75 * target + 0.15 * action + 0.10 * outside_temp
        return np.array([[next_target]], dtype=np.float32)


class SetpointReward(BaseRewardFunction):
    """Reward that prefers targets near a setpoint and mild control effort."""

    def __init__(self, setpoint=10.0):
        super().__init__()
        self.setpoint = setpoint

    def calculate_reward(self, source="predicted"):
        target_name = self.env.target_vars[0]

        if source == "actual":
            self.actual_rewards = -(self.env.df[target_name] - self.setpoint).abs()
            return self.actual_rewards

        target_error = abs(float(self.env.get_targets()[0]) - self.setpoint)
        action_effort = abs(float(self.env.get_actions()[0]))
        return -(target_error + 0.05 * action_effort)


def write_synthetic_data(folder):
    periods = 96
    dates = pd.date_range("2024-01-01", periods=periods, freq="h")
    phase = np.linspace(0, 4 * np.pi, periods)
    control = 4.0 + 1.5 * np.cos(phase)
    outside_temp = 8.0 + 3.0 * np.sin(phase)
    target = 9.5 + 0.4 * np.sin(phase + 0.5) + 0.15 * control

    df = pd.DataFrame(
        {
            "date": dates,
            "control": control,
            "outside_temp": outside_temp,
            "target": target,
        }
    )
    path = Path(folder) / "synthetic_timeseries.csv"
    df.to_csv(path, index=False)
    return path


def build_env(data_folder):
    data = TimeSeriesData(
        data_root_path=str(data_folder),
        data_name="synthetic_timeseries.csv",
        index_col="date",
        time_f=False,
        has_time_f=False,
        num_time_f=0,
        act_vars=["control"],
        exog_vars=["outside_temp"],
        target_vars=["target"],
        obs_vars=["target", "outside_temp"],
    )

    return TimeSeriesEnv(
        use_gpu=False,
        data=data,
        predictor=PersistencePredictor(),
        reward_function=SetpointReward(setpoint=10.0),
        seq_len=8,
        const_el=12,
        min_el=12,
        max_el=13,
        mode="not_live",
    )


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = write_synthetic_data(temp_dir)
        env = build_env(temp_dir)

        obs, info = env.reset(seed=7)
        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = env.choose_actual_action()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        env.close()

    print(f"Ran {steps} steps using {data_path.name}.")
    print(f"Final observation shape: {obs.shape}")
    print(f"Total reward: {total_reward:.3f}")


if __name__ == "__main__":
    main()
