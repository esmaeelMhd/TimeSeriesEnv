import tempfile
import unittest

import gymnasium as gym
import numpy as np

from examples.minimal_env import (
    PersistencePredictor,
    SetpointReward,
    build_env,
    write_synthetic_data,
)
from time_series_env import BaseRewardFunction, TimeSeriesData, TimeSeriesEnv
from time_series_env.env_register import env_register


class TimeSeriesEnvSmokeTest(unittest.TestCase):
    def test_public_imports(self):
        self.assertIsNotNone(TimeSeriesEnv)
        self.assertIsNotNone(TimeSeriesData)
        self.assertIsNotNone(BaseRewardFunction)

    def test_reset_and_step_with_synthetic_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_synthetic_data(temp_dir)
            env = build_env(temp_dir)

            obs, info = env.reset(seed=11)
            self.assertEqual(obs.shape, (2,))
            self.assertEqual(info["round"], 0)

            action = env.choose_actual_action()
            next_obs, reward, done, truncated, info = env.step(action)

            self.assertEqual(next_obs.shape, (2,))
            self.assertIsInstance(float(reward), float)
            self.assertFalse(done)
            self.assertFalse(truncated)
            self.assertEqual(info["round"], 1)

            env.close()

    def test_gymnasium_registration_with_predictor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            write_synthetic_data(temp_dir)
            env = build_env(temp_dir)
            data = env.get_data()

            env_register(
                env_id="TimeSeriesEnvSmoke-v0",
                data=data,
                predictor=PersistencePredictor(),
                reward_function=SetpointReward(setpoint=10.0),
                config={
                    "env_config": {
                        "use_gpu": False,
                        "seq_len": 8,
                        "const_el": 12,
                        "min_el": 12,
                        "max_el": 13,
                        "mode": "not_live",
                    }
                },
            )

            registered_env = gym.make("TimeSeriesEnvSmoke-v0")
            obs, info = registered_env.reset(seed=5)
            action = registered_env.unwrapped.choose_actual_action()
            next_obs, reward, done, truncated, info = registered_env.step(action)

            self.assertIsInstance(next_obs, np.ndarray)
            self.assertEqual(info["round"], 1)
            self.assertFalse(done)
            self.assertFalse(truncated)

            registered_env.close()


if __name__ == "__main__":
    unittest.main()
