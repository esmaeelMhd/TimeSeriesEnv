import gym
import numpy as np

class NormalizeObservationWrapper(gym.Wrapper):
    """
    A wrapper to normalize observations dynamically to zero mean and unit variance.

    Attributes:
    -----------
    mean : np.ndarray
        Running mean of the observations.
    var : np.ndarray
        Running variance of the observations.
    count : float
        Count of observations for mean and variance calculation.
    observation_space : gym.spaces.Box
        Modified observation space with infinite bounds for normalization.

    Methods:
    --------
    reset(**kwargs):
        Resets the environment and normalizes the initial observation.
    step(action):
        Takes a step in the environment and normalizes the observation.
    normalize(observation):
        Normalizes a given observation using current mean and variance.
    """
    def __init__(self, env):
        """
        Initialize the wrapper with an environment.

        Parameters:
        -----------
        env : gym.Env
            The environment to wrap.
        """
        super().__init__(env)
        self.mean = np.zeros(env.observation_space.shape)
        self.var = np.ones(env.observation_space.shape)
        self.count = 1e-4  # Avoid division by zero initially
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def reset(self, **kwargs):
        """
        Reset the environment and normalize the initial observation.

        Returns:
        --------
        np.ndarray
            Normalized initial observation.
        """
        observation = self.env.reset(**kwargs)
        return self.normalize(observation)

    def step(self, action):
        """
        Step through the environment and normalize the observation.

        Returns:
        --------
        tuple
            Normalized observation, reward, done flag, and info.
        """
        observation, reward, done, info = self.env.step(action)
        return self.normalize(observation), reward, done, info

    def normalize(self, observation):
        """
        Normalize the observation using running statistics.

        Returns:
        --------
        np.ndarray
            Normalized observation.
        """
        self.count += 1
        delta = observation - self.mean
        self.mean += delta / self.count
        delta2 = observation - self.mean
        self.var += delta * delta2

        std = np.sqrt(self.var / self.count)
        return (observation - self.mean) / (std + 1e-8)
