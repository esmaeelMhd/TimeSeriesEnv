from abc import ABC, abstractmethod

class BaseRewardFunction(ABC):
    def __init__(self):
        self.env = None

    def set_env(self, env):
        """
        Set the environment reference in the reward function.
        
        Parameters:
        - env: The environment object.
        """
        self.env = env
        self.data = self.env.data

    @abstractmethod
    def calculate_reward(self, **kwargs):
        """
        Abstract method to calculate the reward.
        Must be implemented by the custom reward function.
        
        Parameters:
        - kwargs: Additional keyword arguments.
        
        Returns:
        - reward: The calculated reward.
        """
        pass
