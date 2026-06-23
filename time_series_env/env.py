"""
Created on April 26, 2024

Author:
    Esmaeel Mohammadi
    Email: esm@kruger.dk, esmo@bio.aau.dk
    GitHub: https://github.com/esmaeelMhd

Description:
    This script creates a Deep Reinforcement Learning (DRL) environment for Time Series models.
    The supported models include DLinear, Transformer, Informer, and Autoformer.

    The script performs the following tasks:
        1. Preprocesses the data.
        2. Converts data to tensors for the model.
        3. Loads the model and its parameters.
        4. Uses the environment for generating the next state.
        5. Defines continuous action and observation spaces.

Usage:
    This environment can be used for training and evaluating DRL models on time series data.
"""
import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from functools import lru_cache
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple, Dict, Union
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias for `np.bool_`")
import random

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import logging
from logging.handlers import TimedRotatingFileHandler
import gzip

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from time_series_env.time_series_data import TimeSeriesData
from time_series_env.env_config import AgentArgs
from time_series_env.data_preprocessing import check_frequency_uniformity
from time_series_env.model_management import setup_device
from time_series_env.base_reward import BaseRewardFunction
from time_series_env.visualization import initialize_plots, finalize_and_save_plot, plot_live, plot_not_live
from time_series_env.early_trunc import EarlyTrunc

MODE_LIVE = 'live'
MODE_NOT_LIVE = 'not_live'
MODE_PRINT_STATUS = 'print_status'

SOURCE_ACTUAL = 'actual'
SOURCE_PREDICTED = 'predicted'

LOG_LEVEL_INFO = 'info'
LOG_LEVEL_DEBUG = 'debug'

@dataclass(eq=False)
class TimeSeriesEnv(gym.Env):
    """
    A reinforcement learning environment for time series forecasting and related tasks.
    
    This environment facilitates the training and evaluation of reinforcement learning models 
    on time series data, supporting various configurations, logging, visualization, and handling 
    different variable types. It includes additional configurations for bounds, truncation, and evaluation settings.

    Attributes:
        use_gpu (bool): Whether to use GPU for computation.
        reward_function (BaseRewardFunction): The reward function used in the environment.
        
        data_config (Optional[Dict[str, Any]]): Configuration dictionary for the time series data, including parameters for data preparation.
        act_vars (List[str]): List of active variables in the dataset.
        exog_vars (List[str]): List of exogenous variables in the dataset.
        target_vars (List[str]): List of target variables in the dataset.
        obs_vars (List[str]): List of observed variables in the dataset.
        
        min_max_bounds (bool): Whether to enforce min-max bounds for variables.
        act_bounds_low (Union[np.ndarray, float]): Lower bounds for action variables.
        act_bounds_high (Union[np.ndarray, float]): Upper bounds for action variables.
        obs_bounds_low (Union[np.ndarray, float]): Lower bounds for observation variables.
        obs_bounds_high (Union[np.ndarray, float]): Upper bounds for observation variables.
        
        data (TimeSeriesData): The time series data object.
        model (Optional[Any]): The model used for prediction.
        model_builder (Optional[Any]): The model builder object for constructing models.
        predictor (Optional[Any]): The predictor object used for making predictions.
        
        targets (np.ndarray): Array of target values in the environment.
        actual_targets (np.ndarray): Array of actual target values for comparison.
        actions (np.ndarray): Array of action values taken by the agent.
        actual_actions (np.ndarray): Array of actual action values for comparison.
        observations (np.ndarray): Array of observation values in the environment.
        rewards (np.ndarray): Array of reward values obtained in the environment.
        actual_rewards (np.ndarray): Array of actual reward values for comparison.
        
        do_logging (bool): Whether to enable logging.
        log_file (str): Path to the log file.
        log_level (int): Logging level.
        
        colors (List[str]): List of colors for visualization.
        figure (Optional[Any]): The figure object for visualization.
        figsize (Tuple[float, float]): Default figure size for plotting.
        figures_folder (str): Folder path to save visualization results.
        
        mode (str): Mode of the environment, e.g., 'not_live', 'live'.
        seq_len (int): Length of the input sequence for the environment.
        eval_len (int): Evaluation length, specifying the length of evaluation periods.
        env_idx (int): Index of the environment instance for multi-environment setups.
        flag (str): Indicator for training or testing mode (e.g., 'train').
        
        agent_args (dict): Arguments for initializing the agent in the environment.
        experiment (int): Experiment identifier for tracking environment runs.
        agent_name (str): Name of the agent used in the environment.
        results_root_path (str): Root directory for storing results.
        const_el (int): Constant episode length.
        min_el (int): Minimum allowable episode length.
        max_el (int): Maximum allowable episode length.
        n_eval (int): Number of evaluation episodes.
        eval_starts (Union[list, str]): Starting points for evaluation episodes.
        
        do_early_trunc (bool): Whether to enable early truncation of episodes.
        patience (int): Patience parameter for early truncation.
        verbose (int): Verbosity level for debugging.
        delta (float): Threshold for monitoring target limits.
        target_limit_low (float): Lower limit for target values.
        target_limit_high (float): Upper limit for target values.
        act_limit_low (float): Lower limit for action values.
        act_limit_high (float): Upper limit for action values.

        obs_dim (int): Dimension of the observation space.
        act_dim (int): Dimension of the action space.
        obs_idxs (List[int]): Indices of observation variables in the dataset.
        act_idxs (List[int]): Indices of action variables in the dataset.
        target_idxs (List[int]): Indices of target variables in the dataset.
        exog_idxs (List[int]): Indices of exogenous variables in the dataset.
        time_idxs (List[int]): Indices of time variables in the dataset.
        
        min_obs_vars (List[float]): Minimum values for observation variables.
        max_obs_vars (List[float]): Maximum values for observation variables.
        min_act_vars (List[float]): Minimum values for action variables.
        max_act_vars (List[float]): Maximum values for action variables.
        min_target_vars (List[float]): Minimum values for target variables.
        max_target_vars (List[float]): Maximum values for target variables.
        
        state (np.ndarray): Current state of the environment.
        actual_state (np.ndarray): Actual state of the environment for comparison.
        actual_sequence (np.ndarray): Actual sequence of states for comparison.
        actual_observations (np.ndarray): Actual observations for comparison.
        actual_rewards_ep (np.ndarray): Actual rewards for the current episode.
        sequence (np.ndarray): Sequence of states in the environment.
        df (pd.DataFrame): DataFrame containing the time series data.
        columns (pd.Index): Columns of the DataFrame.
        freq (pd.Timedelta): Frequency of the time series data.
        freq_min (float): Frequency of the time series data in minutes.
        start_date (pd.Timestamp): Start date of the current episode.
        ep_len (int): Length of the current episode.
        ep_start (int): Start index of the current episode.
        info (dict): Additional information about the environment state.
    """
    # Device
    use_gpu: bool = True
    reward_function: Optional[Any] = None

    # Data configuration and variables
    data_config: Optional[Dict[str, Any]] = None
    act_vars: List[str] = field(default_factory=list)
    exog_vars: List[str] = field(default_factory=list)
    target_vars: List[str] = field(default_factory=list)
    obs_vars: List[str] = field(default_factory=list)
    
    # Spaces and bounds
    min_max_bounds: bool = False
    act_bounds_low: Union[np.ndarray, float] = field(default_factory=lambda: 0.0)
    act_bounds_high: Union[np.ndarray, float] = field(default_factory=lambda: np.inf)
    obs_bounds_low: Union[np.ndarray, float] = field(default_factory=lambda: 0.0)
    obs_bounds_high: Union[np.ndarray, float] = field(default_factory=lambda: np.inf)
    
    # Data objects
    data: Optional[Any] = None
    model: Optional[Any] = None
    model_builder: Optional[Any] = None
    predictor: Optional[Any] = None
    
    # Environment state
    targets: np.ndarray = field(default_factory=lambda: np.empty((0,)))  
    actual_targets: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    actions: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    actual_actions: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    observations: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    rewards: np.ndarray = field(default_factory=lambda: np.array([]))
    actual_rewards: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Logging configuration
    do_logging: bool = False
    log_file: str = 'logs/environment.log'
    log_level: int = logging.INFO
    
    # Plotting configuration
    colors: List[str] = field(default_factory=lambda: list(mcolors.TABLEAU_COLORS.values()))
    figure: Optional[Any] = None
    figsize: Tuple[float, float] = (6.5, 4.5)
    figures_folder: str = 'results'
    
    # Environment settings
    mode: str = 'not_live'
    seq_len: int = 1
    eval_len: int = 720
    env_idx: int = 1
    flag: str = 'train'
    
    # Training settings
    agent_args: dict = field(default_factory=dict)
    experiment: int = 1
    agent_name: str = 'Agent'
    results_root_path: str = './results'
    const_el: int = 720
    min_el: int = 60
    max_el: int = 1440
    n_eval: int = 1
    eval_starts: Union[list, str] = field(default_factory=list)
    
    # Early truncation
    do_early_trunc: bool = False
    patience: int = None 
    verbose: int = 0 
    delta: float = None 
    target_limit_low: float = None
    target_limit_high: float = None
    act_limit_low: float = None
    act_limit_high: float = None

    render_mode: str = 'human'
    
    def __post_init__(self):
        self._initialize_internal_state()
        self._initialize_device_and_predictor()
        self._initialize_reward_function()
        self._initialize_data_and_spaces()
        
        self.set_mode(self.mode)
        if self.mode != MODE_LIVE:
            plt.ioff()
        
        self._initialize_logging()
        self._initialize_evaluation()
        self._initialize_early_truncation()
    
    def _initialize_internal_state(self) -> None:
        """Initializes the internal state and variables."""
        self.obs_dim = 0
        self.act_dim = 0
        self.obs_idxs = []
        self.act_idxs = []
        self.target_idxs = []
        self.exog_idxs = []
        self.time_idxs = []
    
        self.min_obs_vars = []
        self.max_obs_vars = []
        self.min_act_vars = []
        self.max_act_vars = []
        self.min_target_vars = []
        self.max_target_vars = []
    
        self.state = np.array([])
        self.actual_state = np.array([])
        self.actual_sequence = np.array([])
        self.actual_observations = np.array([])
        self.actual_rewards_ep = np.array([])
        self.sequence = np.array([])
        self.df = pd.DataFrame()
        self.columns = None
        self.freq = pd.Timedelta(0)
        self.freq_min = 0.0
        self.start_date = pd.Timestamp(0)
        self.ep_len = 0
        self.ep_start = 0
        self.info = {}
        self.round = 0
        self.done = False
        self.truncated = False
    
    def _initialize_evaluation(self) -> None:
        """Initialize evaluation settings if in eval mode."""
        if self.flag != 'eval':
            return
        
        if self.eval_starts is None:
            # Create a list of possible evaluation episode starts
            ep_locs = range(0, len(self.df) - self.seq_len - self.eval_len + 1, self.eval_len)
            self.eval_ep_locs = random.sample(list(ep_locs), self.n_eval)
            self.eval_starts = self.eval_ep_locs
        else:
            # Convert eval_starts strings to DataFrame indices
            timezone = self.df.index.tz
            if isinstance(self.eval_starts, str):
                self.eval_starts = [self.eval_starts]
            eval_datetimes = [pd.to_datetime(d).tz_localize(timezone) for d in self.eval_starts]
            self.eval_starts = [self.df.index.get_loc(e) for e in eval_datetimes]
            self.n_eval = len(self.eval_starts)
        
        self.eval_num = 0

    def _initialize_device_and_predictor(self) -> None:
        """Set up computation device and initialize the predictor."""
        self.device, self.device_ids, self.use_multi_gpu = setup_device(self.use_gpu)
        self.episode_num = 0
        
        # Initialize predictor
        if self.predictor is not None:
            pass  # predictor already provided
        elif self.model_builder is not None:
            self.model_builder.create_predictor()
            self.predictor = self.model_builder.predictor
        elif self.model is not None:
            self.model = self.model.to(self.device)
            self.predictor = self.model
        else:
            raise ValueError('One of `predictor`, `model_builder`, or `model` must be provided.')
        
        if not hasattr(self.predictor, 'predict'):
            raise ValueError("Predictor object must have a `predict` method.")
    
    def _initialize_reward_function(self) -> None:
        """Set up the reward function, ensuring it is bound to this environment."""
        if self.reward_function is None:
            raise ValueError("A `reward_function` must be provided.")
        
        if callable(self.reward_function):
            self.reward_function = self.reward_function()
        self.reward_function.set_env(self)
    
    def _initialize_early_truncation(self) -> None:
        """Configure early truncation logic if enabled."""
        if not self.do_early_trunc:
            return
        
        patience = self.patience
        if patience is None:
            # Base patience on experiment if not provided explicitly
            if self.experiment in [1, 3]:
                patience = int(0.1*self.const_el)
            elif self.experiment in [2, 4]:
                patience = int(0.05*self.const_el)
            else:
                # Default fallback
                patience = int(0.1*self.const_el)
        
        verbose = 0 if self.verbose is None else self.verbose
        delta = 0.01 if self.delta is None else self.delta
        target_limit_low = np.mean(self.min_target_vars) if self.target_limit_low is None else self.target_limit_low
        target_limit_high = np.mean(self.max_target_vars) if self.target_limit_high is None else self.target_limit_high
        act_limit_low = np.mean(self.min_act_vars) if self.act_limit_low is None else self.act_limit_low
        act_limit_high = np.mean(self.max_act_vars) if self.act_limit_high is None else self.act_limit_high
        
        self.early_trunc = EarlyTrunc(
            patience=patience,
            verbose=verbose,
            delta=delta, 
            target_limit_low=target_limit_low,
            target_limit_high=target_limit_high,
            act_limit_low=act_limit_low,
            act_limit_high=act_limit_high
        )
    
    def _initialize_logging(self) -> None:
        """Sets up logging if enabled."""
        if not self.do_logging:
            return
        
        # Use a dedicated logger for this environment
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.log_level)
        
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        handler = TimedRotatingFileHandler(self.log_file, when='midnight', interval=1, backupCount=7)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handlers to the logger
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
    
    def _initialize_data_and_spaces(self) -> None:
        """Load and prepare the data, and initialize observation/action spaces."""
        self._load_and_prepare_data()
        self._initialize_data()
        self._calculate_min_max()
        self._setup_spaces()

    def _load_and_prepare_data(self) -> None:
        """Loads and prepares the data for the environment."""
        if hasattr(self, 'data') and self.data is not None:
            self.df: pd.DataFrame = self.data.df
        elif self.model_builder is not None:
            self.data = self.model_builder.data
            self.df = self.model_builder.df
        else:
            self.data = TimeSeriesData(**self.data_config)
            self.df = self.data.df
    
        # Ensuring only numeric columns are converted
        if not self.df.select_dtypes(include=[np.number]).empty:
            self.df = self.df.astype('float32')
        else:
            raise ValueError("DataFrame must contain numeric values for conversion to 'float32'.")
            
    def _initialize_data(self) -> None:
        """Prepares the time series data by checking uniformity and setting up variables."""
        check_frequency_uniformity(self.df)
        if not self.df.select_dtypes(include=[np.number]).empty:
            self.df = self.df.astype('float32')
        else:
            raise ValueError("DataFrame must contain numeric values for conversion to 'float32'.")
            
        self.columns = self.df.columns
        self.freq = self.df.index.to_series().diff().dropna().mode()[0]
        self.freq_min = self.freq.total_seconds() / 60
        
        self.act_vars = self.data.act_vars
        self.exog_vars = self.data.exog_vars
        self.target_vars = self.data.target_vars
        self.obs_vars = self.data.obs_vars
        self.time_vars = self.data.time_vars
        
        self.act_names = self.data.act_names
        self.exog_names = self.data.exog_names
        self.target_names = self.data.target_names
        
        self._setup_vars()
        
        if self.reward_function is not None:
            self.reward_function.calculate_reward(source=SOURCE_ACTUAL)
            self.actual_rewards = self.reward_function.actual_rewards
        else:
            self.actual_rewards = np.array([])
    
    def _setup_vars(self) -> None:
        """Sets up the different groups of variables in the environment."""
        # Correct the exogenous vars if act_vars or target_vars change
        if len(self.exog_vars) != len(self.columns) - (len(self.act_vars) + len(self.target_vars) + self.data.num_time_f):
            self.exog_vars = [col for col in self.columns[:len(self.columns)-self.data.num_time_f] if col not in self.act_vars and col not in self.target_vars]
        
        # Set the number of time features and actions
        self.n_act = len(self.act_vars)
        self.n_exog = len(self.exog_vars)
        self.n_targets = len(self.target_vars)
        self.n_obs = len(self.obs_vars)
    
        col_get_loc = self.columns.get_loc
        self.target_idxs = sorted([col_get_loc(col) for col in self.target_vars])
        self.obs_idxs = sorted([col_get_loc(col) for col in self.obs_vars])
        self.act_idxs = sorted([col_get_loc(col) for col in self.act_vars])
        self.exog_idxs = sorted([col_get_loc(col) for col in self.exog_vars])
        ### TODO: make sure that if df does not have time features, num_time_f is 0
        self.time_idxs = sorted([col for col in range(len(self.columns)-self.data.num_time_f, len(self.columns))])
    
    def _setup_vars_names(self) -> None:
        """Sets the names for vars, when some of the variables change"""
        self.act_names = self.columns[self.act_idxs]
        self.exog_names = self.columns[self.exog_idxs]
        self.target_names = self.columns[self.target_idxs]
    
    def _setup_spaces(self) -> None:
        """Defines the action and observation space."""
        self.obs_dim = len(self.obs_vars)
        self.act_dim = len(self.act_vars)
        
        self.observation_space = self._create_space(self.obs_bounds_low, self.obs_bounds_high, self.obs_dim)
        self.action_space = self._create_space(self.act_bounds_low, self.act_bounds_high, self.act_dim)
        
    def _create_space(self, low: Union[List[float], Tuple[float], np.ndarray, float], 
                      high: Union[List[float], Tuple[float], np.ndarray, float], dim: int) -> spaces.Box:
        """
        Create a Gym Box space with the given low and high bounds.
        """
        low_array = self._convert_to_array(low, dim)
        high_array = self._convert_to_array(high, dim)

        return spaces.Box(low=low_array, high=high_array, shape=(dim,), dtype=np.float32)
    
    def _convert_to_array(self, bounds: Union[List[float], Tuple[float], np.ndarray, float], dim: int) -> np.ndarray:
        """
        Convert bounds to a numpy array if it's not already.
        If a float is given, return a numpy array with the float repeated `dim` times.
        """
        if isinstance(bounds, (list, tuple)):
            bounds = np.array(bounds, dtype=np.float32)
        elif isinstance(bounds, (float, int)):
            bounds = np.full(dim, bounds, dtype=np.float32)
        elif not isinstance(bounds, np.ndarray):
            raise TypeError("Bounds must be a list, tuple, numpy array, int or float.")
        
        if bounds.shape[0] != dim:
            raise ValueError(f"Bounds dimension {bounds.shape[0]} does not match expected dimension {dim}.")
        
        return bounds

    # @lru_cache(maxsize=10)
    def _calculate_min_max(self) -> None:
        """Calculates the minimum and maximum values for targets and control variables."""
        self.min_obs_vars, self.max_obs_vars = self._get_min_max_vars(self.df, self.obs_vars)
        self.min_act_vars, self.max_act_vars = self._get_min_max_vars(self.df, self.act_vars)        
        self.min_target_vars, self.max_target_vars = self._get_min_max_vars(self.df, self.target_vars)
                
        if self.min_max_bounds:
            self.obs_bounds_low = self.min_obs_vars
            self.obs_bounds_high = self.max_obs_vars
            self.act_bounds_low = self.min_act_vars
            self.act_bounds_high = self.max_act_vars

    def _get_min_max_vars(self, df: pd.DataFrame, variables: List[str]) -> Tuple[List[float], List[float]]:
        """Helper function to get min and max values for a list of variables."""
        min_values: List[float] = df[variables].min().tolist()
        max_values: List[float] = df[variables].max().tolist()
        return min_values, max_values
    
    def _initialize_early_trunc(self) -> None:
        """Sets the early truncation for the environment."""
        if self.patience is None:
            if self.experiment == 1 or self.experiment == 3:
                patience = int(0.1*self.const_el)
            elif self.experiment == 2 or self.experiment == 4:
                patience = int(0.05*self.const_el)
        else:
            patience = self.patience
        
        verbose = 0 if self.verbose is None else self.verbose
        delta = 0.01 if self.delta is None else self.delta
        target_limit_low = np.mean(self.min_target_vars) if self.target_limit_low is None else self.target_limit_low
        target_limit_high = np.mean(self.max_target_vars) if self.target_limit_high is None else self.target_limit_high
        act_limit_low = np.mean(self.min_act_vars) if self.act_limit_low is None else self.act_limit_low
        act_limit_high = np.mean(self.max_act_vars) if self.act_limit_high is None else self.act_limit_high
        
        self.early_trunc=EarlyTrunc(
            patience=patience,
            verbose=verbose,
            delta=delta, 
            target_limit_low=target_limit_low,
            target_limit_high=target_limit_high,
            act_limit_low=act_limit_low,
            act_limit_high=act_limit_high
            )
    
    def _setup_logging(self, log_file='app.log', log_level=logging.INFO) -> None:
        """Sets up the logging configuration with file rotation and archiving."""
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Set up a TimedRotatingFileHandler to rotate logs daily and compress old logs
        handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=7)
        handler.suffix = "%Y-%m-%d"  # Log files will be suffixed with the date
        handler.extMatch = r"^\d{4}-\d{2}-\d{2}$"
    
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[handler, logging.StreamHandler()]
        )
    
        # Function to compress old logs after rotation
        def compress_log_files():
            for filename in os.listdir(log_dir):
                if filename.endswith(".log") and not filename.endswith(".gz"):
                    with open(os.path.join(log_dir, filename), 'rb') as f_in:
                        with gzip.open(os.path.join(log_dir, filename + '.gz'), 'wb') as f_out:
                            f_out.writelines(f_in)
                    os.remove(os.path.join(log_dir, filename))  # Remove the uncompressed log
    
        # Call the compression function after rotating the logs
        handler.doRollover = compress_log_files
        
    def set_agent_args(self, agent_args: AgentArgs, flag: str = 'train', eval_len: int = 720) -> None:
        """Sets the agent arguments and initializes environment parameters accordingly."""
        self.experiment = agent_args.experiment
        self.agent_name = agent_args.agent_name
        self.results_root_path = agent_args.results_root_path
        self.const_el = agent_args.const_el
        self.min_el = agent_args.min_el
        self.max_el = agent_args.max_el
        
        self.flag = flag
        self.eval_len = eval_len
        
        self.figures_folder = os.path.join(self.results_root_path, 
                                           self.agent_name)
        
        self._setup_spaces()
    
    def set_env_idx(self, env_idx:int = 1) -> None:
        """Sets the environment idx for multi-env training."""
        self.env_idx = env_idx
    
    def _validate_vars(self, custom_vars: Any = None) -> Union[str, List[str]]:
        """Validates and processes the custom_vars input."""
        # If custom_vars is a single string, validate it
        if isinstance(custom_vars, str):
            if custom_vars not in self.columns:
                raise ValueError(f"'{custom_vars}' is not in the dataframe columns")
            return custom_vars
    
        # If custom_vars is a list or tuple, validate each entry
        elif isinstance(custom_vars, (list, tuple)):
            if not all(isinstance(var, str) for var in custom_vars):
                raise ValueError("All elements in custom_vars must be strings")
            else:
                # Create an expanded list of observation variables based on custom_var mappings
                expanded_obs_vars = []
                variable_map = {
                    'act_vars': self.act_vars,
                    'exog_vars': self.exog_vars,
                    'target_vars': self.target_vars,
                    'time_vars': self.time_vars
                }
                
                for var in custom_vars:
                    expanded_obs_vars.extend(variable_map.get(var, [var]))
                    
                missing_vars = [var for var in list(set(expanded_obs_vars)) if var not in self.columns]
                if missing_vars:
                    raise ValueError(f"The following variables are not in the dataframe columns: {missing_vars}")
                    
                # Return the unique list of expanded variables
                return list(expanded_obs_vars)
    
        else:
            raise ValueError("custom_vars must be a string or a list/tuple of strings")
    
    def set_obs_vars(self, obs_vars: Union[List[str], str]) -> None:
        """Sets the observation vars, based on custom inputs."""
        self.obs_vars = self._validate_vars(obs_vars)
        self._setup_vars()
        self._setup_spaces()
        self._setup_vars_names()
    
    def set_act_vars(self, act_vars: Union[List[str], str]) -> None:
        """Sets the observation vars, based on custom inputs."""
        self.act_vars = self._validate_vars(act_vars) 
        self._setup_vars()
        self._setup_spaces()
        self._setup_vars_names()

    def set_obs_space(self, low: Union[List[float], Tuple[float], np.ndarray, float]=None, 
                      high: Union[List[float], Tuple[float], np.ndarray, float]=None, dim: int=None) -> None:
        """Sets a custom observation space, based on the inputs."""
        # Use current values if new ones are not provided
        if low is not None:
            self.obs_bounds_low = low
        if high is not None:
            self.obs_bounds_high = high
        if dim is not None:
            self.obs_dim = dim
            
        self._setup_spaces()
    
    def set_act_space(self, low: Union[List[float], Tuple[float], np.ndarray, float]=None, 
                      high: Union[List[float], Tuple[float], np.ndarray, float]=None, dim: int=None) -> None:
        """Sets a custom action space, based on the inputs."""
        # Use current values if new ones are not provided
        if low is not None:
            self.act_bounds_low = low
        if high is not None:
            self.act_bounds_high = high
        if dim is not None:
            self.act_dim = dim
            
        self._setup_spaces()
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None, eval_starts: Union[list, str] = None) -> Tuple[np.ndarray, dict]:
        """Resets the environment for a new episode."""
        early_stopping = False
        if options is not None and 'early_stopping' in options:
            # env_idx starts from 1 when creating multi environment
            early_stopping = options['early_stopping'] if len(options['early_stopping']) == 1 else options['early_stopping'][self.env_idx - 1]
        if not early_stopping and self._is_episode_active():
            return self.obs, self.info
        
        if seed is not None:
            self.seed(seed)
        
        self._reset_environment_state()
        self._initialize_episode(eval_starts)
        
        if self.do_early_trunc:
            self.early_trunc.reset_counters()
        
        info: dict = {'round': self.round}
        return self.obs, info

    def _is_episode_active(self) -> bool:
        """Check if the last episode is still active."""
        return self.episode_num != 0 and self.round != 0 and self.round < self.ep_len and not self.truncated
    
    def _reset_environment_state(self) -> None:
        """Reset the environment state and close figures."""
        plt.close('all')
        # self.observations = np.empty((0, len(self.obs_idxs)))
        self.rewards = np.array([])
        self.targets = np.empty((0, len(self.target_idxs)))
        self.actions = np.empty((0, len(self.act_idxs)))
        self.episode_num += 1
        self.round = 0
        self.done = False
        self.truncated = False

    def _initialize_episode(self, eval_starts: str) -> None:
        """Initialize the episode configuration based on evaluation mode."""
        if self.flag == 'eval':
            # XXX: improve this
            if eval_starts is not None:
                timezone = self.df.index.tz
                eval_datetimes = [pd.to_datetime(d).tz_localize(timezone) for d in eval_starts]
                self.eval_starts = [self.df.index.get_loc(e) for e in eval_datetimes]
                self.n_eval = len(self.eval_starts)
                self.eval_num = 0
            elif self.eval_starts is None:
                self.eval_starts = self.eval_ep_locs
    
        self._set_experiment()
        self._make_sequence()
    
        # self.observations.append(self.obs)
        
        # Extracting the last state
        last_state = self.state
        
        self.targets = np.vstack([self.targets, last_state[self.target_idxs].reshape(1, -1)])
        self.actions = np.vstack([self.actions, last_state[self.act_idxs].reshape(1, -1)])
      
        reward = self.actual_rewards_ep[0] if self.actual_rewards_ep.size > 0 else 0
        self.rewards = np.append(self.rewards, reward)
    
        if self.figure:
            plt.close(self.figure)

    def seed(self, seed=None):
        """Seed the random number generator for reproducibility."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        self.seed_value = seed  # Store the seed for potential re-use in reset
        return [seed]

    def close(self) -> None:
        """Clean up resources used by the environment (if any)."""
        pass
    
    def _reset_array(self, array: np.ndarray) -> np.ndarray:
        """Utility function to reset a NumPy array to an empty state with the same number of columns."""
        num_columns = array.shape[1] if array.size > 0 else 0
        return np.empty((0, num_columns))

    def _make_sequence(self) -> None:
        """Create and configure the sequence for the current episode."""
        self.actual_targets = self._reset_array(self.actual_targets)
        self.actual_actions = self._reset_array(self.actual_actions)

        if self.flag == 'eval':
            self.ep_len = self.eval_len
            self.ep_start = self.eval_starts[self.eval_num]
            self.eval_num += 1
            if self.eval_num == self.n_eval:
                self.eval_num = 0
        else:
            self._configure_training_episode()

        self.start_date = pd.to_datetime(self.df.index[self.ep_start])
        self.sequence = self.df[self.ep_start:self.ep_start + self.seq_len].to_numpy(copy=True)
        self.actual_sequence = self.df.iloc[self.ep_start:self.ep_start + self.seq_len + self.ep_len + 1].to_numpy(copy=True)
        self._setup_initial_state_and_data()

    def _configure_training_episode(self) -> None:
        """Configures episode length and start for training based on episode settings."""
        self.ep_len = np.random.randint(self.min_el, self.max_el) if self.random_episode_length else self.const_el
        if self.random_episode_start:
            self.ep_start = np.random.randint(0, len(self.df) - self.seq_len - self.ep_len)
        else:
            ep_locs = range(0, len(self.df) - self.seq_len - self.ep_len + 1, self.ep_len)
            if self.episode_num > len(ep_locs):
                self.episode_num = 1
            self.ep_start = ep_locs[self.episode_num - 1]

    def _setup_initial_state_and_data(self) -> None:
        """Sets up initial observation state and extracts data for the current episode."""
        # Set up the initial state
        idxs: List[int] = self.act_idxs + self.exog_idxs + self.target_idxs
        self.state: np.ndarray = self.sequence[-1, idxs]
    
        sequence: np.ndarray = self.sequence.astype(np.float32)     
        self.obs: np.ndarray = sequence[-1, self.obs_idxs].reshape(-1)
    
        # Define the start and end of the sequence
        seq_end: int = self.ep_start + self.seq_len
        ep_end: int = seq_end + self.ep_len
    
        # Extract targets, actions, and observations efficiently using .values
        self.actual_targets = self.df.iloc[seq_end - 1:ep_end, self.target_idxs].to_numpy()
        self.actual_actions = self.df.iloc[seq_end - 1:ep_end, self.act_idxs].to_numpy()
        self.actual_observations = self.df.iloc[seq_end - 1:ep_end, self.obs_idxs].to_numpy()
    
        # Directly slice the rewards list to avoid iteration and conversion
        self.actual_rewards_ep = np.array(self.actual_rewards[seq_end - 1:ep_end].values)

    def _make_df(self, arr: np.ndarray, step: int) -> pd.DataFrame:
        """Converts array to DataFrame, applies time features if needed."""
        start = self.start_date
        first_date = start + pd.Timedelta(minutes=step * self.freq_min)
        index = pd.date_range(start=first_date, periods=self.seq_len, freq=self.freq)
        df = pd.DataFrame(arr, index=index, columns=self.columns)
        return df

    def _set_experiment(self) -> None:
        """Configuration mapping for experiments."""
        experiment_config = {
            1: (False, False),
            2: (False, True),
            3: (True, False),
            4: (True, True)
        }
    
        if self.agent_args:
            if self.experiment in experiment_config:
                self.random_episode_start, self.random_episode_length = experiment_config[self.experiment]
            else:
                raise ValueError(f"Unsupported experiment number: {self.experiment}")
        else:
            self.random_episode_start, self.random_episode_length = experiment_config[self.experiment]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Run one timestep of the environment’s dynamics.
    
        Args:
            action (np.ndarray): The action taken by the agent at this timestep.
    
        Returns:
            obs (np.ndarray): The agent’s observation of the current environment.
            reward (float): The reward returned after the previous action.
            done (bool): Whether the episode has ended. This is primarily for 
                         signaling to the agent that the episode is over.
            truncated (bool): Whether the episode was truncated for reasons other
                              than reaching a terminal state. For instance, 
                              an early stopping criterion.
            info (dict): Diagnostic information useful for debugging.
        """
        # Increment the round counter and update info
        self.round += 1
        info = {'round': self.round}
        self.info = info
    
        # Predict the next targets and clip if necessary
        predicted_targets = self._state_predictor()
        predicted_targets = self._clip_predictions(predicted_targets)
    
        # Create the new state based on the action and predicted targets
        new_state = self._create_new_state(action, predicted_targets)
    
        # Update the sequence and observations
        self.sequence = self._update_sequence(new_state)
        self.obs = self._get_observation()
        self._update_histories(new_state)
    
        # Calculate the reward
        reward = self._calculate_reward()
    
        # Check termination conditions
        done = self._check_done()
        self.done = done
    
        # Check for early truncation if applicable
        if self.do_early_trunc:
            self.early_trunc(target=predicted_targets, action=action)
            self.truncated = self.early_trunc.early_trunc
        else:
            self.truncated = done
    
        # Render the environment if needed
        self.render()
        
        # Return the step information
        return self.obs, reward, done, self.truncated, self.info

    def _state_predictor(self) -> np.ndarray:
        """Wrapper for predicting next targets to simplify `step` method."""
        sequence = np.array(self.sequence, dtype=np.float32)
        forecasted = self.predictor.predict(sequence)
        
        if forecasted.ndim == 1:
            forecasted = forecasted.reshape(1, -1)
        elif forecasted.ndim == 3:
            # Ensure batch dimension is 1
            if forecasted.shape[0] != 1:
                raise ValueError("Predictor's forecast should have batch size 1.")
            forecasted = forecasted.reshape(forecasted.shape[1], -1)
        
        return forecasted[0, :].astype(np.float32)
    
    def _create_new_state(self, action: np.ndarray, predicted_state: np.ndarray) -> np.ndarray:
        """Creates the new state array based on the given action and predicted state."""
        new_state = np.zeros((1, len(self.columns)))
        new_state[:, self.act_idxs] = action
        new_state[:, self.target_idxs] = predicted_state
        new_state[:, self.exog_idxs + self.time_idxs] = self._get_exogenous_values()
        
        return new_state
    
    def _update_sequence(self, new_state: np.ndarray) -> np.ndarray:
        """Updates the sequence with the new state, keeping the sequence length constant."""
        updated_sequence = np.roll(self.sequence, shift=-1, axis=0)
        updated_sequence[-1, :] = new_state
        return updated_sequence
    
    def _get_observation(self) -> np.ndarray:
        """Returns the current observation."""
        return np.float32(self.sequence[-1, self.obs_idxs].reshape(-1))
    
    def _update_histories(self, new_state: np.ndarray) -> None:
        """Updates the histories of observations, targets, and actions."""
        # self.observations = np.vstack([self.observations, self.obs])
        self.targets = np.vstack([self.targets, new_state[0, self.target_idxs]])
        self.actions = np.vstack([self.actions, new_state[0, self.act_idxs]])
        self.state = new_state[:, self.act_idxs + self.exog_idxs + self.target_idxs]
        self.actual_state = np.concatenate((
            new_state[0, self.act_idxs + self.exog_idxs].reshape(1, -1),
            self.actual_targets[self.round].reshape(1, -1)), axis=1)
    
    def _calculate_reward(self) -> float:
        """Calculates the reward for the current step."""
        reward = self.reward_function.calculate_reward()
        self.rewards = np.append(self.rewards, reward)
        return reward
    
    def _check_done(self) -> bool:
        """Checks if the episode has ended based on the current round and episode length."""
        return self.round >= self.ep_len
    
    def _get_exogenous_values(self) -> np.ndarray:
        """Fetches the exogenous variables for the current time step from the DataFrame."""
        try:
            # Fetch exogenous variables for the current step
            return self.df.iloc[self.ep_start + self.seq_len + self.round, self.exog_idxs + self.time_idxs]
        except IndexError:
            # Handle potential out-of-bounds access gracefully
            return np.zeros(len(self.exog_idxs) + len(self.time_idxs))

    def _clip_predictions(self, state: np.ndarray, delta_min: float=0.01, delta_max: float=0.001) -> np.ndarray:
        """Normalizes observations based on the observation space.""" 
        delta = random.uniform(delta_min, delta_max)
        
        min_target_with_delta = np.array(self.min_target_vars) + delta
        max_target_with_delta = np.array(self.max_target_vars) - delta
        
        state = np.clip(state, min_target_with_delta, max_target_with_delta)
        
        return state
    
    def render(self) -> None:
        """Handles the rendering of the environment based on the mode specified."""
        # If flag is 'train', exit the method and do nothing
        # XXX: Do this better and add render_mode
        if self.flag == 'train':
            return
        
        # Continue with the regular render logic
        if self.mode == MODE_LIVE:
            self._render_live()
        elif self.mode == MODE_NOT_LIVE:
            if self.done:
                self._render_not_live()
            else:
                return
        elif self.mode == MODE_PRINT_STATUS:
            self._print_environment_status()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def set_mode(self, mode: str) -> None:
        """Sets the mode for the environment, ensuring it's a valid mode."""
        if mode not in {MODE_LIVE, MODE_NOT_LIVE, MODE_PRINT_STATUS}:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {MODE_LIVE}, {MODE_NOT_LIVE}, {MODE_PRINT_STATUS}.")
        self.mode = mode

    def _render_live(self) -> None:
        """Render the environment live with plots updating in real-time."""
        if self.figure is None:
            self.figsize = (6.5, len(self.target_vars) + len(self.act_vars) + 2)
            self.figure, self.target_axes, self.action_axes, self.reward_ax = initialize_plots(self.n_act, self.n_targets, self.figsize)
            
        plt.ion()
        self.figure = plot_live(self, self.figure, self.target_axes, self.action_axes, self.reward_ax, self.colors)
        self.figure.canvas.draw()
        plt.show()
        plt.pause(0.01)  # Small pause to ensure the plot updates

        if self.round == self.ep_len:
            finalize_and_save_plot(self.figure, self.results_root_path, self.agent_name, 'Live_plot')
            plt.ioff()

    def _render_not_live(self) -> None:
        """Render the final state of the environment for the episode."""
        if self.figure is None:
            self.figsize = (6.5, len(self.target_vars) + len(self.act_vars) + 2)
            self.figure, self.target_axes, self.action_axes, self.reward_ax = initialize_plots(self.n_act, self.n_targets, self.figsize)
        
        # eval_num is already been added 1 in the sequence making
        eval_date = self.df.index[self.eval_starts[self.eval_num-1]]
        eval_date = eval_date.strftime('%Y_%m_%d_%H_%M_%S')
        
        fig_name = f'Eval_Live_plot_{eval_date}' if self.flag == 'eval' else 'Live_plot'
        self.figure = plot_not_live(self, self.figure, self.target_axes, self.action_axes, self.reward_ax, self.colors)
        finalize_and_save_plot(self.figure, self.results_root_path, self.agent_name, fig_name)
        fig_name = f'Eval_Live_plot_with_actual_values_{eval_date}' if self.flag == 'eval' else 'Live_plot_with_actual_values'
        self.figure = plot_not_live(self, self.figure, self.target_axes, self.action_axes, self.reward_ax, self.colors)
        finalize_and_save_plot(self.figure, self.results_root_path, self.agent_name, fig_name)
        plt.close('all')

    def _print_environment_status(self) -> None:
        """Prints current environment status for non-visual modes."""
        print(f'Round: {self.round}, Actions: {self.actions[-1]}, Targets: {self.targets[-1]}')
        print(f'Reward Received: {self.rewards[-1]}, Total Reward: {sum(self.rewards)}')
        print("==============================================================")
    
    def _log_environment_status(self) -> None:
        logging.info(f'Round: {self.round}, Actions: {self.actions[-1]}, Targets: {self.targets[-1]}')
        logging.info(f'Reward Received: {self.rewards[-1]}, Total Reward: {sum(self.rewards)}')
        
    def choose_actual_action(self) -> np.ndarray:
        """Selects the actual action from the sequence based on the current round as a 1D NumPy array."""
        action = self.actual_sequence[self.seq_len + self.round, self.act_idxs]
        return action.ravel()
    
    def get_data(self) -> Any:
        """Returns the complete dataset object."""
        return self.data
    
    def get_targets(self) -> np.ndarray:
        """Returns the most recent targets as a 1D NumPy array."""
        return self.targets[-1].ravel()
    
    def get_actual_targets(self) -> np.ndarray:
        """Returns the most recent actual targets as a 1D NumPy array."""
        return self.actual_targets[-1].ravel()
    
    def get_state(self) -> np.ndarray:
        """Returns the current state as a 1D NumPy array."""
        return self.state.ravel()
    
    def get_actual_state(self) -> np.ndarray:
        """Returns the actual state as a 1D NumPy array."""
        return self.actual_state.ravel()
    
    def get_observations(self) -> np.ndarray:
        """Returns the most recent observation as a 1D NumPy array."""
        return self.obs.ravel()
    
    def get_actual_observations(self) -> np.ndarray:
        """Returns the most recent actual observations as a 1D NumPy array."""
        return self.actual_observations[-1].ravel()
    
    def get_actions(self) -> np.ndarray:
        """Returns the most recent actions as a 1D NumPy array."""
        return self.actions[-1].ravel()
    
    def get_actual_actions(self) -> np.ndarray:
        """Returns the most recent actual actions as a 1D NumPy array."""
        return self.actual_actions[-1].ravel()
    
    def get_round(self) -> int:
        """Returns the current round number as an integer."""
        return self.round
    
    def get_reward(self) -> np.ndarray:
        """Returns the most recent actions as a 1D NumPy array."""
        return self.rewards[-1].ravel()

