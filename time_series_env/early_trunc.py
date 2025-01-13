import numpy as np
from typing import Callable, Optional

class EarlyTrunc:
    """
    Improved EarlyTrunc class for monitoring training performance and stopping early
    when target or action values exceed specified thresholds consistently.

    Attributes:
    -----------
    patience : int
        Number of consecutive steps before early stopping is triggered.
    verbose : bool
        If True, print messages when early stopping is triggered.
    delta : float
        Margin to adjust thresholds dynamically.
    early_trunc : bool
        Indicates whether early stopping has been triggered.
    target_limit_low : float
        Lower threshold for the target value.
    target_limit_high : float
        Upper threshold for the target value.
    act_limit_low : float
        Lower threshold for the action value.
    act_limit_high : float
        Upper threshold for the action value.
    aggregate_fn : Callable
        Function to aggregate target and action values (default: np.mean).
    trigger_condition : Callable
        Custom function to decide if stopping should occur (default: patience check).
    threshold_adjust_fn : Optional[Callable]
        Function to dynamically adjust thresholds during training (default: None).
    """

    def __init__(
        self,
        patience: int = 50,
        verbose: bool = False,
        delta: float = 0,
        target_limit_low: Optional[float] = None,
        target_limit_high: Optional[float] = None,
        act_limit_low: Optional[float] = None,
        act_limit_high: Optional[float] = None,
        aggregate_fn: Callable = np.mean,
        trigger_condition: Optional[Callable] = None,
        threshold_adjust_fn: Optional[Callable] = None,
    ):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.early_trunc = False

        # Counters for breaches
        self.target_counter_low = 0
        self.target_counter_high = 0
        self.action_counter_low = 0
        self.action_counter_high = 0

        # Limits for target and action
        self.target_limit_low = target_limit_low + delta if target_limit_low else None
        self.target_limit_high = target_limit_high - delta if target_limit_high else None
        self.act_limit_low = act_limit_low + delta if act_limit_low else None
        self.act_limit_high = act_limit_high - delta if act_limit_high else None

        # Customizable aggregate function and trigger condition
        self.aggregate_fn = aggregate_fn
        self.trigger_condition = trigger_condition or self.default_trigger_condition
        self.threshold_adjust_fn = threshold_adjust_fn

    def __call__(self, target=None, action=None):
        """
        Evaluate target and action values at each step and update counters.

        Parameters:
        -----------
        target : array-like, optional
            The current target value(s) from the environment observation.
        action : array-like, optional
            The current action value(s) taken by the agent.
        """
        if target is not None:
            target_stat = self.aggregate_fn(np.array(target))
            self.update_counters(target_stat, is_action=False)

        if action is not None:
            action_stat = self.aggregate_fn(np.array(action))
            self.update_counters(action_stat, is_action=True)

        # Adjust thresholds dynamically if a function is provided
        if self.threshold_adjust_fn:
            self.threshold_adjust_fn(self)

        # Check trigger condition
        self.early_trunc = self.trigger_condition()
        if self.early_trunc and self.verbose:
            print("Early truncation has been triggered.")

    def update_counters(self, value, is_action: bool):
        """Update counters based on whether the value breaches thresholds."""
        if is_action:
            low_limit, high_limit = self.act_limit_low, self.act_limit_high
            low_counter, high_counter = "action_counter_low", "action_counter_high"
        else:
            low_limit, high_limit = self.target_limit_low, self.target_limit_high
            low_counter, high_counter = "target_counter_low", "target_counter_high"

        if low_limit is not None and value <= low_limit:
            setattr(self, low_counter, getattr(self, low_counter) + 1)
            setattr(self, high_counter, 0)
        elif high_limit is not None and value >= high_limit:
            setattr(self, high_counter, getattr(self, high_counter) + 1)
            setattr(self, low_counter, 0)
        else:
            setattr(self, low_counter, 0)
            setattr(self, high_counter, 0)

    def default_trigger_condition(self) -> bool:
        """Default condition: trigger early stopping if any counter reaches patience."""
        max_target_counter = max(self.target_counter_low, self.target_counter_high)
        max_action_counter = max(self.action_counter_low, self.action_counter_high)
        return max(max_target_counter, max_action_counter) >= self.patience

    def reset_counters(self):
        """Reset all counters to zero."""
        self.target_counter_low = 0
        self.target_counter_high = 0
        self.action_counter_low = 0
        self.action_counter_high = 0
