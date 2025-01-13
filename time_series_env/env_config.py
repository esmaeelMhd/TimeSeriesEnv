from dataclasses import dataclass, field, fields
from typing import List, Optional
import numpy as np

@dataclass
class AgentArgs:
    """A class to hold configuration parameters for an agent in various experiments."""
    agent_name: str = None
    experiment: int = None
    const_el: Optional[int] = None
    min_el: Optional[int] = None
    max_el: Optional[int] = None
    title: str = 'Agent'
    norm_values: np.bool_ = True
    obs_history: int = 1
    results_root_path: str = 'results'
        
    def __post_init__(self) -> None:
        """Perform validation checks based on experiment and delay_type."""                
        if (self.experiment == 1 or self.experiment == 3) and self.const_el is None:
            raise ValueError("const_el must be provided if the experiment is 1 or 3")
        if (self.experiment == 2 or self.experiment == 4) and (self.min_el is None or self.max_el is None):
            raise ValueError("min_el and max_el must be provided if the experiment is 2 or 4")
    
    @staticmethod
    def from_namespace(args, **kwargs) -> 'AgentArgs':
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(AgentArgs))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        filtered_args.update(kwargs)
        return AgentArgs(**filtered_args)
