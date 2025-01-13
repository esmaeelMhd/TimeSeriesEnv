from dataclasses import dataclass, field, fields
from typing import Any, List, Union

from time_series_env.env_util import load_dataset
from time_series_env.data_util import add_simple_time_features, add_cyclic_time_features, add_spline_time_features, add_onehot_time_features

@dataclass(eq=False)
class TimeSeriesData:
    """
    A data structure for managing time series data along with its associated variables.
    
    This class assumes the DataFrame structure as follows:
    - Columns for action variables (`act_vars`)
    - Columns for exogenous variables (`exog_vars`)
    - Columns for target variables (`target_vars`)
    - (Optional) Columns for time features if `time_f` is enabled.
    
    Attributes:
        data_root_path (str): The root directory path where the dataset is stored.
        data_name (str): The name of the dataset to be loaded.
        index_col (str): The column to be used as the index in the dataset.
        time_f (bool): A flag indicating if time features are enabled.
        has_time_f (bool): A flag indicating if the dataset contains time features.
        num_time_f (int): The number of time features in the dataset.
        time_f_list: List[str] = A list indicationg which time features should be added to the data.
            Options: ['hour', 'month', 'day_of_week', 'day_of_month', 'week_of_year']
        time_f_type: str = 'cyclic'
            Options: 'simple', 'cyclic', 'spline', 'onehot'
    
        act_vars (Union[List[str], str, int]): The action variables in the dataset. 
            These variables represent the controls or decision inputs in the model.
        exog_vars (Union[List[str], str, int]): The exogenous variables in the dataset.
            These variables represent external inputs or predictors that influence the model but are not controlled.
        target_vars (Union[List[str], str, int]): The target variables in the dataset.
            These variables represent the outcomes or responses that the model is trying to predict.
        obs_vars (Union[List[str], str]): The observation variables in the dataset.
            These can either be the same as the target variables or additional observational data.
    
        act_names (Union[List[str], str], optional): Names corresponding to action variables. Defaults to None.
        exog_names (Union[List[str], str], optional): Names corresponding to exogenous variables. Defaults to None.
        target_names (Union[List[str], str], optional): Names corresponding to target variables. Defaults to None.
        obs_names (Union[List[str], str], optional): Names corresponding to observation variables. Defaults to None.
        args (Any, optional): Additional arguments for when the model is loaded with namespace args. Defaults to None.
    """
    data_root_path: str
    data_name: str
    index_col: str
    time_f: bool
    has_time_f: bool
    num_time_f: int
    
    # data variables information
    act_vars: Union[List[str], str]
    exog_vars: Union[List[str], str]
    target_vars: Union[List[str], str]
    obs_vars: Union[List[str], str]
    time_vars: Union[List[str], str] = None
    
    act_names: Union[List[str], str] = None
    exog_names: Union[List[str], str] = None
    target_names: Union[List[str], str] = None
    obs_names: Union[List[str], str] = None
    time_names: Union[List[str], str] = None
    
    time_f_list: List[str] = field(default_factory=lambda: ['hour', 'month', 'day_of_week'])
    time_f_type: str = 'cyclic'
    
    args: Any = None   # For when the model is loaded with namespace args
        
    def __post_init__(self):    
        """Post-initialization to validate and process the dataclass fields."""   
        self.df = load_dataset(self.data_root_path, self.data_name)
        
        self._validate_time_features()
        if self.time_f:
            if not self.has_time_f:
                self._add_time_features()
            self.time_vars = self.df.columns[-self.num_time_f:]
        else:
            self.num_time_f = 0
        
        # Setup action and exogenous variables
        self.act_vars, self.num_act = self._parse_variables(['act_vars', 'act_variable'])
        self.exog_vars, self.num_exog = self._parse_variables(['exog_vars', 'exog_variable'])
        self.target_vars, self.num_target = self._parse_variables(['target_vars', 'target_variable'])  
        self.obs_vars = self.target_vars if self.obs_vars is None else self._process_observation_variables()
        
        if len(self.exog_vars) != len(self.df.columns) - (len(self.act_vars) + len(self.target_vars) + self.num_time_f):
            self.exog_vars = [col for col in self.df.columns[:len(self.columns)-self.num_time_f] if col not in self.act_vars and col not in self.target_vars]
        
        self._validate_and_convert_vars()
    
    @staticmethod
    def from_namespace(args, **kwargs) -> 'TimeSeriesData':
        """Create an instance from a namespace, allowing for manual attribute overrides or additions."""
        valid_keys = set(f.name for f in fields(TimeSeriesData))
        filtered_args = {key: getattr(args, key) for key in valid_keys if hasattr(args, key)}
        
        # Override filtered_args with manually set attributes from kwargs
        filtered_args.update(kwargs)
        
        return TimeSeriesData(**filtered_args)

    def _parse_variables(self, attribute_names: List[str]) -> tuple:
        """Parses control or independent variables from model_args using possible attribute names."""
        for attr in attribute_names:
            value = getattr(self, attr, None)
            if value is not None:
                if isinstance(value, str):
                    return [value], len([value])
                elif isinstance(value, list):
                    return value, len(value)
                else:
                    raise ValueError('The variables should be either Strings or Lists.')

        return [], 0  # Return an empty list if no attributes match   
    
    def _process_observation_variables(self) -> List[str]:
        """Processes observation variables by expanding placeholders."""
        expanded_obs_vars = []
        for var in self.obs_vars:
            if var == 'act_vars':
                expanded_obs_vars.extend(self.act_vars)
            elif var == 'exog_vars':
                expanded_obs_vars.extend(self.exog_vars)
            elif var == 'target_vars':
                expanded_obs_vars.extend(self.target_vars)
            elif var == 'time_vars':
                expanded_obs_vars.extend(self.time_vars)
            else:
                expanded_obs_vars.append(var)
                
        return list(set(expanded_obs_vars))
        
    def _validate_and_convert_vars(self) -> None:
        """Validates and converts control, independent, and target variables into lists if they are not already."""
        vars_list = ['act_vars', 'exog_vars', 'target_vars', 'obs_vars']  
        names_list = ['act_names', 'exog_names', 'target_names', 'obs_names'] 
        
        if self.time_f:
            vars_list.append('time_vars')
            names_list.append('time_names')

        for attr_name in vars_list:
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, str):
                setattr(self, attr_name, [attr_value])
            elif not all(isinstance(x, str) for x in attr_value):
                raise ValueError(f"{attr_name} must be a list of strings or a single string")
        
        for i, attr_name in enumerate(names_list):
            attr_value = getattr(self, attr_name)
            # Assign the vars if names are not provided
            if attr_value in [None, '', []]:
                setattr(self, attr_name, getattr(self, vars_list[i]))
            else:   
                if isinstance(attr_value, str):
                    setattr(self, attr_name, [attr_value])
                elif not all(isinstance(x, str) for x in attr_value):
                    raise ValueError(f"{attr_name} must be a list of strings or a single string")

    def _validate_time_features(self) -> None:
        """Validates the provision of required fields."""
        if self.time_f and self.has_time_f is None:
            raise ValueError("has_time_f must be provided if time_f is True")
            
        elif self.time_f and self.has_time_f and self.num_time_f is None:
            raise ValueError("num_time_f must be provided if time_f is True")
    
    def _add_time_features(self) -> None:
        """Adds the time features to the dataframe according to the type."""
        if self.time_f_type == 'simple':
            self.df = add_simple_time_features(self.df, self.time_f_list, date_col=self.index_col)
            self.num_time_f = len(self.time_f_list)
        elif self.time_f_type == 'cyclic':
            self.df = add_cyclic_time_features(self.df, self.time_f_list, date_col=self.index_col)
            self.num_time_f = 2 * len(self.time_f_list)
        elif self.time_f_type == 'spline':
            degree = 3
            n_knots = 4
            self.df, self.num_time_f = add_spline_time_features(self.df, self.time_f_list, date_col=self.index_col, degree=degree, n_knots=n_knots)
        elif self.time_f_type == 'onehot':
            self.df, self.num_time_f = add_onehot_time_features(self.df, self.time_f_list, date_col=self.index_col)
        else:
            raise ValueError(f"the type of time feature as {self.time_f_type} is not defined.")
            
