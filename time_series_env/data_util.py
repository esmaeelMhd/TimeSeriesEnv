import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import SplineTransformer

def add_simple_time_features(df: pd.DataFrame, time_features: list[str], date_col: str='date') -> pd.DataFrame:
    """Adds simple time features (e.g., month, weekday) to the DataFrame based on its date column or index."""
    date_series = pd.to_datetime(df[date_col]) if date_col in df.columns else pd.to_datetime(df.index)
    time_features = [time_features] if isinstance(time_features, str) else time_features
    
    for time_feature in time_features:
        if time_feature == 'month':
            df['month'] = date_series.month
        elif time_feature == 'day_of_week':
            df['day_of_week'] = date_series.weekday
        elif time_feature == 'hour':
            df['hour'] = date_series.hour
        elif time_feature == 'day_of_month':
            df['day_of_month'] = date_series.day
        elif time_feature == 'week_of_year':
            df['week_of_year'] = date_series.isocalendar().week
        else:
            raise ValueError(f"Unsupported time feature type: {time_feature}")

    return df

def add_cyclic_time_features(df: pd.DataFrame, time_features: list[str], date_col: str='date') -> pd.DataFrame:
    """Embeds cyclical time features into the DataFrame based on its date column or index."""
    date_series = pd.to_datetime(df[date_col]) if date_col in df.columns else pd.to_datetime(df.index)
    time_features = [time_features] if isinstance(time_features, str) else time_features

    for time_feature in time_features:
        if time_feature == 'month':
            # As the months start from 1
            feature = date_series.month - 1
            max_val = 12
        elif time_feature == 'day_of_week':
            feature = date_series.weekday
            max_val = 7
        elif time_feature == 'hour':
            feature = date_series.hour
            max_val = 24
        elif time_feature == 'day_of_month':
            feature = date_series.day
            max_val = 31
        else:
            raise ValueError(f"Unsupported time feature type: {time_feature}")

        df[f'sin_{time_feature}'] = np.sin(2 * np.pi * feature / max_val)
        df[f'cos_{time_feature}'] = np.cos(2 * np.pi * feature / max_val)

    return df

def add_spline_time_features(df: pd.DataFrame, time_features: list[str], date_col: str='date', degree: int=3, n_knots: int=4) -> tuple:
    """Embeds spline-based time features into the DataFrame based on its date column or index."""
    date_series = pd.to_datetime(df[date_col]) if date_col in df.columns else pd.to_datetime(df.index)
    time_features = [time_features] if isinstance(time_features, str) else time_features
    
    spline_col_count = 0
    new_columns = {}
    for time_feature in time_features:
        if time_feature == 'month':
            feature = date_series.month
            period, n_splines = 12, 6
        elif time_feature == 'day_of_week':
            feature = date_series.weekday
            period, n_splines = 7, 3
        elif time_feature == 'hour':
            feature = date_series.hour
            period, n_splines = 24, 12
        else:
            raise ValueError(f"Unsupported time feature type: {time_feature}")

        feature = feature.values.reshape(-1, 1)
        
        def periodic_spline_transformer(period, n_splines=None, degree=3):
            if n_splines is None:
                n_splines = period
            n_knots = n_splines + 1  # periodic and include_bias is True
            return SplineTransformer(degree=degree, n_knots=n_knots, knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
                extrapolation="periodic", include_bias=True)

        # Apply SplineTransformer
        spline_features = periodic_spline_transformer(period, n_splines, degree).fit_transform(feature)
        spline_col_count += spline_features.shape[1]

        # Add each spline feature as a new column
        for i in range(spline_features.shape[1]):
            new_columns[f'{time_feature}_spline_{i}'] = spline_features[:, i]

    # Once all new features are calculated, concatenate them with the original DataFrame
    df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    return df, spline_col_count

def add_onehot_time_features(df: pd.DataFrame, time_features: list[str], date_col: str='date') -> tuple:
    """Embeds one-hot encoded time features into the DataFrame based on its date column or index."""
    date_series = pd.to_datetime(df[date_col]) if date_col in df.columns else pd.to_datetime(df.index)
    time_features = [time_features] if isinstance(time_features, str) else time_features
    
    onehot_col_count = 0
    for time_feature in time_features:
        if time_feature == 'month':
            feature = date_series.month
        elif time_feature == 'day_of_week':
            feature = date_series.weekday
        elif time_feature == 'hour':
            feature = date_series.hour
        elif time_feature == 'day_of_month':
            feature = date_series.day
        else:
            raise ValueError(f"Unsupported time feature type: {time_feature}")

        onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        onehot_encoded = onehot_encoder.fit_transform(feature.values.reshape(-1, 1))

        onehot_df = pd.DataFrame(onehot_encoded,
                                 columns=[f'{time_feature}_{i+1}' for i in range(onehot_encoded.shape[1])],
                                 index=df.index)
        
        onehot_col_count += onehot_encoded.shape[1]
        df = pd.concat([df, onehot_df], axis=1)

    return df, onehot_col_count