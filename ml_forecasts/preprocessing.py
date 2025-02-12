import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def scale_independent_variables(independent_variables_df):
    scaler = StandardScaler()
    scaled_independent_variables = scaler.fit_transform(independent_variables_df)
    scaled_independent_variables_df = pd.DataFrame(scaled_independent_variables, columns=independent_variables_df.columns)
    
    return scaled_independent_variables_df

def scale_dependent_variable(dependent_variable_series):
    """
    Scales a pandas Series (dependent variable) using StandardScaler,
    so that its values are on the same scale as the independent variables.
    
    Parameters:
        dependent_variable_series (pd.Series): The dependent variable series to be scaled.
    
    Returns:
        scaled_series (pd.Series): The scaled dependent variable.
        scaler (StandardScaler): The fitted scaler for future inverse transformations.
    """
    from sklearn.preprocessing import StandardScaler
    # Reshape to 2D array
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dependent_variable_series.values.reshape(-1, 1))
    scaled_series = pd.Series(scaled.flatten(), index=dependent_variable_series.index)
    return scaled_series, scaler

def create_sequences(X: pd.DataFrame, y: pd.Series, window_size: int):
    """
    Transforms the data into sequences for an LSTM model.
    
    For each index i, a sequence of features from i to i+window_size-1 is used 
    to predict the target at i+window_size.
    
    Parameters:
        X (pd.DataFrame): DataFrame of independent variables.
        y (pd.Series): Series of the dependent variable.
        window_size (int): Number of time steps per sequence.
    
    Returns:
        X_seq (np.ndarray): 3D array of shape (num_sequences, window_size, num_features).
        y_seq (np.ndarray): 1D array of targets corresponding to each sequence.
    """
    X_values = X.values
    y_values = y.values
    X_seq, y_seq = [], []
    
    for i in range(len(X_values) - window_size):
        X_seq.append(X_values[i:i+window_size])
        y_seq.append(y_values[i+window_size])
    
    return np.array(X_seq), np.array(y_seq)