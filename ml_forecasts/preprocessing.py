import math
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def check_sufficient_data(X, batch_size, epochs):
    steps_per_epoch = math.ceil(len(X) / batch_size)
    if len(X) < steps_per_epoch * epochs:
        raise ValueError("Not enough data for the given batch size and number of epochs.")

def scale_independent_variables(independent_variables_df):
    scaler = StandardScaler()
    scaled_independent_variables = scaler.fit_transform(independent_variables_df)
    scaled_independent_variables_df = pd.DataFrame(scaled_independent_variables, columns=independent_variables_df.columns)
    
    return scaled_independent_variables_df

def scale_dependent_variable(dependent_variable_series):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dependent_variable_series.values.reshape(-1, 1))
    scaled_series = pd.Series(scaled.flatten(), index=dependent_variable_series.index)
    return scaled_series, scaler

def create_sequences(X: pd.DataFrame, y: pd.Series, window_size: int):
    X_values = X.values
    y_values = y.values
    X_seq, y_seq = [], []
    
    for i in range(len(X_values) - window_size):
        X_seq.append(X_values[i:i+window_size])
        y_seq.append(y_values[i+window_size])
    
    return np.array(X_seq), np.array(y_seq)