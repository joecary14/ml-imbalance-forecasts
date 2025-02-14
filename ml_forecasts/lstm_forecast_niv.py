import numpy as np
import pandas as pd

import ml_forecasts.preprocessing as preprocessing
import misc.plotting as plotting

from math import sqrt
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def run_model(independent_variables_df, dependent_variable_series, window_size, 
              test_split, number_of_lstm_layers, lstm_units_per_layer, dropout_rate, dense_units, batch_size, epochs, validation_split):
    scaled_independent_variables_df = preprocessing.scale_independent_variables(independent_variables_df)
    scaled_dependent_variable, dependent_variable_scaler = preprocessing.scale_dependent_variable(dependent_variable_series)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_independent_variables_df, scaled_dependent_variable, test_size=test_split, shuffle=False)
    model, history = train_lstm_model(
        X_train, y_train, window_size, number_of_lstm_layers, lstm_units_per_layer, dropout_rate, dense_units, batch_size, epochs, validation_split)
    
    test_metrics = evaluate_lstm_model(model, X_test, y_test, window_size)
    print("Test set performance:")
    print("MSE:", test_metrics['mse'])
    print("RMSE:", test_metrics['rmse'])
    
    x_test_seq, y_test_seq = preprocessing.create_sequences(X_test, y_test, window_size)
    y_pred_test = model.predict(x_test_seq)
    
    y_pred_test_unscaled = dependent_variable_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
    y_test_unscaled = dependent_variable_scaler.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()
    
    r2 = r2_score(y_test_unscaled, y_pred_test_unscaled)
    print("Test R^2:", r2)
    
    plotting.plot_predictions_vs_actuals(y_pred_test_unscaled, y_test_unscaled)

def build_lstm_model(input_shape, number_of_lstm_layers, lstm_units_per_layer, dropout_rate, dense_units):
    """
    Constructs and compiles an LSTM model.
    
    Parameters:
        input_shape (tuple): Shape of the input data (window_size, num_features).
        lstm_units (int): Number of units in the LSTM layer.
        dropout_rate (float): Dropout rate after the LSTM layer.
        dense_units (int): Number of units in the Dense layer following LSTM.
    
    Returns:
        model (tf.keras.Model): Compiled LSTM model.
    """
    model = Sequential()
    for layer in range(number_of_lstm_layers-1):
        model.add(LSTM(lstm_units_per_layer, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    model.add(LSTM(lstm_units_per_layer, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Single output for regression
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     window_size: int,
                     number_of_lstm_layers,
                     lstm_units_per_layer,
                     dropout_rate,
                     dense_units,
                     batch_size,
                     epochs,
                     validation_split):
    """
    Prepares sequences from the training data, builds the LSTM model, and trains it.
    
    Parameters:
        X_train (pd.DataFrame): Training data for independent variables.
        y_train (pd.Series): Training data for the dependent variable.
        window_size (int): Number of time steps per sequence.
        lstm_units_per_layer (int): Number of units in the LSTM layer.
        dropout_rate (float): Dropout rate after the LSTM layer.
        dense_units (int): Number of units in the Dense layer following LSTM.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of epochs to train.
        validation_split (float): Proportion of training data for validation.
    
    Returns:
        model (tf.keras.Model): The trained LSTM model.
        history (History): Training history returned by model.fit().
    """
    X_seq, y_seq = preprocessing.create_sequences(X_train, y_train, window_size)
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    
    model = build_lstm_model(
        input_shape, number_of_lstm_layers = number_of_lstm_layers, lstm_units_per_layer = lstm_units_per_layer, 
        dropout_rate = dropout_rate, dense_units = dense_units)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_seq, y_seq,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[early_stopping],
                        verbose=1)
    return model, history

def evaluate_lstm_model(model, X_test: pd.DataFrame, y_test: pd.Series, window_size: int):   
    """
    Evaluates the trained LSTM model using Mean Squared Error and RMSE.
    
    Parameters:
        model (tf.keras.Model): Trained LSTM model.
        X_test (pd.DataFrame): Test data for independent variables.
        y_test (pd.Series): True values for the dependent variable.
        window_size (int): Number of time steps per sequence.
    
    Returns:
        metrics (dict): Dictionary containing 'mse' and 'rmse'.
    """
    X_seq, y_seq = preprocessing.create_sequences(X_test, y_test, window_size)
    y_pred = model.predict(X_seq)
    
    mse = mean_squared_error(y_seq, y_pred)
    rmse = sqrt(mse)
    return {'mse': mse, 'rmse': rmse}

def save_lstm_model(model, file_path: str):
    """
    Saves the trained LSTM model to disk.
    
    Parameters:
        model (tf.keras.Model): Trained model.
        file_path (str): Path to save the model.
    """
    model.save(file_path)
    print(f"Model saved to {file_path}")

def load_lstm_model(file_path: str):
    """
    Loads a trained LSTM model from disk.
    
    Parameters:
        file_path (str): Path from which to load the model.
    
    Returns:
        model (tf.keras.Model): Loaded model.
    """
    model = load_model(file_path)
    print(f"Model loaded from {file_path}")
    return model
