import numpy as np
import pandas as pd

import ml_forecasts.preprocessing as preprocessing
import misc.plotting as plotting

from math import sqrt
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

def run_model(independent_variables_df, dependent_variable_series, window_size):
    scaled_independent_variables_df = preprocessing.scale_independent_variables(independent_variables_df)
    scaled_dependent_variable, dependent_variable_scaler = preprocessing.scale_dependent_variable(dependent_variable_series)
    x_seq, y_seq = preprocessing.create_sequences(scaled_independent_variables_df, scaled_dependent_variable, 3)
    model = build_lstm_model((x_seq.shape[1], x_seq.shape[2]), 50, 0.2, 20)
    model, history = train_lstm_model(scaled_independent_variables_df, scaled_dependent_variable, 3)
    
    x_all, y_all = preprocessing.create_sequences(scaled_independent_variables_df, scaled_dependent_variable, window_size)
    y_pred_all = model.predict(x_all)
    y_pred_all_unscaled = dependent_variable_scaler.inverse_transform(y_pred_all.reshape(-1, 1)).flatten()
    y_all_unscaled = dependent_variable_scaler.inverse_transform(y_all.reshape(-1, 1)).flatten()
    
    mse_all = np.mean((y_all_unscaled - y_pred_all_unscaled) ** 2)
    rmse_all = np.sqrt(mse_all)
    r2 = r2_score(y_all_unscaled, y_pred_all_unscaled)
    print("Performance on entire dataset:")
    print("MSE:", mse_all)
    print("RMSE:", rmse_all)
    print("R^2:", r2)
    
    # Plot predictions vs. actual values using your plotting utility
    plotting.plot_predictions_vs_actuals(y_pred_all_unscaled, y_all_unscaled)

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2, dense_units=20):
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
    model.add(LSTM(lstm_units, input_shape=input_shape))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Single output for regression
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_lstm_model(X_train: pd.DataFrame,
                     y_train: pd.Series,
                     window_size: int,
                     lstm_units=50,
                     dropout_rate=0.2,
                     dense_units=20,
                     batch_size=32,
                     epochs=100,
                     validation_split=0.2):
    """
    Prepares sequences from the training data, builds the LSTM model, and trains it.
    
    Parameters:
        X_train (pd.DataFrame): Training data for independent variables.
        y_train (pd.Series): Training data for the dependent variable.
        window_size (int): Number of time steps per sequence.
        lstm_units (int): Number of units in the LSTM layer.
        dropout_rate (float): Dropout rate after the LSTM layer.
        dense_units (int): Number of units in the Dense layer following LSTM.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of epochs to train.
        validation_split (float): Proportion of training data for validation.
    
    Returns:
        model (tf.keras.Model): The trained LSTM model.
        history (History): Training history returned by model.fit().
    """
    X_seq, y_seq = preprocessing.create_sequencescreate_sequences(X_train, y_train, window_size)
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    
    model = build_lstm_model(input_shape, lstm_units, dropout_rate, dense_units)
    
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
