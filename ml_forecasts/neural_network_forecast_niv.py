import numpy as np
import pandas as pd
from math import sqrt

from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

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
    X_seq, y_seq = create_sequences(X_train, y_train, window_size)
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
    X_seq, y_seq = create_sequences(X_test, y_test, window_size)
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
