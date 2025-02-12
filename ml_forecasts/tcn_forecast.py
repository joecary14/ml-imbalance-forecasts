import numpy as np
import pandas as pd
import ml_forecasts.preprocessing as preprocessing
import misc.plotting as plotting

from math import sqrt
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv1D, Dropout, Flatten, Dense
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score

def run_model(independent_variables_df, dependent_variable_series, window_size):
    scaled_independent_variables_df = preprocessing.scale_independent_variables(independent_variables_df)
    scaled_dependent_variable, dependent_variable_scaler = preprocessing.scale_dependent_variable(dependent_variable_series)
    model, history = train_tcn_model(scaled_independent_variables_df, scaled_dependent_variable, window_size)
    metrics = evaluate_tcn_model(model, scaled_independent_variables_df, scaled_dependent_variable, window_size)
    print(f'{key} : {value}' for key, value in metrics.items())
    y_pred_scaled, y_true_scaled = predict_tcn_model(model, scaled_independent_variables_df, scaled_dependent_variable, window_size)
    y_pred = dependent_variable_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = dependent_variable_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    plotting.plot_predictions_vs_actuals(y_pred, y_true)
    

def build_tcn_model(input_shape, filters=64, kernel_size=2, dilations=[1,2,4,8], dropout_rate=0.2, dense_units=20):
    """
    Constructs and compiles a TCN model.
    
    Parameters:
        input_shape (tuple): Shape of the input data (window_size, num_features).
        filters (int): Number of filters for the Conv1D layers.
        kernel_size (int): Size of the convolutional kernel.
        dilations (list): Dilation rates for successive Conv1D layers.
        dropout_rate (float): Dropout rate applied after each Conv1D layer.
        dense_units (int): Number of units in the Dense layer.
    
    Returns:
        model (tf.keras.Model): Compiled TCN model.
    """
    model = Sequential()
    # Add several Conv1D layers with causal padding and increasing dilation rates.
    for idx, dilation_rate in enumerate(dilations):
        if idx == 0:
            model.add(Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             dilation_rate=dilation_rate,
                             activation='relu',
                             padding='causal',
                             input_shape=input_shape))
        else:
            model.add(Conv1D(filters=filters,
                             kernel_size=kernel_size,
                             dilation_rate=dilation_rate,
                             activation='relu',
                             padding='causal'))
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_tcn_model(X_train: pd.DataFrame, y_train: pd.Series, window_size: int,
                    filters=64, kernel_size=2, dilations=[1,2,4,8], dropout_rate=0.2, dense_units=20,
                    batch_size=32, epochs=100, validation_split=0.2):
    """
    Prepares sequences from training data, builds the TCN model, and trains it.
    
    Parameters:
        X_train (pd.DataFrame): Training data for independent variables.
        y_train (pd.Series): Training data for the dependent variable.
        window_size (int): Number of time steps per sequence.
        filters, kernel_size, dilations, dropout_rate, dense_units: Model configuration parameters.
        batch_size (int): Batch size for training.
        epochs (int): Maximum number of epochs to train.
        validation_split (float): Proportion of training data for validation.
    
    Returns:
        model (tf.keras.Model): Trained TCN model.
        history (History): Training history returned by model.fit().
    """
    X_seq, y_seq = preprocessing.create_sequences(X_train, y_train, window_size)
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    model = build_tcn_model(input_shape, filters, kernel_size, dilations, dropout_rate, dense_units)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_seq, y_seq,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[early_stopping],
                        verbose=1)
    return model, history

def evaluate_tcn_model(model, X_test: pd.DataFrame, y_test: pd.Series, window_size: int):
    """
    Evaluates the trained TCN model using Mean Squared Error, RMSE, and R² score.
    
    Parameters:
        model (tf.keras.Model): Trained TCN model.
        X_test (pd.DataFrame): Test data for independent variables.
        y_test (pd.Series): True values for the dependent variable.
        window_size (int): Number of time steps per sequence.
    
    Returns:
        metrics (dict): Dictionary containing 'mse', 'rmse', and 'r2'.
    """
    X_seq, y_seq = preprocessing.create_sequences(X_test, y_test, window_size)
    y_pred = model.predict(X_seq)
    
    mse = mean_squared_error(y_seq, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_seq, y_pred)
    return {'mse': mse, 'rmse': rmse, 'r2': r2}

def predict_tcn_model(model, X: pd.DataFrame, y: pd.Series, window_size: int):
    """
    Generates predictions on provided data and returns both predictions and ground truth.
    
    Parameters:
        model (tf.keras.Model): Trained TCN model.
        X (pd.DataFrame): Data for independent variables.
        y (pd.Series): Dependent variable (ground truth).
        window_size (int): Number of time steps per sequence.
    
    Returns:
        y_pred (np.ndarray): Flattened array of model predictions.
        y_true (np.ndarray): Flattened array of actual target values.
    """
    X_seq, y_seq = preprocessing.create_sequences(X, y, window_size)
    y_pred = model.predict(X_seq)
    return y_pred.flatten(), y_seq.flatten()