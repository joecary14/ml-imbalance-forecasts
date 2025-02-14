import pandas as pd

import ml_forecasts.preprocessing as preprocessing
import misc.plotting as plotting

from math import sqrt
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import LSTM, Dense, Dropout, Input
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def run_model(independent_variables_df, dependent_variable_series, window_size, 
              test_split, number_of_lstm_layers, lstm_units_per_layer, dropout_rate, dense_units, batch_size, epochs, validation_split):
    preprocessing.check_sufficient_data(independent_variables_df, batch_size, epochs)
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
    model = Sequential()
    model.add(Input(shape=input_shape))
    for layer in range(number_of_lstm_layers-1):
        model.add(LSTM(lstm_units_per_layer, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    model.add(LSTM(lstm_units_per_layer))
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

def build_model_for_tuning(
    hp, independent_variables_df, dependent_variable_series, window_size, test_split, lstm_range : list[int], lstm_units_per_layer, 
    dropout_range : list[float], dense_range : list[int]):
    scaled_independent_variables_df = preprocessing.scale_independent_variables(independent_variables_df)
    scaled_dependent_variable, dependent_variable_scaler = preprocessing.scale_dependent_variable(dependent_variable_series)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_independent_variables_df, scaled_dependent_variable, test_size=test_split, shuffle=False)
    
    return setup_model_for_tuning(hp, X_train, y_train, window_size, lstm_range, lstm_units_per_layer, dropout_range, dense_range)

def setup_model_for_tuning(hp, X_train, y_train, window_size, lstm_range : list[int], lstm_units_per_layer, dropout_range : list[float], dense_range : list[int]):
    min_lstm, max_lstm = min(lstm_range), max(lstm_range)
    min_dropout, max_dropout = min(dropout_range), max(dropout_range)
    min_dense, max_dense = min(dense_range), max(dense_range)
    number_of_lstm_layers = hp.Int('number_of_lstm_layers', min_value=min_lstm, max_value=max_lstm, step=1)
    lstm_units_per_layer = hp.Choice('lstm_units_per_layer', values=lstm_units_per_layer)
    dropout_rate = hp.Float('dropout_rate', min_value=min_dropout, max_value=max_dropout, step=0.1)
    dense_units = hp.Int('dense_units', min_value=min_dense, max_value=max_dense, step=16)
    X_seq, y_seq = preprocessing.create_sequences(X_train, y_train, window_size)
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    
    return build_lstm_model(
        input_shape=input_shape,
        number_of_lstm_layers=number_of_lstm_layers,
        lstm_units_per_layer=lstm_units_per_layer,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )