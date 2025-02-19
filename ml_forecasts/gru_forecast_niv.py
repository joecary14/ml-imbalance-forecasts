import pandas as pd

import ml_forecasts.preprocessing as preprocessing
import misc.plotting as plotting

from math import sqrt
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import GRU, Dense, Dropout, Input
from keras._tf_keras.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def run_model(independent_variables_df, dependent_variable_series, window_size, 
              test_split, number_of_gru_layers, gru_units_per_layer, dropout_rate, dense_units, batch_size, epochs, validation_split):
    preprocessing.check_sufficient_data(independent_variables_df, batch_size, epochs)
    scaled_independent_variables_df = preprocessing.scale_independent_variables(independent_variables_df)
    scaled_dependent_variable, dependent_variable_scaler = preprocessing.scale_dependent_variable(dependent_variable_series)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_independent_variables_df, scaled_dependent_variable, test_size=test_split, shuffle=False)
    
    model, history = train_gru_model(
        X_train, y_train, window_size, number_of_gru_layers, gru_units_per_layer, dropout_rate, dense_units, batch_size, epochs, validation_split)
    
    test_metrics = evaluate_gru_model(model, X_test, y_test, window_size)
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

def build_gru_model(input_shape, number_of_gru_layers, gru_units_per_layer, dropout_rate, dense_units):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Add GRU layers with return_sequences=True for all but the final GRU layer
    for _ in range(number_of_gru_layers - 1):
        model.add(GRU(gru_units_per_layer, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    # Final GRU layer without return_sequences
    model.add(GRU(gru_units_per_layer))
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Single output for regression
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_gru_model(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    window_size: int,
                    number_of_gru_layers,
                    gru_units_per_layer,
                    dropout_rate,
                    dense_units,
                    batch_size,
                    epochs,
                    validation_split):
    
    X_seq, y_seq = preprocessing.create_sequences(X_train, y_train, window_size)
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    
    model = build_gru_model(
        input_shape=input_shape,
        number_of_gru_layers=number_of_gru_layers,
        gru_units_per_layer=gru_units_per_layer,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_seq, y_seq,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split,
                        callbacks=[early_stopping],
                        verbose=1)
    return model, history

def evaluate_gru_model(model, X_test: pd.DataFrame, y_test: pd.Series, window_size: int):   
    X_seq, y_seq = preprocessing.create_sequences(X_test, y_test, window_size)
    y_pred = model.predict(X_seq)
    
    mse = mean_squared_error(y_seq, y_pred)
    rmse = sqrt(mse)
    return {'mse': mse, 'rmse': rmse}

def save_gru_model(model, file_path: str):
    model.save(file_path)
    print(f"Model saved to {file_path}")

def load_gru_model(file_path: str):
    model = load_model(file_path)
    print(f"Model loaded from {file_path}")
    return model

def build_model_for_tuning(
    hp, independent_variables_df, dependent_variable_series, window_size, test_split, gru_range: list[int], gru_units_per_layer, 
    dropout_range: list[float], dense_range: list[int]):
    scaled_independent_variables_df = preprocessing.scale_independent_variables(independent_variables_df)
    scaled_dependent_variable, dependent_variable_scaler = preprocessing.scale_dependent_variable(dependent_variable_series)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_independent_variables_df, scaled_dependent_variable, test_size=test_split, shuffle=False)
    
    return setup_model_for_tuning(hp, X_train, y_train, window_size, gru_range, gru_units_per_layer, dropout_range, dense_range)

def setup_model_for_tuning(hp, X_train, y_train, window_size, gru_range: list[int], gru_units_per_layer, dropout_range: list[float], dense_range: list[int]):
    min_gru, max_gru = min(gru_range), max(gru_range)
    min_dropout, max_dropout = min(dropout_range), max(dropout_range)
    min_dense, max_dense = min(dense_range), max(dense_range)
    number_of_gru_layers = hp.Int('number_of_gru_layers', min_value=min_gru, max_value=max_gru, step=1)
    gru_units_per_layer = hp.Choice('gru_units_per_layer', values=gru_units_per_layer)
    dropout_rate = hp.Float('dropout_rate', min_value=min_dropout, max_value=max_dropout, step=0.1)
    dense_units = hp.Int('dense_units', min_value=min_dense, max_value=max_dense, step=16)
    
    X_seq, y_seq = preprocessing.create_sequences(X_train, y_train, window_size)
    input_shape = (X_seq.shape[1], X_seq.shape[2])
    
    return build_gru_model(
        input_shape=input_shape,
        number_of_gru_layers=number_of_gru_layers,
        gru_units_per_layer=gru_units_per_layer,
        dropout_rate=dropout_rate,
        dense_units=dense_units
    )
