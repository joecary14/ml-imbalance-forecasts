import asyncio

import ml_forecasts.regression_forecast_niv as regression_forecast_niv
import ml_forecasts.lstm_forecast_niv as lstm_forecast_niv
import ml_forecasts.preprocessing as preprocessing
import ml_forecasts.tcn_forecast_niv as tcn_forecast_niv
import ml_forecasts.gru_forecast_niv as gru_forecast_niv
import ml_forecasts.feature_selection as feature_selection
import ml_forecasts.tune_hyperparameters as tune_hyperparameters

import misc.constants as ct
import misc.plotting as plotting
import numpy as np

from sklearn.metrics import r2_score

test_start_date = '2024-09-24'
test_end_date = '2025-02-07'
number_of_rows_to_drop = 0
window_size = 48
test_split = 0.2
number_of_lstm_layers = 3
lstm_units_per_layer = 128
dropout_rate = 0.2
dense_units = 20
batch_size = 31
epochs = 30
validation_split = 0.2
tcn_filters = 64
tcn_kernel_size = 8
tcn_dilations = [1,2,4,8]
collinearity_threshold = 0.8

hyperparameter_space = {
    'number_of_lstm_layers': (1, 3),
    'lstm_units_per_layer': (64, 256),
    'dropout_rate': (0.1, 0.3),
    'dense_units': (10, 30),
}
max_iterations = 10

async def main():
    independent_variables_df, dependent_variable_series = await preprocessing.get_training_data(
        ct.DatabaseNames.NIV_CHASING.value, ct.DatabaseNames.SYSTEM_PROPERTIES.value, ct.TableNames.NIV_FORECAST_TRAINING_DATA.value, 
        test_start_date, test_end_date, ct.TableNames.SYSTEM_IMBALANCE.value, ct.ColumnHeaders.NET_IMBALANCE_VOLUME.value, number_of_rows_to_drop)
    
    feature_selection.perform_feature_selection(independent_variables_df, dependent_variable_series, collinearity_threshold)
    
    regression_forecast_niv.run_regression_model(independent_variables_df, dependent_variable_series, 'LinearRegression', test_split)
    
    gru_forecast_niv.run_model(independent_variables_df, dependent_variable_series, window_size, test_split, number_of_lstm_layers, lstm_units_per_layer, dropout_rate, dense_units, batch_size, epochs, validation_split)
    
    lstm_forecast_niv.run_model(independent_variables_df, dependent_variable_series, window_size, test_split, number_of_lstm_layers, lstm_units_per_layer, dropout_rate, dense_units, batch_size, epochs, validation_split)
    
    tcn_forecast_niv.run_model(independent_variables_df, dependent_variable_series, window_size, test_split, tcn_filters, tcn_kernel_size, tcn_dilations, dropout_rate, dense_units, batch_size, epochs, validation_split)
    
    best_hp, best_model, best_loss = tune_hyperparameters.bayesian_optimization_tuning_model(lstm_forecast_niv.build_lstm_model, hyperparameter_space, 
        independent_variables_df, dependent_variable_series, test_split, window_size, batch_size, epochs, validation_split, max_iterations)
    print(f"Best hyperparameters: {best_hp}")
    
asyncio.run(main())