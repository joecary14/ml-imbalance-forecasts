import asyncio

import ml_forecasts.regression_forecast_niv as regression_forecast_niv
import ml_forecasts.lstm_forecast_niv as lstm_forecast_niv
import ml_forecasts.preprocessing as preprocessing
import ml_forecasts.tcn_forecast as tcn_forecast

import misc.constants as ct
import misc.plotting as plotting
import numpy as np

from sklearn.metrics import r2_score

test_start_date = '2025-01-01'
test_end_date = '2025-02-07'
window_size = 48*7
test_split = 0.2
number_of_lstm_layers = 3
lstm_units_per_layer = 128
dropout_rate = 0.2
dense_units = 20
batch_size = 18
epochs = 100
validation_split = 0.2


async def main():
    independent_variables_df, dependent_variable_series = await regression_forecast_niv.get_training_data(
        ct.DatabaseNames.NIV_CHASING.value, ct.DatabaseNames.SYSTEM_PROPERTIES.value, ct.TableNames.NIV_FORECAST_TRAINING_DATA.value, 
        test_start_date, test_end_date, ct.TableNames.SYSTEM_IMBALANCE.value, ct.ColumnHeaders.NET_IMBALANCE_VOLUME.value, 3)
    lstm_forecast_niv.run_model(independent_variables_df, dependent_variable_series, 
        window_size, test_split, number_of_lstm_layers, lstm_units_per_layer, dropout_rate, dense_units, batch_size, epochs, validation_split)
    
asyncio.run(main())