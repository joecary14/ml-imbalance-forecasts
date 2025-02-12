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
test_end_date = '2025-01-03'
total_observations = 141
window_size = 3

async def main():
    independent_variables_df, dependent_variable_series = await regression_forecast_niv.get_training_data(
        ct.DatabaseNames.NIV_CHASING.value, ct.DatabaseNames.SYSTEM_PROPERTIES.value, ct.TableNames.NIV_FORECAST_TRAINING_DATA.value, 
        test_start_date, test_end_date, ct.TableNames.SYSTEM_IMBALANCE.value, ct.ColumnHeaders.NET_IMBALANCE_VOLUME.value, 3)
    tcn_forecast.run_model(independent_variables_df, dependent_variable_series, window_size)
    
asyncio.run(main())