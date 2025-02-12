import asyncio

import ml_forecasts.regression_forecast_niv as regression_forecast_niv
import ml_forecasts.neural_network_forecast_niv as neural_network_forecast_niv
import misc.constants as ct
import misc.plotting as plotting

test_start_date = '2025-01-01'
test_end_date = '2025-01-03'

async def main():
    independent_variables_df, dependent_variable_series = await regression_forecast_niv.get_training_data(
        ct.DatabaseNames.NIV_CHASING.value, ct.DatabaseNames.SYSTEM_PROPERTIES.value, ct.TableNames.NIV_FORECAST_TRAINING_DATA.value, 
        test_start_date, test_end_date, ct.TableNames.SYSTEM_IMBALANCE.value, ct.ColumnHeaders.NET_IMBALANCE_VOLUME.value, 3)
    x_seq, y_seq = neural_network_forecast_niv.create_sequences(independent_variables_df, dependent_variable_series, 3)
    model = neural_network_forecast_niv.build_lstm_model((x_seq.shape[1], x_seq.shape[2]), 50, 0.2, 20)
    model, history = neural_network_forecast_niv.train_lstm_model(independent_variables_df, dependent_variable_series, 3)
    y_pred = model.predict(x_seq)
    metrics = neural_network_forecast_niv.evaluate_lstm_model(model, independent_variables_df, dependent_variable_series, 3)
    
    
asyncio.run(main())