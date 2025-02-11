import asyncio

import ml_forecasts.regression_forecast_niv as regression_forecast_niv
import misc.constants as ct
import misc.plotting as plotting

test_start_date = '2025-01-01'
test_end_date = '2025-01-03'

async def main():
    independent_variables_df, dependent_variable_series = await regression_forecast_niv.get_training_data(
        ct.DatabaseNames.NIV_CHASING.value, ct.DatabaseNames.SYSTEM_PROPERTIES.value, ct.TableNames.NIV_FORECAST_TRAINING_DATA.value, 
        test_start_date, test_end_date, ct.TableNames.SYSTEM_IMBALANCE.value, ct.ColumnHeaders.NET_IMBALANCE_VOLUME.value, 3)
    X_train, X_test, y_train, y_test = regression_forecast_niv.split_data(independent_variables_df, dependent_variable_series)
    pipeline = regression_forecast_niv.train_regression_model(X_train, y_train, 'LinearRegression')
    metrics = regression_forecast_niv.evaluate_regression_model(pipeline, X_test, y_test)
    y_predict = pipeline.predict(X_test)
    plotting.plot_predictions_vs_actuals(y_predict, y_test)
    print(metrics)
    
    
asyncio.run(main())