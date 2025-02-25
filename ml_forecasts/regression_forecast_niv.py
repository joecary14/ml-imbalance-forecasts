import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import database_interaction.download_from_database as db_downloader
import database_interaction.database_general as db_general
import misc.datetime_functions as datetime_functions

async def get_training_data(database_name, dependent_variable_database_name, independent_variable_table_name, start_date, end_date, 
                            dependent_variable_table_name, dependent_variable_column_to_download, dependent_variable_rows_to_drop=0):
    db_connection = await db_general.try_connect_to_database(database_name)
    independent_variables_df = await db_downloader.fetch_table_from_database(db_connection, independent_variable_table_name)
    independent_variables_df.drop(columns=["id"], inplace=True)
    settlement_dates_and_periods = datetime_functions.get_list_of_settlement_dates_and_periods(start_date, end_date)
    await db_connection.close()
    db_connection = await db_general.try_connect_to_database(dependent_variable_database_name)
    dependent_variable_df = await db_downloader.get_values_by_dates_and_periods(
        db_connection, dependent_variable_table_name, settlement_dates_and_periods, [dependent_variable_column_to_download])
    
    await db_connection.close()
    
    dependent_variable_df["period"] = (
    dependent_variable_df["settlement_date_and_period"]
    .str.split("-")
    .str[-1]
    .astype(int)
    )
    dependent_variable_df["date"] = (
    dependent_variable_df["settlement_date_and_period"]
    .str.rsplit("-", n=1, expand=True)[0]
    )
    dependent_variable_df["date"] = pd.to_datetime(dependent_variable_df["date"], errors="coerce")
    dependent_variable_df.sort_values(by=["date", "period"], inplace=True)
    dependent_variable_df.drop(columns=["date", "period"], inplace=True)
    dependent_variable_df.reset_index(drop=True, inplace=True)
    dependent_variable_df = dependent_variable_df.iloc[dependent_variable_rows_to_drop:]
    
    dependent_variable_series = pd.Series(dependent_variable_df[dependent_variable_column_to_download])
    
    return independent_variables_df, dependent_variable_series

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def train_regression_model(X_train: pd.DataFrame,
                           y_train: pd.Series,
                           model_type: str = 'RandomForest',
                           **model_params):
    """
    Trains a regression model to predict system outturn.
    
    Parameters:
        X_train (pd.DataFrame): Training data for independent variables.
        y_train (pd.Series): Training data for the dependent variable.
        model_type (str): Type of regression model ('RandomForest' or 'LinearRegression').
        **model_params: Additional keyword arguments to pass to the model constructor.
        
    Returns:
        model (Pipeline): A scikit-learn Pipeline that includes preprocessing and the trained regressor.
    """
    if model_type == 'RandomForest':
        regressor = RandomForestRegressor(**model_params)
    elif model_type == 'LinearRegression':
        regressor = LinearRegression(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_regression_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the regression model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE),
    and R² score on the test dataset.
    
    Parameters:
        model: Trained regression model (or Pipeline).
        X_test (pd.DataFrame): Testing data for independent variables.
        y_test (pd.Series): True values for the dependent variable.
        
    Returns:
        metrics (dict): A dictionary with keys 'mse', 'rmse', and 'r2' containing the evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {'mse': mse, 'rmse': rmse, 'r2': r2}
    return metrics