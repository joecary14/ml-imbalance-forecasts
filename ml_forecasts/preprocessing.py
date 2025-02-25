import math
import pandas as pd
import numpy as np
import database_interaction.download_from_database as db_downloader
import database_interaction.database_general as db_general
import misc.datetime_functions as datetime_functions
import misc.constants as ct
import misc.excel_handler as excel_handler

from sklearn.preprocessing import StandardScaler

async def get_training_data(database_name, dependent_variable_database_name, independent_variable_table_name, start_date, end_date, 
                            dependent_variable_table_name, dependent_variable_column_to_download, dependent_variable_rows_to_drop=0):
    db_connection = await db_general.try_connect_to_database(database_name)
    settlement_dates_and_periods = datetime_functions.get_list_of_settlement_dates_and_periods(start_date, end_date)
    independent_variables_df = await db_downloader.get_values_by_dates_and_periods(
        db_connection, independent_variable_table_name, settlement_dates_and_periods)
    await db_connection.close()
    db_connection = await db_general.try_connect_to_database(dependent_variable_database_name)
    dependent_variable_df = await db_downloader.get_values_by_dates_and_periods(
        db_connection, dependent_variable_table_name, settlement_dates_and_periods, [dependent_variable_column_to_download])
    
    await db_connection.close()
    
    dependent_variable = order_df_by_settlement_date_and_period(dependent_variable_df)
    dependent_variable = dependent_variable.iloc[dependent_variable_rows_to_drop:]
    independent_variables = order_df_by_settlement_date_and_period(independent_variables_df)
    independent_variables = independent_variables.drop(columns=[ct.ColumnHeaders.DATE_PERIOD_PRIMARY_KEY.value])
    
    dependent_variable_series = pd.Series(dependent_variable[dependent_variable_column_to_download])
    
    return independent_variables, dependent_variable_series

def scale_independent_variables(independent_variables_df):
    scaler = StandardScaler()
    scaled_independent_variables = scaler.fit_transform(independent_variables_df)
    scaled_independent_variables_df = pd.DataFrame(scaled_independent_variables, columns=independent_variables_df.columns)
    
    return scaled_independent_variables_df

def scale_dependent_variable(dependent_variable_series):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dependent_variable_series.values.reshape(-1, 1))
    scaled_series = pd.Series(scaled.flatten(), index=dependent_variable_series.index)
    return scaled_series, scaler

def create_sequences(X: pd.DataFrame, y: pd.Series, window_size: int):
    X_values = X.values
    y_values = y.values
    X_seq, y_seq = [], []
    
    for i in range(len(X_values) - window_size):
        X_seq.append(X_values[i:i+window_size])
        y_seq.append(y_values[i+window_size])
    
    return np.array(X_seq), np.array(y_seq)

def order_df_by_settlement_date_and_period(df):
    df_copy = df.copy()
    df_copy["period"] = (
    df_copy["settlement_date_and_period"]
    .str.split("-")
    .str[-1]
    .astype(int)
    )
    df_copy["date"] = (
    df_copy["settlement_date_and_period"]
    .str.rsplit("-", n=1, expand=True)[0]
    )
    df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
    df_copy.sort_values(by=["date", "period"], inplace=True)
    df_copy.drop(columns=["date", "period"], inplace=True)
    df_copy.reset_index(drop=True, inplace=True)
    
    return df_copy

def drop_dummy_variables(X: pd.DataFrame):
    dummy_columns = X.columns[X.columns.str.contains("settlement|day|month", case=False)]
    X_copy = X.copy()
    return X_copy.drop(columns=dummy_columns)