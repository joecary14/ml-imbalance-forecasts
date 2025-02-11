import asyncio
import asyncpg
import pandas as pd

import misc.constants as ct

postgres_username = "postgres"
postgres_host = "localhost"
postgres_password = "1234567890"
postgres_port = 5432

async def fetch_table_from_database(
    db_connection: asyncpg.Connection,
    table: str,
) -> pd.DataFrame:

    query = f"SELECT * FROM {table};"
    records = await db_connection.fetch(query)

    data = [dict(record) for record in records]
    df = pd.DataFrame(data)
    return df

async def get_values_by_dates_and_periods(db_connection, table_name, settlement_dates_and_periods, columns_to_download):
    """
    Asynchronously fetch data from the specified table, downloading the given columns
    filtered by a list of settlement dates and periods.
    """
    primary_key = ct.ColumnHeaders.DATE_PERIOD_PRIMARY_KEY.value
    if primary_key not in columns_to_download:
        columns_to_download = columns_to_download + [primary_key]

    download_query, dynamic_params = generate_download_query_by_column_values(
        table_name, columns_to_download, primary_key, settlement_dates_and_periods)
    
    records = await db_connection.fetch(download_query, *dynamic_params)
    data = [dict(record) for record in records]
    df = pd.DataFrame(data)
    return df

def generate_download_query_by_column_values(table_name, columns_to_download, column_name, values):
    """
    Generate a SQL query string and parameters for downloading selected columns from a table
    where a specific column's value is within a list of given values.

    Parameters:
        table_name (str): Name of the table.
        columns_to_download (list): List of column names to select.
        column_name (str): Column used for filtering (e.g., settlement_date_and_period).
        values (list): List of values to filter on.

    Returns:
        tuple: (query_str, params) where query_str is the SQL query and params is a tuple of parameters.
    """
    
    columns_str = ", ".join([f'"{col}"' for col in columns_to_download])
    table_str = f'"{table_name}"'
    filter_col = f'"{column_name}"'
    
    query_str = f"SELECT {columns_str} FROM {table_str} WHERE {filter_col} = ANY($1)"
    params = (values,)
    
    return query_str, params