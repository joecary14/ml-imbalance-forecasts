import asyncpg

postgres_username = "postgres"
postgres_host = "localhost"
postgres_password = "1234567890"
postgres_port = 5432

import asyncpg
import logging

postgres_username = "postgres"
postgres_host = "localhost"
postgres_password = "1234567890"
postgres_port = 5432

logging.basicConfig(level=logging.INFO)

async def try_connect_to_database(
    database_name,
    user=postgres_username,
    password=postgres_password,
    host=postgres_host,
    port=postgres_port,
    timeout: float = 10.0
):
    """
    Attempt to asynchronously connect to the specified PostgreSQL database.

    Parameters:
        database (str): Database name.
        user (str): Username.
        password (str): Password.
        host (str): Host.
        port (int): Port.
        timeout (float): Connection timeout duration in seconds.

    Returns:
        asyncpg.Connection: Connection object if successful, otherwise None.
    """
    try:
        conn = await asyncpg.connect(
            database=database_name,
            user=user,
            password=password,
            host=host,
            port=port,
            timeout=timeout
        )
        logging.info(f"Connected to database '{database_name}' successfully.")
        return conn
    
    except Exception as e:
        logging.error(f"Failed to connect to database '{database_name}': {e}")
        return None