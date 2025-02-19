import pandas as pd
import os
import tempfile

def open_excel_without_saving(df: pd.DataFrame):
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        temp_file_path = tmp.name
        df = df.map(lambda x: x.replace(tzinfo=None) if isinstance(x, pd.Timestamp) else x)
        df.to_excel(temp_file_path, index=False)
    
    os.system(f'open "{temp_file_path}"')
