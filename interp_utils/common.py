import pandas as pd

def append_row(table: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    return pd.concat([
        table, 
        pd.DataFrame([row], columns=row.index)]
    ).reset_index(drop=True)
