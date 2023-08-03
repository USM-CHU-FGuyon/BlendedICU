import pandas as pd


def compute_offset(df, col_intime, col_measuretime):
    # TODO: move this to its class in next code update.
    df[col_intime] = pd.to_datetime(df[col_intime])
    df[col_measuretime] = pd.to_datetime(df[col_measuretime])
    df[col_measuretime] = (df[col_measuretime] - df[col_intime]).dt.total_seconds()
    return df.drop(columns=[col_intime])
