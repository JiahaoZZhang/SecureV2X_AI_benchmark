import numpy as np
import pandas as pd
from abench.store.data_management import explore_csv_hierarchy,augment_csvs_with_metadata

def calculate_random_descriptor(df: pd.DataFrame,**kwargs):
    random_class = np.random.randint(0,5,len(df))
    return(random_class)


def macro_id(df):
    df['seqId'] = [i.replace('.csv','_') for i in df['filename'].values + (df.index.values//50).astype(str)]
    return(df)

def calculate_length(
    df: pd.DataFrame,
    *,
    x_col="positionX",
    y_col="positionY",
    time_col="ts",
    assume_sorted: bool = False):
    
    """
    Total polyline length: sum of EuclIdean distances between consecutive points.

    Parameters
    ----------
    traj : pd.DataFrame
        Trajectory slice (one metaId).
    x_col, y_col, time_col : str
        Column names.
    assume_sorted : bool
        If False, sort by time_col insIde the function. Use True when you know
        the input is already sorted by time to avoId re-sorting.

    Returns
    -------
    float
        Total length (same units as x/y).
    """
    g = df if assume_sorted else df.sort_values(time_col)
    x = g[x_col].to_numpy()
    y = g[y_col].to_numpy()
    if x.size < 2:
        return 0.0

    dx = np.diff(x)
    dy = np.diff(y)
    return float(np.hypot(dx, dy).sum())
