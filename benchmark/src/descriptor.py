import numpy as np
import pandas as pd
from abench.store.data_management import explore_csv_hierarchy,augment_csvs_with_metadata
from sklearn.metrics.pairwise import pairwise_distances


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



def add_label(df):
    df['attacked'] = np.repeat(0,len(df))
    return(df)


# marco Time alignement
def time_alignement(df):
    df['time_aligned'] = np.round(df['timestamp'].to_numpy()/100)
    return df


def mark_attack(df:pd.DataFrame,
                *,
                weight: float = 0.5):
    """Relabeling the attacked window with a predefined attack weight

    Args:
        df (pd.DataFrame): input window
        weight (float, optional): attack weight, if it's greater than the predefined attack weight,
                                    this window is considered as an attacked window. Defaults to 0.5.

    Returns:
        Boolean: labeled window
    """

    if 'attacked' not in df.columns:
        df['attacked'] = np.repeat(0,len(df))
    
    avg = df['attacked'].astype(bool).sum()/len(df)
    if avg > weight:
        res = True
    else:
        res = False
    return res


def mark_zone(
            df:pd.DataFrame,
            zone: [list],
            *,
            col: str = 'edge'
            ):
    """relabeling the window, check if the window is in the specified zones

    Args:
        df (pd.DataFrame): input window 
        zone (list): the specified zones , format should be [[]]
        col (str, optional): the targeted column. Defaults to 'edge'.

    Returns:
        the classification of zone
    """

    new_df = df[col].copy()
    for i in range(len(zone)+1):
        if i == len(zone):
            c = sum(zone,[])
            new_df = new_df.where((df[col].isin(c)),0)
            break
        new_df = new_df.where(~(df[col].isin(zone[i])),i+1)
    
    res = new_df.value_counts().index.to_numpy()[0]
    
    return int(res) 




def calculate_neighbors(
    df: pd.DataFrame,
    *,
    group_col: str | list = 'time_aligned', 
    r: float = 20.):
    """calculate the number of neighbors within a range

    Args:
        df (pd.DataFrame): Input DataFrame
        group_col (str | list, optional): groupby columns. Defaults to 'time_aligned'.
        r (float, optional): prefinded search range. Defaults to 20..

    Returns:
        df: output DataFrame
    """
    # group all vehicles in the scene at a moment 
    for k, sub in df.groupby(group_col):
        sub = sub.drop_duplicates(['Id'])
        X = sub.loc[:,['positionX','positionY']].to_numpy()
        distance_matrix = pairwise_distances(X, metric='euclidean')

        num_neighbor = distance_matrix < r
        num_neighbor = np.count_nonzero(num_neighbor,axis=0)
        df.loc[sub.index,'neighbors'] = num_neighbor

    df = df.ffill()
    return df