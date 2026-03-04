#! /bin/python3
import math
import os
import pandas as pd
import numpy as np 
import json 
import sumolib

# retrieve all csv files
def retrieve_csv_files(directory:str
                       ) -> list:
    try:
        csv_files = sorted([file for file in os.listdir(directory) if file.endswith(".csv")])
        return csv_files
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        

# concat all csv files
def read_all_csv(csv_files:list, 
                 path:str
                 ) -> pd.DataFrame:
    res = []
    for f in csv_files[:]:
        df = pd.read_csv(path+f,delimiter=";").dropna(subset=['Id','timestamp'])
        # print(df.shape)
        res.append(df)
    df = pd.concat(res, ignore_index=True)
    return df

  
# Select objects from a class based on a defined rate.
# Return the list of filtered object IDs.
def filtrate_object(df:pd.DataFrame, 
                    cls:str="VEHICLE",
                    rate:float=0.5
                    ) -> list:
    df1 = df.groupby(['Id'],dropna=False)
    vehicle_list = []
    for key,val in df1:
        df2 = val.copy().value_counts(['Id','Class'],normalize=True)
        if cls in df2.index.levels[1]:
            if df2[:,cls].values > rate:
                vehicle_list.append(key[0])
    return vehicle_list


# Select objects by ID list
def select_by_id(df:pd.DataFrame,
                 id_list:list
                 ) -> pd.DataFrame:
    if not id_list:
        raise Exception("id list is empty") 
    df_v = [df[df['Id']==i].copy() for i in id_list]
    df_v = pd.concat(df_v)
    return df_v


# get rsu info
def rsu_info(rsu_json:str):
    with open(rsu_json,"r") as f:
        rsu_info = json.load(f)
    lat = rsu_info["geographicalPosition"]["latitude"]
    lon = rsu_info["geographicalPosition"]["longitude"]
    head =  rsu_info["trueHeading"]
    return lat, lon, head


# revision of data position according to RSU(fused) 
# pos -> [x, y]
def rot_data(df:pd.DataFrame,
             heading:float
             ) -> pd.DataFrame:
    
    new_df = df.copy()
    theta = math.radians(heading)
    rotation = np.matrix([[np.sin(theta),np.cos(theta)],
                        [np.cos(theta),-np.sin(theta)]])
    
    rot_df = new_df.loc[:,['positionX','positionY']].dot(rotation)
    new_df['rot_x'] = rot_df.iloc[:,0]
    new_df['rot_y'] = rot_df.iloc[:,1]
    
    return new_df


# filter all vehicles that are not on the road
def on_road(x,y,
            net:sumolib.net,
            radius:float=0.5
            ):
    edges = net.getNeighboringEdges(x, y, radius)
    # pick the closest edge
    if len(edges) > 0:
        distancesAndEdges = sorted([(dist, edge) for edge, dist in edges], key=lambda x:x[0])
        dist, closestEdge = distancesAndEdges[0]
        return (dist, closestEdge) 
    return (None, None)


# select the time series data within a fixed observation period.
# start: minimal perceived duration (n*period)
# stop: maximal perceived duration (n*period)
# period: 1ms
# segmentation for the long period ? 
def data_select_period(df:pd.DataFrame,
                       *,
                       start:float = 0.,
                       stop:float = 150., #15s
                       scale:float = 100.
                    ):
    """select the time series data within a fixed observation period.

    Args:
        df (pd.DataFrame): Input DataFrame
        start (float, optional): minimal perceived duration (n*period). Defaults to 0..
        stop (float, optional): maximal perceived duration (n*period). Defaults to 150..
        scale: Time scale

    Returns:
        tuple(selected, low, up) : return DataFrames of three intervals
    """
    
    low_list = []
    up_list = []
    v_list = []
    
    for k,v in df.groupby('Id'):
        # There are some duplications, drop duplicated Id at the same time.
        new_v = v.copy().drop_duplicates(['Id','timestamp']).reset_index(drop=True)        
        # round off data for every 100ms which correspond to the sensor collection period
        new_v['ts'] = np.ceil((new_v['timestamp'] - new_v['timestamp'].min())/scale)
        
        if new_v['ts'].max() < start:
            low_list.append(k)
        elif new_v['ts'].max() >= stop:
            up_list.append(k)
        else:
            v_list.append(k)       
        
        # segmentation
    return v_list, low_list, up_list
        


# padding 
def data_padding(df:pd.DataFrame,
                 *,
                 period_start:int = 0,
                 period_stop:int = 50,
                 scale:float = 100.,
                 pad_method:str = 'pad',
                 pad_columns:list = ['timestamp','positionX','positionY','positionZ','VelX','VelY','VelZ','Vel','rot_x','rot_y'],
                 pad_axis = 0
                 ) -> pd.DataFrame:
    """DataFrame interpolation, add colum ['ts'] as sequence timestamp 

    Args:
        df (pd.DataFrame): Input DataFrame
        period_start (int, optional): period start time. Defaults to 0.
        period_stop (int, optional): period end time, if -1, set as the time of the end of sequence. Defaults to 50.
        scale (float, optional): time scale. Defaults to 100..
        pad_method (str, optional): padding method. Defaults to 'pad'.
        pad_columns (list, optional): padding colums. Defaults to ['timestamp','positionX','positionY','positionZ','VelX','VelY','VelZ','Vel','rot_x','rot_y'].
        pad_axis (int, optional): padding axis. Defaults to 0.

    Returns:
        pd.DataFrame: return padded DataFrame
    """
    
    # template_v = pd.DataFrame().reindex_like(df[:1])
    v_list = []
    new_df = df.copy().dropna(how='all')
    
    for _,v in new_df.groupby('Id'):
        new_v = v.copy().sort_values(by=['timestamp'])
        
        # Filter abnormal data (focus on speed, < +3*std, > -3*std)
        c1 = new_v['VelX'].mean() + 3*new_v['VelX'].std()
        c2 = new_v['VelX'].mean() - 3*new_v['VelX'].std()
        c1 = new_v['VelX'] < c1
        c2 = new_v['VelX'] > c2

        c3 = new_v['VelY'].mean() + 3*new_v['VelY'].std()
        c4 = new_v['VelY'].mean() - 3*new_v['VelY'].std()
        c3 = new_v['VelY'] < c3
        c4 = new_v['VelY'] > c4
        new_v = new_v[c1 & c2 & c3 & c4]
        
        if new_v.empty:
            continue
        
        new_v['ts'] = np.ceil((new_v['timestamp'] - new_v['timestamp'].min())/scale)
        # add NaN padding for a fixed period 
        # if period_stop == -1, correspond the max lifetime length
        if period_stop == -1:
            period_stop_f = int(new_v['ts'].max())
        else:
            period_stop_f = period_stop
            
        new_v = new_v.drop_duplicates(['ts']).set_index(['ts'],drop=True).reindex(range(period_start,period_stop_f))
        
        # use pd.DataFrame.interpolate() method ? 
        # depend on the model, how we imterpolate the value
        # pad_columns = ['timestamp','positionX','positionY','positionZ','VelX','VelY','VelZ','Vel','rot_x','rot_y','ts']
        new_v[pad_columns] = new_v[pad_columns].interpolate(method=pad_method, limit_direction='forward', axis=pad_axis)
        tmp_v = new_v.convert_dtypes().ffill().dropna(how='all')
        tmp_v['ts'] = range(period_start,period_stop_f)
        
        if not tmp_v.empty:
            v_list.append(tmp_v)
        
    res = pd.concat(v_list, ignore_index=True)  
    
    return res






# rolling window
def rolling_window(df:pd.DataFrame,
                   *,
                   window_size: int = 50,
                   horizon_start: int = 0,
                   horizon_end: int = -1,
                   groupby: str | list = ['Id'],
                   time_col: str | list = ['ts'],
                   padding: bool = False,
                   stride: int = 50) -> pd.DataFrame :
    """rolling window dataset generator

    Args:
        df (pd.DataFrame): Input dataFrame
        window_size (int, optional): sliding window size. Defaults to 50.
        horizon_start (int, optional): window start position. Defaults to 0.
        horizon_end (int, optional): window end position; if -1, it equals the length of the grouped subset. Defaults to -1.
        groupby (str | list, optional): Used to determine the groups for the groupby. Defaults to ['Id'].
        time_col (str | list, optional): Used to determine the timestamp of the sequences. Defaults to ['ts'].
        padding (bool, optional): Used to fill the missing values; if false, drop the tail. Defaults to False.
        stride (int, optional): rolling window step length. Defaults to 50.

    Returns:
        pd.DataFrame: return a Rolling Window Dataset
    """
    
    window_list = []
    for _, v in df.groupby(groupby):
        new_v = v.sort_values(by=time_col).copy()
        if horizon_end == -1:
            sq_end = len(new_v)
        else:
            sq_end = horizon_end
        
        nb_windows = (sq_end-horizon_start-window_size)//stride
        for n_window in range(nb_windows+1):
            win_start = horizon_start+stride*n_window
            win_end = win_start+window_size
            sliding_window = new_v[win_start:win_end].copy()
            sliding_window['window'] = np.repeat(n_window,len(sliding_window))
            window_list.append(sliding_window)
        
        # if there's a tail
        if padding == True:
            if nb_windows < 1:
                win_start = horizon_start
            else:
                win_start += stride
                
            if win_start < horizon_end:
                sliding_window = new_v[win_start:horizon_end].copy().reset_index().reindex(range(stride),method='ffill')
                sliding_window['window'] = np.repeat(nb_windows,len(sliding_window))
                window_list.append(sliding_window)

    res = pd.concat(window_list, ignore_index=True)   
    
    return res