from glob import glob
import pandas as pd
from datetime import datetime as dt
from tqdm import tqdm
import matplotlib.pyplot as plt

def loadIntoOneFile(logging=False):
    if logging:
        p = lambda _: print(_)
    else:
        p = lambda _: None
    logging = not logging
    # Get all the file names
    file_names = glob('data/*.csv')
    # Loop through the file names
    data = []
    # load and clean data
    for file_name in tqdm(file_names, disable=logging):
        name = file_name.split(f"\\")[1].split(".")[0]
        # Read the file
        df = pd.read_csv(file_name)
        df.drop('machineId', axis=1, inplace=True)
        if name == 'pressure':
            df = df.reindex(columns=['_time', 'sensor', 'pressure'])
        data += df.values.tolist()
    # parse time data
    for index, value in tqdm(enumerate(data), total=3947991, disable=logging):
        date_time: str = value[0]
        try:
            if len(date_time) > 24:
                date_time = date_time[:24]
            if len(date_time) == 20:
                date_time = date_time.replace("Z", ".000Z")
            while len(date_time) < 24:
                date_time = date_time.replace("Z", "0Z")
            data[index][0] = dt.fromisoformat(date_time[:-1])
        except ValueError as e:
            p(data[index-2])
            p(f"{index=}, {value=}, {len(value[0])=}")
            p(f"{date_time=} {len(date_time)=}\n")
            raise ValueError(e)
    df = pd.DataFrame(data, columns=['time', 'device', 'value'])
    df['device'] = df['device'].fillna('pressure0')
    devices = df['device'].unique()
    data = df.values.tolist()
    # Create empty dataframe with column names
    cols = ['time'] + devices.tolist()
    df = []
    indexLookup = {}
    for i, device in enumerate(cols):
        indexLookup[device] = i
    # loop through the lists
    for time, device, value in tqdm(data, disable=logging):
        temp = [0]*len(cols)
        temp[0] = time
        try:
            temp[indexLookup[device]] = value
        except Exception as e:
            print(time, device, value)
            raise e
        df.append(temp)

    df = pd.DataFrame(df, columns=cols)
    df.sort_values(by="time", ascending=True, inplace=True)
    df = df.drop_duplicates()
    p(df.head())
    df.to_csv('combined.csv', index=False)
    return df

def interpolate(df: pd.DataFrame|str):
    if isinstance(df, str):
        df = pd.read_csv(df)
    print(len(df))
    # to_fill = df.columns.tolist()[1:]
    df['time'] = pd.DatetimeIndex(df['time'])
    df = df.replace(to_replace=0, method='bfill')
    df = df.replace(to_replace=0, method='ffill')
    df = df.groupby(
        pd.Grouper(key='time', freq='10S')
    ).mean()
    df.dropna(inplace=True)
    df = (df - df.mean()) / df.std()
    df.dropna(axis=1, inplace=True, how='all')
    df.fillna(0, inplace=True)
    df.to_csv('pre_processed.csv', index=True)
    return df

if __name__ == '__main__':
    # df = loadIntoOneFile(logging=True)
    df = interpolate('combined.csv')
    print(len(df))
    print(df.head())
    print(df.tail())
