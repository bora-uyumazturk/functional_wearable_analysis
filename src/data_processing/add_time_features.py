"""Add time features to collection of csvs holding wearable activity data."""
from datetime import datetime
import concurrent.futures
import argparse
import numpy as np

import pandas as pd

def get_date_information(date_str):
    """Extract data information from datetime string."""
    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S%z')
    return ((date.month % 12)//3, date.month, date.weekday(), date.hour)

def add_datetime_info(sid, base, out):
    """Add datetime information to subject data and save."""
    print("Processing subject {}".format(sid))
    df = pd.read_csv(base.format(sid))
    print("read csv")
    df.columns = ['time', 'hr', 'accel_magnitude', 'skin_temp']

    df.loc[:, 'season'] = None
    df.loc[:, 'month'] = None
    df.loc[:, 'weekday'] = None
    df.loc[:, 'hour'] = None

    df.loc[:, ['season', 'month', 'weekday', 'hour']] = [get_date_information(x) for x in df.time]

    weekday_shift = np.zeros(len(df)+1)
    weekday_shift[-len(df):] = df.weekday
    weekday_shift[0] = weekday_shift[1]
    weekday_shift = weekday_shift[:-1]
    df['weekday_shift'] = weekday_shift

    # indicator column for new week
    df['new_wk'] = 0
    df.loc[(df['weekday']==0) & (df['weekday_shift'] == 6), 'new_wk'] = 1

    # column of week number using cumsum
    df['week_num'] = df.new_wk.cumsum()

    # drop unnecessary columns
    df.drop(['new_wk', 'weekday_shift'], inplace=True, axis=1)

    df.to_csv(out.format(sid))
    print("Subject {} completed".format(sid))
    return 1


def process_dataframes(sids, base, out, func):
    """Add datetime information to subject data in parallel."""
    pool = concurrent.futures.ThreadPoolExecutor(4)
    future_list = []
    for sid in sids:
        #future_list.append(pool.submit(add_datetime_info, sid, base, out))
        future_list.append(pool.submit(func, sid, base, out))
    concurrent.futures.wait(future_list)
    for f in future_list:
        print(f.result())

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--first", type=int)
   parser.add_argument("--last", type=int)
   parser.add_argument("--what", type=str)
   args = parser.parse_args();
   sids = reversed([str(x).zfill(3) for x in range(args.first, args.last+1)])
   out = "../wearables/Basis_Watch_Data_Processed_New/Basis_{}.csv"
   base = "/Users/spare/Desktop/BIODS215/project/wearables/Basis_Watch_Data_Processed/Basis_{}.csv"
   if args.what == "datetime_info":
       func = add_datetime_info
   if args.what == "add_wk_column":
       func = add_wk_column
   
   process_dataframes(sids, base, out, func)
