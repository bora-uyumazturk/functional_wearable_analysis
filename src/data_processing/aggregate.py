"""
File: aggregate.py
Author: Lilia and Bora
Usage: --outpath <file to write to>
	make sure to edit process_fn directly
""" 
import pandas as pd
import numpy as np
import argparse
import os

import concurrent.futures
import threading

class Aggregator:

    def __init__(self, csv_list, processing_fn, resolution_type):
        """
        csv_list : list of csvs to read,
        processing_fn : processing function with args
                        csv (csv path), self.resolution_type
        resolution_type : resolution for processing function
        """
        self.csv_list = csv_list
        self.processing_fn = processing_fn
        self.resolution_type = resolution_type
        self.out = None
        self.lock = threading.Lock()


    def process_csv(self, csv):
        df_cur, sid = self.processing_fn(csv, self.resolution_type)
        self.lock.acquire()
        print("processing: {}".format(sid))
        if self.out is None:
            self.out = df_cur
            self.out.columns = [sid]
        else:
            df_cur.columns = [sid]
            self.out = self.out.merge(df_cur, how='outer', left_index=True, right_index=True)
        print("completed: {}".format(sid))
        self.lock.release()


    def process(self, outpath=None, merge=False):
        """Processes and saves matrix to outpath."""
        print("Processing csvs...")

        pool = concurrent.futures.ThreadPoolExecutor(4)
        future_list = []
        for csv in self.csv_list:
            future_list.append(pool.submit(self.process_csv, csv))
        concurrent.futures.wait(future_list)

        self.out = self.out.T
        self.out.sort_index(inplace=True)
        if isinstance(self.out.columns, pd.MultiIndex):
            self.out.columns = range(len(self.out.columns))

        if outpath is not None:
            self.out.to_csv(outpath)
            print("Saved matrix to %s." %outpath)


def average_hour_per_day(csv, resolution='hour'):
    """returns df of average accel_magnitude for each hour of day"""
    df = pd.read_csv(csv) 
    subject_id = csv[-7:-4] # extract id
    if "hour" in df.columns:
        result = pd.DataFrame(df.groupby("hour").accel_magnitude.mean())
    else:
        raise(Exception("DataFrame lacks hour columnns"))
    return result, subject_id


def average_hour_per_week(csv, resolution='hour'):
    """returns df of average hour per week. assumes csv has
    columns corresponding to 'weekday', 'hour'"""
    df = pd.read_csv(csv) 
    subject_id = csv[-7:-4] # extract id
    if "weekday" not in df.columns or "hour" not in df.columns:
        raise(Exception("DataFrame lacks weekday and/or hour columns"))
    result = pd.DataFrame(df.groupby(["weekday", "hour"]).accel_magnitude.mean()) 
    return result, subject_id


def main():
    """writes matrix of average hour activity"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir", type=str)
    parser.add_argument("--timescale", type=str)
    parser.add_argument("--outpath", type=str)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=43)
    args = parser.parse_args()

    if not os.path.exists(args.csv_dir):
        raise(Exception("csv dir does not exist"))
        return
    
    csv_template = os.path.join(args.csv_dir, "Basis_{}.csv")
    csvs = [csv_template.format(str(i).zfill(3)) for i in range(args.first, args.last+1)]
    if args.timescale == "day_hour":
        f = average_hour_per_day
    elif args.timescale == "week_hour":
        f = average_hour_per_week
    else:
        raise(Exception("invalid aggregator"))
        return 
    aggtr = Aggregator(csvs, f, 'hour')
    aggtr.process(args.outpath, merge=True)


if __name__ == "__main__":
    main()
