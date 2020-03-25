"""
File: make_matrix.py
Author: Lilia
Usage: --outpath <file to write to>
	make sure to edit process_fn directly
"""
import pandas as pd
import numpy as np
import argparse
import os

from matrix import Matrix_Creator

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
    parser.add_argument("--aggregator", type=str)
    parser.add_argument("--outpath", type=str)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=43)
    args = parser.parse_args()

    if not os.path.exists(args.csv_dir):
        raise(Exception("csv dir does not exist"))
        return
    
    csv_template = os.path.join(args.csv_dir, "Basis_{}.csv")
    csvs = [csv_template.format(str(i).zfill(3)) for i in range(args.first, args.last+1)]
    if args.aggregator == "day_hour":
        f = average_hour_per_day
    elif args.aggregator == "week_hour":
        f = average_hour_per_week
    else:
        raise(Exception("invalid aggregator"))
        return 
    mtrx = Matrix_Creator(csvs, f, 'hour')
    mtrx.process(args.outpath, merge=True)

if __name__ == "__main__":
    main()
