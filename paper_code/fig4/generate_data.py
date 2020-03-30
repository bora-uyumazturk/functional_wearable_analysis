"""
Generate data for figure 4, csvs with DFT coefficents for each
timescale:
    - dft_week.csv
    - dft_day.csv 
"""
import numpy as np
import pandas as pd

from utils import transform_time_series

if __name__ == "__main__":
    # read data
    Xw = pd.read_csv('../../data/week_hour.csv', index_col=0)
    Xw.dropna(axis='rows', inplace=True)
    Xd = pd.read_csv('../../data/day_hour.csv', index_col=0)
    Xd.dropna(axis='rows', inplace=True)
    
    # factorize for each time scale
    Xw_out = transform_time_series(Xw, 'dft')
    Xd_out = transform_time_series(Xd, 'dft')

    # save data
    Xw_out.to_csv('dft_week.csv')
    Xd_out.to_csv('dft_day.csv')
    
