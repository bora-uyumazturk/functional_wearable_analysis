"""
File: matrix.py
Author: Lilia
Usage: contains Matrix_Creator class that takes in csv list and merges time data together
"""
import pandas as pd
from tqdm import tqdm

class Matrix_Creator:

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

    def process(self, outpath=None, merge=False):
        """Processes and saves matrix to outpath."""
        print("Processing csvs...")
        out = None
        for csv in tqdm(self.csv_list):
            df_cur, sid = self.processing_fn(csv, self.resolution_type)
            print(df_cur.head())
            if out is None:
                out = df_cur
                out.columns = [sid]
            else:
                df_cur.columns = [sid]
                out = out.merge(df_cur, how='outer',
                                left_index=True, right_index=True)

        if outpath is not None:
            out.T.to_csv(outpath)
            print("Saved matrix to %s." %outpath)

        return out
