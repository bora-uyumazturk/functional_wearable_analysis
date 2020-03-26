"""
File: matrix.py
Author: Lilia
Usage: contains Matrix_Creator class that takes in csv list and merges time data together
"""
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import threading

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
       
        return self.out.T
            
