import argparse
import pandas as pd

from models import DFTFactorizer
from utils import transform_time_series 

def main(source, dest):
    source_df = pd.read_csv(source, index_col=0)
    source_df = source_df.dropna(axis='rows')
    out_df = transform_time_series(source_df, 'dft')
    out_df.to_csv(dest)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--dest", type=str)
    
    args = parser.parse_args()
    main(args.source, args.dest)
