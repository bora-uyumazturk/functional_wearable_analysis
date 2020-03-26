"""Generate data for figure 2 of paper"""
import argparse
import os
import pandas as pd
import numpy as np

from data_processing import average_hour_per_day, average_hour_per_week
from models import LinearFactorizationModel

from utils import *


def get_factorizer(method):
    method_map = {"dft": DFTFactorizer,
                  "nmf": NMFFactorizer,
                  "pca": PCAFactorizer,
                  "kmeans": KMeansFactorizer}
    return method_map[method]
    

def get_timescale_data(timescale, filename):
    if filename is not None:
        if timescale == 'day_hour':
            df = pd.read_csv(filename, index_col=0)
        elif timescale == 'week_hour':
            df = pd.read_csv(filename, index_col=0)

    elif data_dir is not None:
        f = None
        if timescale == 'day_hour':
            f = average_hour_per_day
        elif timescale == 'week_hour':
            f = average_hour_per_week
        csv_template = os.path.join(data_dir, "Basis_{}.csv")
        csvs = [csv_template.format(str(i).zfil(3)) for i in range(1, 44)]

    return df


def cross_validate(model, X, y):
    preds = np.zeros(y.shape)
    n = X.shape[0]
    for i in range(n):
        X_i = X[indices_except(i, n), :]
        y_i = y[indices_except(i, n), :]
        model.fit(X_i, y_i)
        preds[i, 0] = model.predict(X[i, :].reshape((1, -1)))
    return np.abs(preds - y)


def compute_results(k_range, static_data, timescale, method, data_file):
    print("in compute results: {}".format(timescale))
    results_df = pd.DataFrame(columns = ['num_components', 'value'])
    data = get_timescale_data(timescale, data_file)
    print("got data")

    X_static = static_data.drop(['y'], axis='columns').values
    num_static = X_static.shape[1]
    y = static_data.loc[:, 'y'].values.reshape((-1, 1))
    X_ts = data.values

    X = combine(X_static, X_ts)

    X, y = remove_nans(X, y)

    for k in k_range:
        print("getting method: {}".format(method))
        model_fn = LinearFactorizationModel
        model = model_fn(k, method, num_static)
        df_k = pd.DataFrame(columns = ['num_components', 'value'])
        df_k.loc[:, 'value'] = cross_validate(model, X, y).reshape((-1))
        df_k.loc[:, 'num_components'] = k
        results_df = pd.concat([results_df, df_k], axis=0, ignore_index=True)

    return results_df 


def main(out_dir, static_data, day_hour_file, week_hour_file):
    make_directory(out_dir)
    methods = ['dft', 'nmf', 'pca', 'kmeans']
    timescales = ['day_hour', 'week_hour']
    static_df = pd.read_csv(static_data, index_col=0)
    k_range = range(13)

    for t, f in zip(timescales, [day_hour_file, week_hour_file]):
        if t == 'day_hour':
            k_range = range(13)
        if t == 'week_hour':
            k_range = range(25)
        for m in methods:
            df = compute_results(k_range, static_df, t, m, f)
            df.to_csv(os.path.join(out_dir, "{}_{}.csv".format(t, m)))
        #df = compute_results(k_range, static_df, t, methods, f)
        #df.to_csv(os.path.join(out_dir, "{}.csv".format(t))) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default = ".")
    parser.add_argument("--static_data", type=str)
    parser.add_argument("--day_hour_file", type=str, default=None)
    parser.add_argument("--week_hour_file", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    main(args.output_dir, args.static_data, args.day_hour_file, args.week_hour_file)
