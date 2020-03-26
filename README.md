# functional_wearable_analysis

Analyzing wearable activity using functional methods and matrix factorization.

## Data

Aggregated data used for analysis is located in `data`. 

Original data can be downloaded at http://ipop-data.stanford.edu/wearable_data/Stanford_Wearables_data.tar

Once downloaded, to produce the aggregate activity data, follow these steps:

1. Add time information `src/data_processing/add_time_features.py`
2. Aggregate on weekly and daily scale using `src/data_processing/aggregate.py`

## Analysis

To recreate figures used in the paper, run code in corresponding folder in `paper_code`. For each figure, follow these steps:

1. Run `generate_data.py`. These use the aggregated weekly and daily accelerometer data as well as static data in `data` to to run analysis and produce data for plotting
2. Run `plots.Rmd` (R Markdown) or `plots.ipynb` (jupyter notebook) (depending on the figure).

Note: for fig1 and fig3, you must change `DATA_DIR` at the beginning of the notebook to location of data csv files with time features added (i.e. after running `add_time_features.py`).
