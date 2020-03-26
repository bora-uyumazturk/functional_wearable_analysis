# Data Descriptions

### y_and_static.csv

Contains heart rate and average steps per day for each subject (referred to as `static_data` in `generate_data.py` files for figures).

Columns:
  - `y`: average resting heart rate 
  - `Steps.day.AVG`: average steps per day 
  
The index is the patient ID. 

### day_hour.csv

Each column is an hour of the day. Index is patient ID. Each entry is average accelerometer reading over the course of that hour.

### week_hour.csv

Each column is an hour of the week. Index is patient ID. Each entry is average accelerometer reading over the course of that hour.
