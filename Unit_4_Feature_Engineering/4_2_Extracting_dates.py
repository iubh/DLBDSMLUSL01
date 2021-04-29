# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Extracting dates

#%% import libraries
import pandas as pd
from datetime import date

#%% generate sample data
Table = {'Customer-ID': [1, 2, 3, 4, 5], \
    'Date': ['12-05-2019', '18-08-2018', \
        '05-07-2017', '03-09-2020', '09-07-2018'], \
    'Consumption': [70, 140, 65, 40, 65]}
TDF = pd.DataFrame(data=Table)

#%% convert 'Date'-column to date format
TDF['Date'] = pd.to_datetime(TDF.Date, format="%d-%m-%Y")

# extract the year
TDF['year'] = TDF['Date'].dt.year

# extract the month
TDF['month'] = TDF['Date'].dt.month

# extract quarter of the year
TDF['quarter'] = TDF['Date'].dt.quarter

# show resulting table
print(TDF)

# console output:
# 	Customer-ID	Date	    Consumption	year	month	quarter
# 0	1	        2019-05-12	70	        2019	5   	2
# 1	2	        2018-08-18	140	        2018	8   	3
# 2	3	        2017-07-05	65	        2017	7   	3
# 3	4	        2020-09-03	40	        2020	9   	3
# 4	5	        2018-07-09	65	        2018	7   	3

#%% extract passed years 
years_diff = date.today().year - TDF['Date'].dt.year
TDF['passed_years'] = years_diff

#%% extract passed months 
months_diff = (date.today().year - \
    TDF['Date'].dt.year) * 12 + \
        date.today().month - \
            TDF['Date'].dt.month
TDF['passed_months'] = months_diff

#%% extract the name of weekday
TDF['day_name'] = TDF['Date'].dt.day_name()

#%% extract the day of the week
TDF['dow'] = pd.to_datetime(TDF['Date']).dt.dayofweek

#%% extract weekends
TDF['weekend'] = TDF['dow'].\
    map(lambda x: 0 if x < 5 else 1)
