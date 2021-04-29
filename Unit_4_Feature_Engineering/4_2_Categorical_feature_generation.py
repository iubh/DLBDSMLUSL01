# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Categorical features generation

#%% import libraries
import pandas as pd
import numpy as np

#%% generate sample data
Table = { \
    'Customer-ID': [1, 2, 3, 4, 5], \
    'Names': [ \
        'Joe B. BARBY 12-05-2019', \
        'Juliette KARB 18-08-2018', \
        'Lucien VAN 05-07-2017', \
        'Danielle G. REB 03-09-2020', \
        'Lydia HAM 09-07-2018'], \
    'Gender': ['M', 'F', 'M', 'F', 'F'], \
    'Work-type': [1, 2, 2, 3, 3], \
    'Client-satisfaction': [3, 0, 4, 3, 5], \
    'Number-occupants': [2, 4, 2, 1, 2], \
    'Consumption': [70, 140, 65, 40, 65]}
TDF = pd.DataFrame(data=Table)

## spliting columns

# %% split information in one column into three
TDF['First-Name'] = TDF.Names.str.split(" ").\
    map(lambda x: x[0])
TDF['Second-Name'] = TDF.Names.str.split(" ").\
    map(lambda x: x[-2])
TDF['Birth-Year'] = TDF.Names.\
    str.split("-", n=2, expand=True)[2]

#%% generate sample data
dat_tidy = pd.DataFrame({'Customer-ID': [1, 2, 3], \
    'Gender': ['M', 'F', 'M']})
dat_non_tidy = pd.DataFrame(\
    {'Customer-ID': [1, 2, 3, 2, 3, 1],
     'Consumption': [70, 50, 65, 57, 69, 73]})

## aggregation and joining

#%% aggregate consumption per customer
dat = dat_non_tidy.groupby('Customer-ID')['Consumption'].\
    agg(['sum'])

# rename column
dat.columns = ['Sum-Consumption']

# join tables
dat = dat_tidy.merge(dat, left_on='Customer-ID', \
    right_index=True, how='left')

# show the resulting table
dat
# console output:

print(dat)
#   Customer-ID	Gender	Sum-Consumption
# 0 1           M	    143
# 1 2           F	    107
# 2 3           M	    134
