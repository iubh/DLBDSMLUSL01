# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Categorical feature imputation

#%% import libraries
import pandas as pd

#%% generate sample data
Table = { \
    'Customer-ID': [1, 2, 3, 4, 5], \
    'Gender':['M', pd.NA, 'M', 'F', 'F'], \
    'Work-type': [1, 2, 2, 3, 3], \
    'Client-satisfaction':[3, 0, 4, 3, 5], \
    'Number-occupants': [2, 4, 2, 1, 2], \
    'Consumption': [70, 140, 65, 40, 65]}
TDF = pd.DataFrame(data=Table)

#%% print the number of missing values per column
print(TDF.isnull().sum())
# console output:
# Customer-ID            0
# Gender                 1
# Work-type              0
# Client-satisfaction    0
# Number-occupants       0
# Consumption            0

#%% simple imputation by mode
TDF = TDF.fillna(TDF['Gender'].value_counts().index[0])
print(TDF.isnull().sum())
# console output:
# Customer-ID            0
# Gender                 0
# Work-type              0
# Client-satisfaction    0
# Number-occupants       0
# Consumption            0
