# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Numerical feature scaling

#%% import libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler

#%% generate sample data
Table = {\
    'Customer-ID': [1, 2, 3, 4, 5], \
    'Gender': ['M', 'F', 'M',' F', 'F'], \
    'Work-type': [1, 2, 2, 3, 3], \
    'Client-satisfaction': [3, 0, 4, 3, 5], \
    'Number-occupants': [2, 4, 2, 1, 2], \
    'Consumption': [70, 140, 65, 40, 65]}
TDF = pd.DataFrame(data=Table)

#%% apply Min-Max Scaling on the feature 'Consumption'
MMS = MinMaxScaler().fit_transform(TDF[['Consumption']])
TDF['Consumption'] = MMS

print(TDF['Consumption'])
# console output: 
#      0.30
# 1    1.00
# 2    0.25
# 3    0.00
# 4    0.25

print(TDF['Consumption'].describe())
# console output:
# count    5.000000
# mean     0.360000
# std      0.376497
# min      0.000000
# 25%      0.250000
# 50%      0.250000
# 75%      0.300000
# max      1.000000

#%% apply standardization on the feature 'Consumption'
TDF = pd.DataFrame(data=Table)
ST = StandardScaler().fit_transform(TDF[['Consumption']])
TDF['Consumption'] = ST

print(TDF['Consumption'])
# console output: 
# 0   -0.178174
# 1    1.900524
# 2   -0.326653
# 3   -1.069045
# 4   -0.326653

print(TDF['Consumption'].describe())
# console output:
# Name: Consumption, dtype: float64
# count    5.000000e+00
# mean    -3.330669e-17
# std      1.118034e+00
# min     -1.069045e+00
# 25%     -3.266526e-01
# 50%     -3.266526e-01
# 75%     -1.781742e-01
# max      1.900524e+00

#%% apply Robust Scaling to the column 'Consumption'
RS = RobustScaler().fit_transform(TDF[['Consumption']])
TDF['Consumption'] = RS

print(TDF['Consumption'])
# console output:
# 0     1.0
# 1    15.0
# 2     0.0
# 3    -5.0
# 4     0.0

print(TDF['Consumption'].describe())
# console output:
# Name: Consumption, dtype: float64
# count     5.00000
# mean      2.20000
# std       7.52994
# min      -5.00000
# 25%       0.00000
# 50%       0.00000
# 75%       1.00000
# max      15.00000