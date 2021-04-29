# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Categorical features encoding

#%% import libraries
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder

#%% generate sample data
Table = { \
    'Customer-ID': [1, 2, 3, 4, 5], \
    'Gender':['M', 'F', 'M', 'F', 'F'], \
    'Work-type': [1, 2, 2, 3, 3], \
    'Client-satisfaction':[3, 0, 4, 3, 5], \
    'Number-occupants': [2, 4, 2, 1, 2], \
    'Consumption': [70, 140, 65, 40, 65]}
TDF = pd.DataFrame(data=Table)

#%% 
# encode the categories in the column
# 'Gender' by numbers
TDF['Gender'] = LabelEncoder().fit_transform(TDF['Gender'])

# show the resulting column
TDF['Gender']
# console output:
# 0    1
# 1    0
# 2    1
# 3    0
# 4    0

#%% one-hot-encoding
TDF = pd.DataFrame(data=Table)

print(TDF.dtypes)
# console output:
# Customer-ID             int64
# Gender                 object
# Work-type               int64
# Client-satisfaction     int64
# Number-occupants        int64
# Consumption             int64

#%% specify categorical columns as text
TDF['Work-type'] = TDF['Work-type'].\
    astype(str)
TDF['Client-satisfaction'] = TDF['Client-satisfaction'].\
    astype(str)
print(TDF.dtypes)

# console output:
# Customer-ID             int64
# Gender                 object
# Work-type              object
# Client-satisfaction    object
# Number-occupants        int64
# Consumption             int64

#%% one-hot-encode categorical features
TDF_1hot = pd.get_dummies(TDF)

#%%
# print the column names of the resulting
# dataframe
print(list(TDF_1hot.columns))

# console output:
# ['Customer-ID',
#  'Number-occupants',
#  'Consumption',
#  'Gender_F',
#  'Gender_M',
#  'Work-type_1',
#  'Work-type_2',
#  'Work-type_3',
#  'Client-satisfaction_0',
#  'Client-satisfaction_3',
#  'Client-satisfaction_4',
#  'Client-satisfaction_5']
