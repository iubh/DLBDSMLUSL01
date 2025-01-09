# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Categorical features encoding

#%% import libraries
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder

#%% generate sample data
Table = {
    'Customer-ID': [1, 2, 3, 4, 5],
    'Gender':['M', 'F', 'M', 'F', 'F'],
    'Work-type': [1, 2, 2, 3, 3],
    'Client-satisfaction':[3, 0, 4, 3, 5],
    'Number-occupants': [2, 4, 2, 1, 2],
    'Consumption': [70, 140, 65, 40, 65]
    }
TDF = pd.DataFrame(data=Table)

#%% show datatypes
print(TDF.dtypes)
# console output:
# Customer-ID             int64
# Gender                 object
# Work-type               int64
# Client-satisfaction     int64
# Number-occupants        int64
# Consumption             int64
# dtype: object

#%% specify categorical columns as text
TDF['Work-type'] = TDF['Work-type'].astype(str)
TDF['Client-satisfaction'] = TDF['Client-satisfaction'].\
    astype(str)
print(TDF.dtypes)

# console output:
# Customer-ID             int64
# Gender                 object
# Work-type               int64
# Client-satisfaction     int64
# Number-occupants        int64
# Consumption             int64
# dtype: object

#%% encode the categories in the column
# 'Gender' by ones and zeros in dedicated columns
encoder = OneHotEncoder()
one_hot_encoded = encoder.fit_transform(TDF[['Gender']]).toarray()
one_hot_encoded_df = pd.DataFrame(one_hot_encoded,
                        columns=encoder.get_feature_names_out())

# concatenate the orignal and engineered features
TDF = pd.concat([TDF, one_hot_encoded_df], axis=1)

#%% show the resulting column
print(TDF[['Gender', 'Gender_F', 'Gender_M']])
# console output:
#   Gender  Gender_F  Gender_M
# 0      M       0.0       1.0
# 1      F       1.0       0.0
# 2      M       0.0       1.0
# 3      F       1.0       0.0
# 4      F       1.0       0.0

#%% one-hot-encode categorical features
# with the pandas library
to_encode = TDF.drop(columns=['Gender_F', 'Gender_M'])
TDF_1hot = pd.get_dummies(to_encode)

#%% print the column names of the resulting
# dataframe
print(list(TDF_1hot.columns))

# console output:
# ['Customer-ID', 'Number-occupants', 'Consumption',
# 'Gender_F', 'Gender_M', 'Work-type_1', 'Work-type_2',
# 'Work-type_3', 'Client-satisfaction_0', 'Client-satisfaction_3',
# 'Client-satisfaction_4', 'Client-satisfaction_5']


# %%
