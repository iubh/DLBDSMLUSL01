# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Numerical interaction features

#%% import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

#%% generate sample data
Table = {\
    'Customer-ID': [1, 2, 3, 4, 5], \
    'Gender': ['M','F','M','F','F'], \
    'Work-type': [1, 2, 2, 3, 3], \
    'Client-satisfaction':[3, 0, 4, 3, 5], \
    'Number-occupants': [2, 4, 2, 1, 2],
    'Consumption':[70, 140, 65, 40, 65]}
TDF = pd.DataFrame(data=Table)

#%% create and fit a polynomial feature creator
pf =  PolynomialFeatures(\
    degree = 2, \
    interaction_only=True, \
    include_bias = False).\
        fit(TDF[['Number-occupants', 'Consumption']])

#%% apply the polynomial feature creator to the data
int_feat = pf.transform(TDF[['Number-occupants', \
    'Consumption']])

print(int_feat)
# console output:
# [[  2.  70. 140.]
#  [  4. 140. 560.]
#  [  2.  65. 130.]
#  [  1.  40.  40.]
#  [  2.  65. 130.]]

#%%
# convert the generated interaction feature array
# to a dataframe
int_feat = pd.DataFrame(int_feat, \
    columns=['Number-occupants', 'Consumption', \
        'nOcc_x_Conspt'])

#%% append generated interaction feature to dataframe
TDF = pd.concat([TDF, int_feat], axis=1)
