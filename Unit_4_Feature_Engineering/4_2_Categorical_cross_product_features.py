# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Categorical cross-product features

#%% import libraries
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

#%% generate sample data
Table = {'Gender': ['M','F','M','F','F','M', 'F', 'M'], \
         'Science-Grant': ['Y', 'N', 'Y', 'N', \
             'Y', 'N', 'N', 'N']}
TDF = pd.DataFrame(data=Table)

#%% one-hot encoding
TDF = pd.get_dummies(TDF)

#%% generate interaction features (female x grant)
pf = PolynomialFeatures(degree=2, \
    interaction_only=True, include_bias=False).\
        fit(TDF[['Gender_F','Science-Grant_N']])
int_feat = pf.transform(TDF[['Gender_F', \
    'Science-Grant_N']])

print(int_feat)
# console output:
# [[0. 0. 0.]
#  [1. 1. 1.]
#  [0. 0. 0.]
#  [1. 1. 1.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [1. 1. 1.]
#  [0. 1. 0.]]

#%%
# convert the generated interaction feature array
# to a dataframe
fem_x_grant = pd.DataFrame(int_feat, \
    columns=['Gender_F','Science-Grant_N', 'FxGrant'])

#%%
# Calculate the mean value of female scientists
# who did not obtain the grant
print(fem_x_grant['FxGrant'].mean(0))
# console output:
# 0.375

#%% generate interaction features (male x grant)
pf = PolynomialFeatures(degree=2, \
    interaction_only=True, include_bias=False).\
        fit(TDF[['Gender_M','Science-Grant_N']])
int_feat = pf.transform(TDF[['Gender_M', \
    'Science-Grant_N']])

#%%
# convert the generated interaction feature array
# to a dataframe
male_x_grant = pd.DataFrame(int_feat, \
    columns=['Gender_M','Science-Grant_N', 'MxGrant'])

#%%
# Calculate the mean value of female scientists
# who did not obtain the grant
print(male_x_grant['MxGrant'].mean(0))
# console output:
# 0.25

