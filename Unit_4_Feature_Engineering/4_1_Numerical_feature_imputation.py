# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Numerical feature imputation

#%% load libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge

#%% generate sample data
Table = {\
    'Customer-ID': [1, 2, 3, 4, 5], \
    'Gender': ['M','F','M','F','F'], \
    'Work-type': [1, 2, 2, 3, 3], \
    'Client-satisfaction':[3, 0, 4, 3, 5], \
    'Number-occupants': [2, 4, 2, 1, 2],
    'Consumption':[70, 140, 65, np.NaN, 65]}
TDF = pd.DataFrame(data=Table)

#%% simple imputation of missing values
imput = SimpleImputer(missing_values = np.nan, \
    strategy = 'mean')
imput = imput.fit(TDF[['Consumption']])
imput = imput.transform(TDF[['Consumption']])
TDF['Consumption']= imput

#%% generate sample data
table = {'Var-1': [10, np.NaN, 2, 1, 5],\
         'Var-2': [2, 1, 0.4, 0.2, np.NaN]}
TDF = pd.DataFrame(data=table)
TDF
# console output:
# 	Var-1	Var-2
# 0	10.0	2.0
# 1	NaN	    1.0
# 2	2.0	    0.4
# 3	1.0	    0.2
# 4	5.0	    NaN

#%% apply regression imputation using ‘Bayesian Ridge’
imputbr = IterativeImputer(BayesianRidge())
TDF = pd.DataFrame(imputbr.fit_transform(TDF))
TDF
# console output:
# 	0	        1
# 0	10.000000	2.000000
# 1	5.000531	1.000000
# 2	2.000000	0.400000
# 3	1.000000	0.200000
# 4	5.000000	0.999973

#%%
# apply regression imputation using
# ‘Extra Trees Regressor’
TDF = pd.DataFrame(data=table)
imputetr = IterativeImputer(ExtraTreesRegressor())
TDF = pd.DataFrame(imputetr.fit_transform(TDF))
TDF
# console output:
# 	0	    1
# 0	10.00	2.000
# 1	5.35	1.000
# 2	2.00	0.400
# 3	1.00	0.200
# 4	5.00	0.898
