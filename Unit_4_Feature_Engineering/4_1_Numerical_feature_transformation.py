# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Numerical feature transformation

#%% import libraries
from numpy import exp
from numpy.random import randn
from feature_engine import transformation as vt
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import matplotlib.pyplot as plt

#%% generate sample data from a normal distribution 
dat = randn(999)

#%% add a skew to the generated data points
dat_skew = exp(dat)
plt.hist(dat_skew, bins = 25)
plt.show()

#%% convert the data into a dataframe
dat_skew = dat_skew.reshape((len(dat), 1))
dat_skew = pd.DataFrame(dat_skew, columns = ['Value'])

#%% generate and fit log transformer
lgt = vt.LogTransformer(variables= ['Value'])
lgt.fit(dat_skew)

#%% apply log transformation 
dat_lg = lgt.transform(dat_skew)

#%% plot the distribution of the transformed data
plt.hist(dat_lg['Value'], bins=25)
plt.show()

#%% generate and fit quantile transformer
qt = QuantileTransformer(output_distribution='normal')
qt.fit(dat_skew[['Value']])

#%% apply quantile transformation 
dat_q = qt.transform(dat_skew[['Value']])

#%% plot the distribution of the transformed data
plt.hist(dat_q, bins=25)
plt.show()

# %%
