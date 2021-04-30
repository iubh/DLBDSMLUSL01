# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Correlation matrix

#%% load libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

#%% load sample data
iris = load_iris()
X = pd.DataFrame(iris.data, \
    columns=iris.feature_names)
y = pd.DataFrame(iris.target, \
    columns=['Response_variable'])
XY = pd.concat([X, y], axis=1)


#%% calcualte the correlation matrix
cor_mat = XY.corr(method='pearson')
cor_mat

#%% correlation matrix as heatmap
ax = sns.heatmap(cor_mat, vmin=-1, vmax=1, \
    annot=True, fmt="f")
plt.show()

## Spearman correlation

#%% generate sampe data
dat = pd.DataFrame({'x': np.arange(0,10), \
    'y': np.exp(-np.arange(0,10))})

#%%
# compute Pearson’s correlation and
# display the correlation matrix
XY.corr(method='pearson')

# console output:
#   x	        y
# x	1.00000	    -0.71687
# y	-0.71687	1.00000

#%% test for normal distribution of the data
stats.normaltest(XY['x'])
# console output:
# NormaltestResult(statistic=2.02697581498966, 
# pvalue=0.362950830342156)

stats.normaltest(XY['y'])
# console output:
# NormaltestResult(statistic=19.779358749097575, 
# pvalue=5.0695197559354735e-05)

#%%
# compute Spearman’s rank correlation and
# display the correlation matrix
XY.corr(method='spearman')

# console output:
# 	x	    y
# x	1.0	    -1.0
# y	-1.0	1.0

