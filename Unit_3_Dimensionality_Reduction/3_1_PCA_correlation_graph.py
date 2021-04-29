# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# PCA correlation graph

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_pca_correlation_graph

#%% load and standardize data
iris = datasets.load_iris()
X = iris.data
X_std = StandardScaler().fit_transform(X)

#%% specify feature names
feature_names = ['sepal length','sepal width',  
                 'petal length','petal width']

#%%
# calculate the correlation matrix and
# create a correlation graph
fig, cor_mat = plot_pca_correlation_graph(X_std, \
    feature_names, dimensions=(1, 2), \
    figure_axis_size=10)

#%%
# show the numbers of the correlation
# matrix for the 4 features 
print(cor_mat)
# console output:
#                  Dim 1     Dim 2
# sepal length -0.890169 -0.360830
# sepal width   0.460143 -0.882716
# petal length -0.991555 -0.023415
# petal width  -0.964979 -0.064000

# %%
