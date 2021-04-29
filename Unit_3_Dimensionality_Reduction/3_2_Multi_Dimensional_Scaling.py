# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Multi-Dimensional Scaling (MDS)

#%% import libraries
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.preprocessing import MinMaxScaler

#%% load the sample data set
iris = datasets.load_iris()
X = iris.data

#%% normalize the data
X_scaled = MinMaxScaler().fit_transform(X)

#%% conduct MDS on the data
mds = MDS(2,random_state=0)
X_2d = mds.fit_transform(X_scaled)

#%%
# Plot the projected Iris data points into the reduced
# feature space by MDS 
plt.scatter(x=X_2d[:,0], y=X_2d[:,1], c=iris.target)
plt.show()
