# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Multi-Dimensional Scaling - Swiss roll

#%% import libraries
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
import matplotlib.pyplot as plt

#%% generate swiss roll data
n_pts = 1000
X, color = datasets.make_s_curve(n_pts, random_state=0)

#%% plotting the data in 3D
ax = plt.axes(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=color)
plt.show()

#%%
# project the data into a 2-dimensional feature
# space using MDS
n_components = 2
mds = MDS(2,random_state=0)
X_2d = mds.fit_transform(X)

# display the projected data in the 2-dimensional
# feature space
plt.scatter(X_2d[:,0], X_2d[:,1], c=color)
plt.show()
