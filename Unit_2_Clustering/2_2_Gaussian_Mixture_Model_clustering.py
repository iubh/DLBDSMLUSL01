# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Gaussian Mixture Model clustering

#%% import libraries
from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt

#%% generate sample data
X1 = 4 + np.random.rand(50,2)
X2 = 5 + np.random.rand(50,2)
X3 = 6 + np.random.rand(50,2)
Z = np.concatenate((X1,X2,X3))

#%% plot the sample data
plt.scatter(Z[:, 0], Z[:, 1], marker='+')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#%% specify Gaussian Mixture Model
gmm = mixture.GaussianMixture(n_components=3)

#%% fit the model
gmm.fit(Z)

#%%
# extract the clusters predictions according to
# the highest probability
labels = gmm.predict(Z)

#%% show the predicted labels
print(labels)
# console output:
# [1 1 1 1 1 1 1 1 1 1 1 1...
#  ...2 2 2 2 2 2 2 2 2 2 ...
#  ...0 0 0 0 0 0 0 0 0 0 ]

#%% extract the probabilities to belong to a cluster
probs = gmm.predict_proba(Z)
print(probs)
# console output:
# [[1.85387989e-24 9.99998029e-01 1.97120198e-06]
#  [6.54360391e-28 9.99999964e-01 3.56085199e-08]
#  ...
#  [1.00000000e+00 2.36479665e-39 2.81008490e-10]
#  [9.99470972e-01 1.24217497e-24 5.29027653e-04]]

#%% show results visually
plt.scatter(x=Z[:,0], y=Z[:,1], c=labels, cmap='viridis')
plt.show()

# %%
