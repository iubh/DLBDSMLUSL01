# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Sample generation

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture

#%% generate sample data
X1 = 4 + np.random.rand(50,2)
X2 = 5 + np.random.rand(50,2)
X3 = 6 + np.random.rand(50,2)
Z = np.concatenate((X1,X2,X3))

#%% Build and fit a GMM model
gmm = mixture.GaussianMixture(n_components=3)
gmm.fit(Z)

#%% generate new samples
newdata = gmm.sample(150)

#%% extract the feature values, i.e. coordinates
vals = newdata[0]

#%% extract the labels
labs = newdata[1]

#%% plot the generated samples 
plt.scatter(x=vals[:,0], y=vals[:,1], c=labs)
plt.show()
