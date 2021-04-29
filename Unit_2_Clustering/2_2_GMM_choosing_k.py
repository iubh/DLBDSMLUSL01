# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Gaussian Mixture Model clustering
# Choosing the number of clusers, k

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.metrics import silhouette_score

#%% generate sample data
X1 = 4 + np.random.rand(50,2)
X2 = 5 + np.random.rand(50,2)
X3 = 6 + np.random.rand(50,2)
Z = np.concatenate((X1,X2,X3))

#%%
# calculate the Silhouette score and BIC
# for the number of clusters, k = 2 to 6
S = []
bic = []
n_cluster_range = [2, 3, 4, 5, 6]
for n_cluster in n_cluster_range:
    gmm = mixture.GaussianMixture(n_components=n_cluster)
    gmm.fit(Z)
    lab = gmm.predict(Z)
    S.append(silhouette_score(Z, lab))
    bic.append(gmm.bic(Z))

#%% show the resuls visuallay
# figure with two plots
fig, (ax1, ax2) = plt.subplots(1, 2)

# first plot: Silhouette score
ax1.plot(n_cluster_range, S)
ax1.set_title('Silhouette Score')
ax1.set(xlabel='Number of clusters', \
    ylabel='Silhouette Score')

# second plot: BIC
ax2.plot(n_cluster_range, bic)
ax2.set_title('BIC')
ax2.set(xlabel='Number of clusters', \
    ylabel='BIC')

plt.show()
