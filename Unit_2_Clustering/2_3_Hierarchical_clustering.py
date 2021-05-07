# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Hierarchical clustering

#%% import libraries
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.cluster import AgglomerativeClustering

#%% generate sample data
X1 = np.random.rand(5,2)
X2 = 2 + np.random.rand(5,2)
X3 = 3 + np.random.rand(5,2)
Z = np.concatenate((X1,X2,X3))

#%% calculate distances
linkage_array = ward(Z)

#%% visualize the dendrogram
# create a dendrogram
dendrogram(linkage_array)
ax = plt.gca()
bounds = ax.get_xbound()

# add the boundary for two/three clusters
ax.plot(bounds, [4, 4], '--', c='k')
ax.plot(bounds, [2, 2], '--', c='k')

# add an annotation to the marked boundary
ax.text(bounds[1], 4, ' two clusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 2, ' three clusters', va='center', fontdict={'size': 15})

# label the axes
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")

# show the plot
plt.show()

#%% execute agglomerative clusering with 3 clusters
agg = AgglomerativeClustering(n_clusters=3)

# predict clusters
labs = agg.fit_predict(Z)

print(labs)
# console output: [0 0 0 0 0 2 2 2 2 2 1 1 1 1 1]

