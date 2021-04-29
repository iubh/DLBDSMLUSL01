# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

'''
In this example, we will see how we can use Python
to conduct k-Means clustering on a simple 2-dimensional
data set.
'''

#%% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#%% generate random variables
X= np.random.rand(50,2)
Y= 2 + np.random.rand(50,2)

#%% gather random variable in a dataframe
Z= np.concatenate((X,Y))
df = pd.DataFrame(Z, columns=['xpt', 'ypt'])

#%% glimpse at the data
df.head()
# console output:
# 	xpt     	ypt
# 0	0.843469	0.719464
# 1	0.283066	0.002213
# 2	0.327358	0.211112
# 3	0.087454	0.286058
# 4	0.606084	0.568120

#%% plot the data points
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(x=df['xpt'], y=df['ypt'])
plt.show()

#%% clustering
kmeans = KMeans(n_clusters=2, random_state=0).fit(Z)

#%% extract centroids of clusters into a dataframe
centers = kmeans.cluster_centers_
centroids = pd.DataFrame(centers,columns=['xpt','ypt'])

#%% extract cluster labels
lab = kmeans.labels_

#%% add cluster information to the dataframe
df['lab']=lab

#%% glimpse at the 'labeled' data
df.head()
# console output:
# 	xpt     	ypt     	lab
# 0	0.684751	0.846389	1
# 1	0.572355	0.144059	1
# 2	0.752941	0.768759	1
# 3	0.722927	0.790472	1
# 4	0.105875	0.761570	1

#%% plot the 'labeled' data with centroids
# create a figure and axes
fig, ax = plt.subplots(figsize=(6,4))

# add data points
ax.scatter(x=df['xpt'], y=df['ypt'], c=df['lab'])

# add cluster centroids
ax.scatter(centroids['xpt'], centroids['ypt'])
plt.show()

#%% calculate the maximum radius around earch cluster
radii = [cdist(df[lab == i].iloc[:,[0,1]], [center]).\
    max() \
        for i, center in enumerate(centers)]

#%% glimpse at the found radii
radii
# console output:
# [0.7096052458280815, 0.6578703939400616]

#%% display clusters’ zones
# create a figure and axes
fig, ax = plt.subplots(figsize=(6,4))

# add data points
ax.scatter(x=df['xpt'], y=df['ypt'], c=df['lab'])

# add cluster centroids
ax.scatter(centroids['xpt'], centroids['ypt'])

# set the axis scale on both axes equally
ax.axis('equal')

# draw a circle around each cluster centroid
for c, r in zip(centers, radii):
    ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', \
                        lw=3, alpha=0.5))

# show the plot                        
plt.show()
