# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Silhouette score

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score

#%% generate sample data
X= np.random.rand(50,2)
Y= 2 + np.random.rand(50,2)
Z= np.concatenate((X,Y))

#%% conduct a k-Means clustering
model = KMeans(n_clusters=2, random_state=0).fit(Z)

#%% extract labels, i.e. cluster associations
lab=model.labels_

#%% calculate the overall Silhouette score
S = silhouette_score(Z, lab)
print(S)
# console output: 0.8123455046726186

#%% generate, fit and show a Silhouette visualizer
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(Z)        
visualizer.show()

# %%
