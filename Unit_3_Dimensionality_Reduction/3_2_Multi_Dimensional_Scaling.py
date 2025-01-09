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
mds = MDS(2, random_state=0)
X_2d = mds.fit_transform(X_scaled)

#%% Plot the projected Iris data points in the reduced
# feature space by MDS 
plt.scatter(x=X_2d[:,0], y=X_2d[:,1])
plt.show()

# %% extract the stress for this model
stress_2d = mds.stress_

#%% reduce to different numbers of dimensions
# and extract the stress values
for k in range(1,5):
    mds = MDS(k, random_state=0)
    mds_fit = mds.fit_transform(X_scaled)
    print('Stress with ' + str(k) + ' dimensions: '
          + str(mds.stress_))
    
# console output:
# Stress with 1 dimensions: 1733.719555432615
# Stress with 2 dimensions: 21.633324137180807
# Stress with 3 dimensions: 1.8234756184195682
# Stress with 4 dimensions: 1.5208397803450002