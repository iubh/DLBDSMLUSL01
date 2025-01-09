# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Local Linear Embedding (LLE)

#%% import the required libraries
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import manifold, datasets

#%% generate 1000 data points of the Swiss roll dataset 
X, color = datasets.make_s_curve(1000, random_state=0)

#%% display the data points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs=X[:,0], ys=X[:,1], zs=X[:,2], c=color)
plt.show()    

#%% apply LLE to the generated Swiss roll dataset
# to project it into a 2-dimensional feature space
embedding = LocallyLinearEmbedding(n_neighbors=12, n_components=2)
X_2d = embedding.fit_transform(X)

#%% display the data points in the
# reduced feature space
plt.scatter(X_2d[:,0], X_2d[:,1], c=color)
plt.show()    

# %% use the reconstruction error to evaluate the number of
# dimensions to be used for the reduced feature space
for k in range(1,4):
    embedding = LocallyLinearEmbedding(n_neighbors=12, n_components=k)
    X_2d = embedding.fit_transform(X)
    recon_error = embedding.reconstruction_error_
    print('Reconstruction error with ' + str(k) + ' dimensions: '
          + str(recon_error))

# console output:
# Reconstruction error with 1 dimensions: 1.0919767883014307e-09
# Reconstruction error with 2 dimensions: 2.4743682803511204e-07
# Reconstruction error with 3 dimensions: 5.165242320513663e-07