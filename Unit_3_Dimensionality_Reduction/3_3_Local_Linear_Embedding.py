# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Local Linear Embedding (LLE)

#%% import the required libraries
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import manifold, datasets

#%% generate 1000 data points of the Swiss roll dataset 
n_pts = 1000
X, color = datasets.make_s_curve(n_pts, random_state=0)

#%% display the data points
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs=X[:,0], ys=X[:,1], zs=X[:,2], c=color)
plt.show()    

#%%
# apply LLE to the generated Swiss roll dataset
# to project it into a 2-dimensional feature space
embedding = LocallyLinearEmbedding(n_neighbors=12, \
    n_components=2)
X_2d = embedding.fit_transform(X)

#%%
# display the data points in the
# reduced feature space
plt.scatter(X_2d[:,0], X_2d[:,1], c=color)
plt.show()    

# %%
