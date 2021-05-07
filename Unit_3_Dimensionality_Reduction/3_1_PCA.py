# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Principal component analysis

#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

#%% load the sample data
iris = datasets.load_iris()
X = iris.data
y = iris.target

#%% standardize the data scales
X_std = StandardScaler().fit_transform(X)

#%% calculate the variance-covariance matrix
cov_X_std = np.cov(X_std.T)

print(cov_X_std)
# console output:
# [[ 1.00671141 -0.11835884  0.87760447  0.82343066]
#  [-0.11835884  1.00671141 -0.43131554 -0.36858315]
#  [ 0.87760447 -0.43131554  1.00671141  0.96932762]
#  [ 0.82343066 -0.36858315  0.96932762  1.00671141]]

#%% compute Eigenvectors and Eigenvalues
eig_vals, eig_vecs = np.linalg.eig(cov_X_std)

print(eig_vals)
# console output: [2.93808505 0.9201649  0.14774182 0.02085386]

print(eig_vecs)
# console output:
# [[ 0.52106591 -0.37741762 -0.71956635  0.26128628]
#  [-0.26934744 -0.92329566  0.24438178 -0.12350962]
#  [ 0.5804131  -0.02449161  0.14212637 -0.80144925]
#  [ 0.56485654 -0.06694199  0.63427274  0.52359713]]


#%% calculating the total sum of Eigenvalues
tot = sum(eig_vals) 
print(tot)
# console output: 4.026845637583896

#%% sorting Eigenvalues in decreasing order
exp_var = sorted(eig_vals, reverse=True)

#%% present values as percentages
exp_var = exp_var/tot * 100

#%% print the explained variance
print("Explained variance per PC:", exp_var)
# console output: 
# Explained variance per PC: [72.96244541 22.85076179 
# 3.66892189  0.51787091]

#%% Computing and print the explained cumulative variance 
cum_exp_var = np.cumsum(exp_var)
print("Cumulative Explained Variance:",cum_exp_var)
# console output:
# Cumulative Explained Variance: [ 72.96244541 95.8132072
# 99.48212909 100. ]

#%%
# display the proportion of variance
# which is explained by the first two PC
print("Explained variance by PC1 and PC2:", \
    sum(exp_var[0:2]))
# console output:
# Explained variance by PC1 and PC2: 95.81320720000163

#%% construct a projection matrix
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) 
             for i in range(len(eig_vals))]
PR = np.hstack((eig_pairs[0][1].reshape(4,1), 
                eig_pairs[1][1].reshape(4,1))) 

#%%
# project the original data to the feature
# which is reduced in dimensions
Y = X_std.dot(PR) 
YL = np.append(Y, y.reshape(150,1), axis=1)

#%% plot the projected data
plt.scatter(x=YL[:,0], y=YL[:,1], c=YL[:,2])
plt.show()

#%% doing it the easy way
pca = PCA().fit(X_std)

#%% extract the explaind variance ratios
var_exp = pca.explained_variance_ratio_
print(var_exp)
# console output:
# [0.72962445 0.22850762 0.03668922 0.00517871]

#%% calculate the explained cumulative variance
cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)
# console output:
# [0.72962445 0.95813207 0.99482129 1.]

#%% extract the Eigenvectors
eig_vecs = pca.components_

#%%
# use PCA to project the data to a two-dimensional
# feature space
Y = PCA(n_components=2).fit(X_std).transform(X_std)

#%% plot the projected data
plt.scatter(x=Y[:,0], y=Y[:,1])
plt.show()