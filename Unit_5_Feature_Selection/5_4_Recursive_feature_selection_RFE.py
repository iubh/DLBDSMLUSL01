# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Recursive feature selection
# Sequential Forward Feature Selection (SFS)

#%% import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from mlxtend.feature_selection \
    import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#%% load sample data
iris = load_iris()
x = pd.DataFrame(iris.data, \
    columns=iris.feature_names)

#%% create a logistic regression object
lr = LogisticRegression()

#%% create an SFS object
sfs = SFS(estimator=lr,
          k_features=(1, 3),  
          forward=True,       
          scoring='accuracy', 
          cv=5)      


#%% fit the model
sfs = sfs.fit(x, iris.target)

#%% show the selected features
sfs.k_feature_names_
# console output:
# ('sepal length (cm)', 'petal length (cm)', 
# 'petal width (cm)')

#%% show a full report on the feature selection
sfs_results = pd.DataFrame(sfs.get_metric_dict()).\
    T. \
    sort_values(by='avg_score', ascending=False)

#%% show feature importance visually
# create figure and axes
fig, ax = plt.subplots()

# plot bars
y_pos = np.arange(len(sfs_results))
ax.barh(y_pos, sfs_results['avg_score'], \
    xerr=sfs_results['std_err'])

# set axis ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(sfs_results['feature_names'])
ax.set_xlabel('Accuracy')

# limit range to overimpose differences
plt.xlim([0.95, 0.98])

# show the plot
plt.show()

# %%
