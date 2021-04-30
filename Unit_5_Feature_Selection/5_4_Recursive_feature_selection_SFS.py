# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Recursive feature selection
# Sequential Backward Feature Selection (SBS)

#%% import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from mlxtend.feature_selection \
    import SequentialFeatureSelector as SBS
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#%% load sample data
iris = load_iris()
x = pd.DataFrame(iris.data, \
    columns=iris.feature_names)

#%% create a logistic regression object
lr = LogisticRegression()

#%% create an SBS object
sbs = SBS(estimator=lr,
          k_features=(1, 3),  
          forward=False,       
          scoring='accuracy', 
          cv=5)      
      
#%% fit the model
sbs = sbs.fit(x, iris.target)

#%% show the selected features
sbs.k_feature_names_
# console output:
# ('sepal length (cm)', 'petal length (cm)', 
# 'petal width (cm)')

#%% show a full report on the feature selection
sbs_results = pd.DataFrame(sbs.get_metric_dict()).\
    T. \
    sort_values(by='avg_score', ascending=False)

#%% show feature importance visually
# create figure and axes
fig, ax = plt.subplots()

# plot bars
y_pos = np.arange(len(sbs_results))
ax.barh(y_pos, sbs_results['avg_score'], \
    xerr=sbs_results['std_err'])

# set axis ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(sbs_results['feature_names'])
ax.set_xlabel('Accuracy')

# limit range to overimpose differences
plt.xlim([0.95, 0.98])

# show the plot
plt.show()

# %%
