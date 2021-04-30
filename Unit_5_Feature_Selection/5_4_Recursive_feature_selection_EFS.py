# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Recursive feature selection
# Exclusive Feature Selection (EFS)

#%% import libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from mlxtend.feature_selection \
    import ExhaustiveFeatureSelector as EFS
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

#%% load sample data
iris = load_iris()
x = pd.DataFrame(iris.data, \
    columns=iris.feature_names)

#%% create a logistic regression object
lr = LogisticRegression()

#%% create an EFS object
efs = EFS(estimator=lr,        
          min_features=1,      
          max_features=3,      
          scoring='accuracy',  
          cv=5)

#%% fit the model
efs = efs.fit(x, iris.target)

#%% show the selected features
efs.best_feature_names_
# console output:
# ('sepal length (cm)', 'petal length (cm)', 
# 'petal width (cm)')

#%% show a full report on the feature selection
efs_results = pd.DataFrame(efs.get_metric_dict()).\
    T. \
    sort_values(by='avg_score', ascending=False)

#%% show feature importance visually
# create figure and axes
fig, ax = plt.subplots()

# plot bars
y_pos = np.arange(len(efs_results))
ax.barh(y_pos, efs_results['avg_score'], \
    xerr=efs_results['std_err'])

# set axis ticks and labels
ax.set_yticks(y_pos)
ax.set_yticklabels(efs_results['feature_names'])
ax.set_xlabel('Accuracy')

# show the plot
plt.show()
