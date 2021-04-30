# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Feature variance

#%% load libraries
import numpy as np
import pandas as pd
from sklearn.feature_selection\
    import VarianceThreshold
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

#%% generate sample data
X = np.array([[0, 2, 0, 3], [0, 3, 4, 3], \
    [0, 5, 1, 2]])

#%% apply variance threshold
selector = VarianceThreshold(threshold=0.4)
Xs = selector.fit_transform(X)

#%% show the variances per feature
# (the ones above the threshold were chosen)
print(selector.variances_)
# console output:
# [0. , 1.55555556, 2.88888889, 0.22222222]

#%% load sample data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

#%% create and fit feature selector
selector = VarianceThreshold(threshold=0.6)
selector.fit_transform(X)

#%%
# show the variances per feature
# (the ones above the threshold were chosen)
pd.DataFrame({'features': iris.feature_names, \
    'variances': selector.variances_})
# console output:
# 	features	        variances
# 0	sepal length (cm)	0.681122
# 1	sepal width (cm)	0.188713
# 2	petal length (cm)	3.095503
# 3	petal width (cm)	0.577133

#%% create a barplot visual based on the variances
plt.bar(x=iris.feature_names, height=selector.variances_)
plt.ylabel('Feature Variance')
plt.title('Iris features variance comparison')
plt.show()

#%%
for selected_feature in selector.get_support(indices=True):
    print('* ' + feature_names[selected_feature])
# console output:
# * sepal length (cm)
# * petal length (cm)
