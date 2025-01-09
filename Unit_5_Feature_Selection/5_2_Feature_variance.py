# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Feature variance

#%% load libraries
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris

#%% load sample data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

#%% create and fit feature selector
selector = VarianceThreshold(threshold=0.6)
selector.fit_transform(X)

#%%
# show the variances per feature
# (the ones above the threshold were chosen)
res = pd.DataFrame({'features': iris.feature_names,
                    'variances': selector.variances_})
print(res)

# console output:
# 	features	        variances
# 0	sepal length (cm)	0.681122
# 1	sepal width (cm)	0.188713
# 2	petal length (cm)	3.095503
# 3	petal width (cm)	0.577133
