# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Feature Importance
# ANOVA

#%% import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest   
from sklearn.feature_selection import f_classif

#%% load sample data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

#%% run ANOVA to select 2 features
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

#%% print F- and p-values per feature
res = pd.DataFrame({'features': iris.feature_names,
                    'Scores': selector.scores_,
                    'p-values': selector.pvalues_})
print(res)

# console output:
# 	features	        Scores	    p-values
# 0	sepal length (cm)	119.264502	1.669669e-31
# 1	sepal width (cm)	49.160040	4.492017e-17
# 2	petal length (cm)	1180.161182	2.856777e-91
# 3	petal width (cm)	960.007147	4.169446e-85

# %%
