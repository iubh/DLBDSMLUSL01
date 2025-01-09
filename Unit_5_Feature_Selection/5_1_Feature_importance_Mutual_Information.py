# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Feature Importance
# Mutual information

#%% import libraries
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
import pandas as pd
from sklearn.datasets import load_iris

#%% load sample data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

#%% create and fit feature selector
selector = SelectKBest(score_func=mutual_info_classif, k=2)
X_new = selector.fit_transform(X, y)

#%% print mutual information per feature
res = pd.DataFrame({'features': X.columns.values,
                    'Scores': selector.scores_})
print(res)

# console output:
#             features    Scores
# 0  sepal length (cm)  0.504630
# 1   sepal width (cm)  0.213946
# 2  petal length (cm)  0.995217
# 3   petal width (cm)  0.968754
