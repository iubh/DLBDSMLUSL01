# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Feature Importance
# Permutation feature importance

#%% load libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsClassifier

#%% load sample data
iris = load_iris()

#%% split the data into training and testing
X_train, X_test, y_train,y_test = train_test_split(iris.data, iris.target)

#%% create and fit a kNN model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

#%% assess feature importance by permutation 
feat_imp = permutation_importance(model,
                                  X_test, y_test, n_repeats=10,
                                  scoring='accuracy')

#%% display calculated feature importances
res = pd.DataFrame({'features': iris.feature_names,
                    'importances_mean': feat_imp['importances_mean'],
                    'importances_std': feat_imp['importances_std']})
print(res)

# console output:
#             features  importances_mean  importances_std
# 0  sepal length (cm)     -2.105263e-02         0.019693
# 1   sepal width (cm)     -3.330669e-17         0.020384
# 2  petal length (cm)      5.210526e-01         0.082213
# 3   petal width (cm)      8.684211e-02         0.026447