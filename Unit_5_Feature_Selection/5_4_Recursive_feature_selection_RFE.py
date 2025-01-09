# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Recursive feature selection
# Recursive Feature Elimination

#%% import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

#%% load sample data
iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)

#%% create a logistic regression object
lr = LogisticRegression()

#%% create and fit logistic regressor with RFE
rfe = RFE(estimator=lr, n_features_to_select=3)
rfe = rfe.fit(x, iris.target)

#%% show which feature were selected
rfe_res = pd.DataFrame({'features': iris.feature_names,
                        'Selected features': rfe.support_,
                        'Feature ranks': rfe.ranking_})
print(rfe_res)

# console output:
# 	features	        Selected features	Feature ranks
# 0	sepal length (cm)	False	            2
# 1	sepal width (cm)	True	            1
# 2	petal length (cm)	True	            1
# 3	petal width (cm)	True	            1

#%% REF with cross-validation
rfecv = RFECV(estimator=lr, step=1, cv=5,
              scoring='accuracy', min_features_to_select= 3)
rfecv = rfecv.fit(x, iris.target)


#%% show the CV results
print(rfecv.cv_results_)

# console output:
# {'mean_test_score': array([0.96666667, 0.97333333]),
# 'std_test_score': array([0.02108185, 0.02494438]),
# 'split0_test_score': array([0.96666667, 0.96666667]),
# 'split1_test_score': array([0.96666667, 1.        ]),
# 'split2_test_score': array([0.93333333, 0.93333333]),
# 'split3_test_score': array([0.96666667, 0.96666667]),
# 'split4_test_score': array([1., 1.]),
# 'n_features': array([3, 4])}

# %%
