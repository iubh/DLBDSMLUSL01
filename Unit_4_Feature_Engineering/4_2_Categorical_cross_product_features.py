# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Categorical cross-product features

#%% import libraries
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

#%% generate sample data
Table = {
    'Treatment': ['P','M','P','M','M','P', 'M'],
    'Sports': ['high', 'low', 'high', 'low', 'high', 'low', 'low']
    }
TDF = pd.DataFrame(data=Table)

#%% one-hot encoding
TDF = pd.get_dummies(TDF)

#%% generate interaction features (medication x active)
pf = PolynomialFeatures(degree=2, interaction_only=True,
                        include_bias=False).\
        fit(TDF[['Treatment_M','Sports_high']])
int_feat = pf.transform(TDF[['Treatment_M', 'Sports_high']])

print(int_feat)
# console output:
# [[0. 1. 0.]
#  [1. 0. 0.]
#  [0. 1. 0.]
#  [1. 0. 0.]
#  [1. 1. 1.]
#  [0. 0. 0.]
#  [1. 0. 0.]]

#%% convert the generated interaction feature array
# to a dataframe
int_feat = pd.DataFrame(int_feat,
    columns=['Treatment_M','Sports_high', 'med_x_sports'])

#%% print results
print(int_feat)
# console output:
#    Treatment_M  Sports_high  med_sports
# 0          0.0          1.0         0.0
# 1          1.0          0.0         0.0
# 2          0.0          1.0         0.0
# 3          1.0          0.0         0.0
# 4          1.0          1.0         1.0
# 5          0.0          0.0         0.0
# 6          1.0          0.0         0.0
# %%
