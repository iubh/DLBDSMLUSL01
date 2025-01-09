# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Feature Importance
# Chi²

#%% import libraries
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

#%% log into kaggle
api = KaggleApi()
api.authenticate()

#%% download the data
kaggle_user = 'sonalidasgupta95'
kaggle_project = 'churn-prediction-of-bank-customers'
api.dataset_download_files(kaggle_user + '/' + kaggle_project)

# unzip the data
zip = zipfile.ZipFile(kaggle_project + '.zip').\
    extractall()

#%% load the data
churn_df = pd.read_csv('Churn_Modelling.csv')

#%% prepare the data
y = churn_df['Exited']
X = churn_df[['Gender', 'HasCrCard', 'IsActiveMember']]

#%% apply one-hot-encoding
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)

#%% create, fit, and apply the feature selector
selector = SelectKBest(chi2, k=2)
selector.fit(X,y)
X_new = selector.transform(X)

#%% print Chi²-statistics- and p-values per feature
res = pd.DataFrame({'features': X.columns.values,
                    'Scores': selector.scores_,
                    'p-values': selector.pvalues_})
print(res)

# console output:
#          features      Scores      p-values
# 0       HasCrCard    0.150041  6.984962e-01
# 1  IsActiveMember  118.199414  1.568036e-27
# 2     Gender_Male   51.539926  7.015575e-13
