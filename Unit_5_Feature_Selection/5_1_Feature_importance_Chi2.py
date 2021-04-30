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
from sklearn.preprocessing import LabelEncoder

#%% load sample data
# log into kaggle
api = KaggleApi()
api.authenticate()

# download the data
kaggle_user = 'sonalidasgupta95'
kaggle_project = 'churn-prediction-of-bank-customers'
api.dataset_download_files(kaggle_user + '/' + kaggle_project)

# unzip the data
zip = zipfile.ZipFile(kaggle_project + '.zip').\
    extractall()

# load the data
churn_df = pd.read_csv('Churn_Modelling.csv')

#%% prepare the data
y = churn_df['Exited']
X = churn_df[['Gender', 'HasCrCard', 'IsActiveMember']]

#%% apply label encoder
X['Gender'] = LabelEncoder().fit_transform(X['Gender'])

#%% create and fit feature selector
selector = SelectKBest(chi2, k=2)
selector.fit(X,y)

#%% apply feature selector to the data
X_new = selector.transform(X)

#%% print Chi²-statistics- and p-values per feature
pd.DataFrame({'features': X.columns.values, \
    'Scores': selector.scores_, \
    'p-values': selector.pvalues_})

# console output:
# 	features	    Scores	    p-values
# 0	Gender	        51.539926	7.015575e-13
# 1	HasCrCard	    0.150041	6.984962e-01
# 2	IsActiveMember	118.199414	1.568036e-27

#%% delete data from local file system
import os
def rm_file(f):
    if os.path.exists(f):
        os.remove(f)
rm_file(kaggle_project + '.zip')
rm_file('Churn_Modelling.csv')