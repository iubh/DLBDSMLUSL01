# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

'''
In this example, we will see how we can use Python
to extract added value within a data set based on dates.
'''

#%% load the required packages
import pandas as pd
from scipy.stats import spearmanr
from datetime import date
import seaborn as sns

#%% load the data
dat = pd.read_csv('./data/data_visitors.csv', delimiter=' ')

#%% glimpse at the data
dat

#%% correlate the date and the visitors
spearmanr(dat['date'], dat['visitors'])

#%% define the date as such
dat['date'] = pd.to_datetime(dat['date'])

#%% extract the weekdays from the date
dat['weekday'] = dat['date'].apply(lambda date: date.weekday())

#%% glimpse at the data
dat

#%% extract the weekends
dat['weekend'] = dat['weekday'].\
    apply(lambda weekday: 0 if (weekday < 5) else 1)

#%% glimpse at the data
dat

#%% correlate the weekends and the visitors
spearmanr(dat['weekend'], dat['visitors'])

#%% visually glimpse at the data
ax = sns.barplot(x="weekend", y="visitors", data=dat)
# %%
