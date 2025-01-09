# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Text features
# Term Frequency-Inverse Document Frequency (TF-IDF)

#%% import libraries
from sklearn.feature_extraction.text import TfidfVectorizer

#%% generate sample data
corpus = ['Martin is not a bad person.',
          'Kevin is the brother of Martin.',
          'Kevin is a bad person.'] 



#%% create TF-IDF tokenizer with normalization
TFIDF= TfidfVectorizer(lowercase=False, norm = 'l2',
                       stop_words='english')

#%% fit tokenizer
TFIDFtext = TFIDF.fit_transform(corpus)

#%% print feature names
print(TFIDF.get_feature_names_out())
# console output:
# ['Kevin', 'Martin', 'bad', 'brother', 'person']

#%% print the values of each Word (second entry in
# parenthesis) in each document (first entry in parenthesis)
print(TFIDFtext) 
# console output:
# (0, 1)	0.5773502691896257
# (0, 2)	0.5773502691896257
# (0, 4)	0.5773502691896257
# (1, 1)	0.5178561161676974
# (1, 0)	0.5178561161676974
# (1, 3)	0.680918560398684
# (2, 2)	0.5773502691896257
# (2, 4)	0.5773502691896257
# (2, 0)	0.5773502691896257

# %%
