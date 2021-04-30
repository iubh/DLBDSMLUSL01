# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Text features
# Bag-of-Words (BoW) and Bag-of-2-Grams (Bo2G)

#%% import libraries
from sklearn.feature_extraction.text \
    import CountVectorizer

#%% generate sample data
corpus = ['Martin is not a bad person.',
          'Kevin, is the brother of Martin.',
          'Kevin is a bad person.'] 

#%% create bag-of-words tokenizer
vectorizer = CountVectorizer(lowercase = False,  
                             stop_words='english')

#%% fit tokenizer
BoW = vectorizer.fit_transform(corpus)

#%% print feature names
print(vectorizer.get_feature_names())
# console output:
# ['Kevin', 'Martin', 'bad', 'brother', 'person']

#%%
# print the number of occurrences of each
# feature in each text element
print(BoW.toarray())

# console output:
# [[0 1 1 0 1]
#  [1 1 0 1 0]
#  [1 0 1 0 1]]

#%% create bag Bag-of-n-Grams tokenizer
vectorizer2 = CountVectorizer(analyzer='word', \
    ngram_range=(2, 2), lowercase=False,
    stop_words='english')

#%% fit the vectorizer
Bo2G = vectorizer2.fit_transform(corpus)

#%% print feature names
print(vectorizer2.get_feature_names())
# console output:
# ['Kevin bad', 'Kevin brother', 'Martin bad', 
#  'bad person', 'brother Martin']

#%%
# print the number of occurrences of each
# feature in each text element
print(Bo2G.toarray())
# console output (extract):
# [[0 0 1 1 0]
#  [0 1 0 0 1]
#  [1 0 0 1 0]]
