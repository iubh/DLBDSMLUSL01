# IU - International University of Applied Science
# Machine Learning - Unsupervised Machine Learning
# Course Code: DLBDSMLUSL01

# Feature Extraction from DL

#%% import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

#%% load the dataset
iris = load_iris()

#%% preprocess the data
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = iris.target.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

#%% define a simple neural network model
input_layer = Input(shape=(4,))
hidden_layer = Dense(4, activation='relu')(input_layer)
output_layer = Dense(3, activation='softmax')(hidden_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

#%% train the model
model.fit(X_scaled, y_encoded, epochs=50, batch_size=8,
          validation_split=0.2, verbose=0)

#%% define a feature extraction model
feature_model = Model(inputs=model.input,
                      outputs=model.layers[-2].output)

#%% generate features from the trained model
extracted_features = feature_model.predict(X_scaled)
print(pd.DataFrame(extracted_features).describe())

# console output:
#                 0           1           2           3
# count  150.000000  150.000000  150.000000  150.000000
# mean     0.866842    0.732252    0.949339    0.473582
# std      0.826123    0.915698    1.208514    0.453544
# min      0.000000    0.000000    0.000000    0.000000
# 25%      0.000000    0.000000    0.000000    0.035525
# 50%      0.800305    0.224484    0.019181    0.357864
# 75%      1.526661    1.262004    2.219630    0.809972
# max      2.988739    3.351760    3.687942    1.858005
# %%
