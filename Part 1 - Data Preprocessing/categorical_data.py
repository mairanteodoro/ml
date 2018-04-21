# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# matrix of features
X = dataset.iloc[:, :-1].values
# dependent variable array
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
# will replace nan by the mean of the neighbor values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# fit function to features (only Age & Salary)
imputer = imputer.fit(X[:, 1:3])
# apply transformation
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# simple way of encoding (countries will be replaced by an 'index' or 'rank')
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# this will create dummy variables to avoid indexing/ranking categorical variables
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



