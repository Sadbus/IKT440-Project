import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

data = pd.read_csv('connect-4.data')
data_in_numpy = data.values

x = data.iloc[:, :-1].values
y = data['class']

le = preprocessing.LabelEncoder()
for i in range(len(x)):
    x[i] = le.fit_transform(x[i])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
