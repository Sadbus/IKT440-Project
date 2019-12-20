from time import time

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('../connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

enc = preprocessing.OrdinalEncoder()
x_transformed = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.20)

clf = GaussianNB()

start_train = time()
clf.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = clf.predict(x_test)
stop_test = time()

print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (
accuracy_score(y_pred, y_test) * 100, stop_train - start_train, stop_test - start_test))
print("\n Confusion Matrix:\n", confusion_matrix(y_pred, y_test))
print('\n Classification Report:\n', classification_report(y_pred, y_test))
