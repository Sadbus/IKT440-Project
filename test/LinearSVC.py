import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from time import time

data = pd.read_csv('connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

enc = preprocessing.OrdinalEncoder()
x_transformed = enc.fit_transform(x)

le = preprocessing.LabelEncoder()
y_transformed = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y_transformed, test_size=0.20)

lin_clf = svm.LinearSVC()

start_train = time()
lin_clf.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = lin_clf.predict(x_test)
stop_test = time()

print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (accuracy_score(y_test, y_pred)*100, stop_train - start_train, stop_test - start_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print('\n Classification Report:\n', classification_report(y_pred, y_test, labels=np.unique(y_pred)))
