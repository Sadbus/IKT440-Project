from time import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, \
    accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('../connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

enc = preprocessing.OneHotEncoder()
x_transformed = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.20, random_state=42)


clf = KNeighborsClassifier(
    metric='manhattan',
    n_neighbors=28,
    weights='distance',
    n_jobs=-1)

start_train = time()
clf.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = clf.predict(x_test)
#y_pred = clf.predict_proba(y_test)
stop_test = time()

print("Accuracy: %.2f%%" % accuracy_score(y_test, y_pred))
print("F1 Micro: %.2f%%" % f1_score(y_test, y_pred, average='micro'))
print("F1 Macro: %.2f%%" % f1_score(y_test, y_pred, average='macro'))
print("F1 Weighted: %.2f%%" % f1_score(y_test, y_pred, average='weighted'))
print("Precision Micro: %.2f%%" % precision_score(y_test, y_pred, average='micro'))
print("Precision Macro: %.2f%%" % precision_score(y_test, y_pred, average='macro'))
print("Precision Weighted: %.2f%%" % precision_score(y_test, y_pred, average='weighted'))
print("Recall Micro: %.2f%%" % recall_score(y_test, y_pred, average='micro'))
print("Recall Macro: %.2f%%" % recall_score(y_test, y_pred, average='macro'))
print("Recall Weighted: %.2f%%" % recall_score(y_test, y_pred, average='weighted'))

print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs \n" % (f1_score(y_pred, y_test, average='macro')*100, stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('\nClassification Report:\n', classification_report(y_pred, y_test))
