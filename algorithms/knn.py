from time import time

import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from cm import plot_confusion_matrix

data = pd.read_csv('../connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

enc = preprocessing.OneHotEncoder()
x_transformed = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.20, random_state=42)

baseline_model = KNeighborsClassifier(n_jobs=-1)  # Baseline
# clf = KNeighborsClassifier(metric='manhattan',n_neighbors=12,weights='distance',n_jobs=-1)  # F1 Micro
tuned_model = KNeighborsClassifier(metric='minkowski', n_neighbors=8, weights='uniform', n_jobs=-1)  # F1 Macro
# clf = KNeighborsClassifier(metric='manhattan',n_neighbors=10,weights='distance',n_jobs=-1)  # Precision Micro
# clf = KNeighborsClassifier(metric='manhattan',n_neighbors=28,weights='distance',n_jobs=-1)  # Precision Macro

start_train = time()
baseline_model.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = baseline_model.predict(x_test)
stop_test = time()

print("Basline")
print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs \n" % (
f1_score(y_pred, y_test, average='macro') * 100, stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('\nClassification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(confusion_matrix(y_test, y_pred, ['win', 'loss', 'draw']),
                      target_names=['win', 'loss', 'draw'],
                      title='Baseline KNN',
                      normalize=False)

start_train = time()
tuned_model.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = tuned_model.predict(x_test)
stop_test = time()

print("Tuned")
print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs \n" % (
f1_score(y_pred, y_test, average='macro') * 100, stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('\nClassification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(confusion_matrix(y_test, y_pred, ['win', 'loss', 'draw']),
                      target_names=['win', 'loss', 'draw'],
                      title='Optimized KNN',
                      normalize=False)
