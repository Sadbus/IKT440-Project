from time import time

import matplotlib.pyplot as plt
import pandas as pd
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from pyTsetlinMachineParallel.tools import Binarizer
from sklearn import preprocessing
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from cm_plot import plot_confusion_matrix

data = pd.read_csv('connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

b = Binarizer(max_bits_per_feature=10)
b.fit(x)
x_transformed = b.transform(x)

le = preprocessing.LabelEncoder()
y_transformed = le.fit_transform(y)

# Split the dataset in two equal parts
x_train, x_test, y_train, y_test = train_test_split(
    x_transformed, y_transformed, test_size=0.2, random_state=42)

tm = MultiClassTsetlinMachine(2000, 80, 2.0, weighted_clauses=True)

start_train = time()
tm.fit(x_train, y_train, epochs=1, incremental=True)
stop_train = time()

start_test = time()
y_pred = tm.predict(x_test)
stop_test = time()

print("Accuracy: %.2f%% Training: %.2fs Testing: %.2f" % (
    f1_score(y_test, y_pred, average='macro'), stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=["Predicted"], margins=True))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

y_test = le.inverse_transform(y_test)
y_pred = le.inverse_transform(y_pred)

plot_confusion_matrix(confusion_matrix(y_test, y_pred, labels=['win', 'loss', 'draw']),
                      target_names=['win', 'loss', 'draw'],
                      title="Optimized Tsetlin Machine")
