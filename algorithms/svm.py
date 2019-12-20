from time import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing, svm
from sklearn.metrics import classification_report, f1_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('../connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

enc = preprocessing.OneHotEncoder()
x_transformed = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.20, random_state=42)

baseline_model = svm.SVC(gamma='scale', decision_function_shape='ovo', random_state=42)

start_train = time()
baseline_model.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = baseline_model.predict(x_test)
stop_test = time()

print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100, stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

disp = plot_confusion_matrix(baseline_model, x_test, y_test,
                             labels=['win', 'loss', 'draw'],
                             cmap=plt.cm.Blues,
                             normalize='true')
disp.ax_.set_title('Optimized Support-Vector Machine')
plt.show()



tuned_model = svm.SVC(decision_function_shape='ovo',
                      C=10,
                      gamma=0.1,
                      kernel='rbf',
                      random_state=42)

start_train = time()
tuned_model.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = tuned_model.predict(x_test)
stop_test = time()

print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (
f1_score(y_pred, y_test, average='macro') * 100, stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

disp = plot_confusion_matrix(tuned_model, x_test, y_test,
                             labels=['win', 'loss', 'draw'],
                             cmap=plt.cm.Blues,
                             normalize='true')
disp.ax_.set_title('Optimized Support-Vector Machine')
plt.show()

