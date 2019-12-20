from time import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('../connect-4.data')

x = data.drop(columns='class')
y = data['class']

feature_categories = [['b', 'o', 'x'] for i in range(42)]
enc = preprocessing.OneHotEncoder(categories=feature_categories)
x_transformed = enc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.2, random_state=42)

baseline_model = RandomForestClassifier(n_jobs=-1, random_state=42)  # Baseline
tuned_model = RandomForestClassifier(n_estimators=100, max_features=113, max_depth=29, n_jobs=-1,
                                     random_state=42)  # Tuned

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

start_train = time()
tuned_model.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = tuned_model.predict(x_test)
stop_test = time()

print("Optimized")
print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs \n" % (
f1_score(y_pred, y_test, average='macro') * 100, stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('\nClassification Report:\n', classification_report(y_pred, y_test))

disp = plot_confusion_matrix(tuned_model, x_test, y_test,
                             labels=['win', 'loss', 'draw'],
                             cmap=plt.cm.Blues,
                             normalize='true')
disp.ax_.set_title('Optimized Random Forest')
plt.show()
