from time import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report, f1_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('../connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

enc = preprocessing.OrdinalEncoder()
x_transformed = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.20, random_state=42)

clf = GaussianNB()

start_train = time()
clf.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = clf.predict(x_test)
stop_test = time()

print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (
    f1_score(y_test, y_pred, average='macro') * 100, stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('\nClassification Report:\n', classification_report(y_test, y_pred))

disp = plot_confusion_matrix(clf, x_test, y_test,
                             labels=['win', 'loss', 'draw'],
                             cmap=plt.cm.Blues,
                             normalize='true')
disp.ax_.set_title('Gaussian Naive Bayes')
plt.show()
