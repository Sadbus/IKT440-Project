import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred, ['win', 'loss', 'draw'])
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + ['win', 'loss', 'draw'])
    ax.set_yticklabels([''] + ['win', 'loss', 'draw'])

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] < thresh else "black")
    fig.tight_layout()

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Read data set into panda data frame
data = pd.read_csv('connect-4.data')
data_in_numpy = data.values

# Seperate features (x) and labels (y)
x = data.iloc[:, :-1].values
y = data['class']

# Encode features into numbers
enc = preprocessing.OrdinalEncoder()
x_transformed = enc.fit_transform(x)

# Split into test and training data
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.20)

#
# KNN: K Nearest Neighbors
#
KNN = KNeighborsClassifier(n_neighbors=15)

train_start = time()
KNN.fit(x_train, y_train)
train_stop = time()

test_start = time()
y_pred = KNN.predict(x_test)
test_stop = time()

print("\nKNN")
print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (accuracy_score(y_pred, y_test)*100, train_stop - train_start, test_stop - test_start))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "KNN")

#
# GNB: Gaussian Naive Bayes
#
gnb = GaussianNB()

train_start = time()
gnb.fit(x_train, y_train)
train_stop = time()

test_start = time()
y_pred = gnb.predict(x_test)
test_stop = time()

print("\nGNB")
print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (accuracy_score(y_pred, y_test)*100, train_stop - train_start, test_stop - test_start))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "GNB")


#
# Linear SVM
#
lin_SVC = svm.LinearSVC()

start_train = time()
lin_SVC.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = lin_SVC.predict(x_test)
stop_test = time()

print("\nLinear SVC")
print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (accuracy_score(y_test, y_pred)*100, stop_train - start_train, stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "Linear SVC")

#
# Multi-class SVM: Support Vector Machine
#
SVC = svm.SVC(gamma='scale', decision_function_shape='ovo')

start_train = time()
SVC.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = SVC.predict(x_test)
stop_test = time()

print("\nSVM")
print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (accuracy_score(y_pred, y_test)*100, stop_train - start_train, stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "SVC")
