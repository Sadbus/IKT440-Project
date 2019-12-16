import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score
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

# Seperate features (x) and labels (y)
x = data.iloc[:, :-1].values
y = data['class']

# Encode features into numbers
enc = preprocessing.OneHotEncoder()
x_transformed = enc.fit_transform(x)

# Split into test and training data
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.20, random_state=42)


#
# KNN: K Nearest Neighbors
#
KNN = KNeighborsClassifier(n_jobs=-1)

start_train = time()
KNN.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = KNN.predict(x_test)
stop_test = time()

print("\nKNN")
print("F1 Score: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100,
                                                           stop_train - start_train,
                                                           stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "KNN Baseline (" +  str(f1_score(y_pred, y_test, average='macro')*100) + "%)")


#
# GNB: Gaussian Naive Bayes
#
gnb = GaussianNB()


start_train = time()
gnb.fit(x_train.toarray(), y_train)
stop_train = time()

start_test = time()
y_pred = gnb.predict(x_test.toarray())
stop_test = time()

print("\nGNB")
print("F1 Score: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100,
                                                           stop_train - start_train,
                                                           stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "GNB Baseline (" +  str(f1_score(y_pred, y_test, average='macro')*100) + "%)")


#
# Random Forest
#
rf = RandomForestClassifier(n_jobs=-1, random_state=42)

start_train = time()
rf.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = rf.predict(x_test)
stop_test = time()

print("\nRandom Forest")
print("F1 Score: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100,
                                                           stop_train - start_train,
                                                           stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "Random Fores Baseline (" +  str(f1_score(y_pred, y_test, average='macro')*100) + "%)")


#
# Linear Regression
#
lr = LinearRegression(n_jobs=-1)

start_train = time()
lr.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = lr.predict(x_test)
stop_test = time()

print("\nRandom Forest")
print("F1 Score: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100,
                                                           stop_train - start_train,
                                                           stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "Linear Regression Baseline (" +  str(f1_score(y_pred, y_test, average='macro')*100) + "%)")


#
# Logistic Regression
#
lr = LogisticRegression(n_jobs=-1, random_state=42)

start_train = time()
lr.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = lr.predict(x_test)
stop_test = time()

print("\nRandom Forest")
print("F1 Score: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100,
                                                           stop_train - start_train,
                                                           stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "Logistic Regression Baseline (" +  str(f1_score(y_pred, y_test, average='macro')*100) + "%)")


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
print("F1 Score: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100,
                                                           stop_train - start_train,
                                                           stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "Linear SVC Baseline (" +  str(f1_score(y_pred, y_test, average='macro')*100) + "%)")


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
print("F1 Score: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100,
                                                           stop_train - start_train,
                                                           stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot_confusion_matrix(y_test, y_pred, "SVC Baseline (" +  str(f1_score(y_pred, y_test, average='macro')*100) + "%)")

