import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def plot(model, x_test, y_test, title):
    disp = plot_confusion_matrix(model, x_test, y_test,
                                 labels=['win', 'loss', 'draw'],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title(title + 'normalized true')
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
plot(KNN, x_test, y_test, "Baseline: K-Nearest Neighbors")


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
plot(gnb, x_test.toarray(), y_test, "Baseline: Gaussian Naive Bayes")


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
plot(rf, x_test, y_test, "Baseline: Random Forest")

# Logistic Regression
#
lr = LogisticRegression(max_iter=5000, n_jobs=-1, random_state=42)

start_train = time()
lr.fit(x_train, y_train)
stop_train = time()

start_test = time()
y_pred = lr.predict(x_test)
stop_test = time()

print("\nLogistic Regression")
print("F1 Score: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100,
                                                           stop_train - start_train,
                                                           stop_test - start_test))
print('Classification Report:\n', classification_report(y_pred, y_test))
plot(lr, x_test, y_test, "Baseline: Logistic Regression")


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
plot(lin_SVC, x_test, y_test, "Baseline: Linear SVM")


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
plot(SVC, x_test, y_test, "Baseline: SVM")

