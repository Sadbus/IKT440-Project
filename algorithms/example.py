from time import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv('../connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

enc = preprocessing.OneHotEncoder()
x_transformed = enc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.20, random_state=42)

start_train = time()
# Fit
stop_train = time()

start_test = time()
#predict
stop_test = time()



print("Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (f1_score(y_pred, y_test, average='macro')*100, stop_train - start_train, stop_test - start_test))
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
print('\nClassification Report:\n', classification_report(y_pred, y_test))
