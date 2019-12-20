import pickle

import pandas as pd
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('connect-4.data')

x = data.drop(columns='class')
y = data['class']

feature_categories = [['b', 'o', 'x'] for i in range(42)]
enc = preprocessing.OneHotEncoder(categories=feature_categories)

x_transformed = enc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100,
                            max_features=113,
                            max_depth=29,
                            n_jobs=-1,
                            random_state=42)

svm = svm.SVC(decision_function_shape='ovo',
              C=10,
              gamma=0.1,
              kernel='rbf',
              random_state=42,
              probability=True)

print("Fitting RF model ...")
rf.fit(x_train, y_train)

print("Fitting SVM model ...")
svm.fit(x_train, y_train)

print("Saving models to file ...")
filename = 'random_forest_model.sav'
pickle.dump(rf, open(filename, 'wb'))

filename = 'svm_model.sav'
pickle.dump(svm, open(filename, 'wb'))

print("Finished.")
