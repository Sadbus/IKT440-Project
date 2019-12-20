import pickle

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../connect-4.data')

x = data.drop(columns='class')
y = data['class']

feature_categories = [['b', 'o', 'x'] for i in range(42)]
enc = preprocessing.OneHotEncoder(categories=feature_categories)

x_transformed = enc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, max_features=113, max_depth=29, n_jobs=-1, random_state=42)

print("Fitting ...")
rf.fit(x_train, y_train)

print("Saving model to file ...")
filename = 'random_forest_model.sav'
pickle.dump(rf, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

print("Finished.")

