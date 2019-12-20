import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

# Loading the data set
data = pd.read_csv('connect-4.data')

x = data.iloc[:, :-1].values
y = data['class']

enc = OneHotEncoder()
x_transformed = enc.fit_transform(x)

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.5, random_state=42)

# Set the parameters by cross-validation
svm_params = [
    {
        'kernel': ['rbf'],
        'gamma': [1e-3, 1e-4],
        'C': [1, 10, 100, 1000]
    },
    {
        'kernel': ['linear'],
        'C': [1, 10, 100, 1000]
    }
]

knn_params = {
    'n_neighbors': list(range(1, 31)),
    'weights': ['uniform', 'distance'],
    'metric': ['minkowski', 'euclidean', 'manhattan']
}
rf_params = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
rf2_params = {
    'n_estimator': [1, 2, 4, 8, 16, 32, 64, 100, 200],
    'max_depth': np.linspace(1, 32, 32, endpoint=True),
}
svm_params = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}

#score_func = make_scorer(fbeta_score, beta=0.5, pos_label='win')

print("# Tuning hyper-parameters for F1 Macro")
#clf = GridSearchCV(KNeighborsClassifier(),knn_params,scoring='precision_macro',verbose=1,n_jobs=-1)
#clf = RandomizedSearchCV(KNeighborsClassifier(),knn_params, n_iter=10, scoring=score_func,verbose=1,n_jobs=-1)
clf = GridSearchCV(SVC(),svm_params, scoring='f1_macro',verbose=3,n_jobs=-1)

clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print(clf.best_params_)

print("\nGrid scores on development set:")

means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

print("\nDetailed classification report:")
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
