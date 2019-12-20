from time import time

import pandas as pd
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from pyTsetlinMachineParallel.tools import Binarizer
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('connect-4.data')


x = data.iloc[:, :-1].values
y = data['class']

b = Binarizer(max_bits_per_feature=10)
b.fit(x)
x_transformed = b.transform(x)

le = preprocessing.LabelEncoder()
y_transformed = le.fit_transform(y)

# Split the dataset in two equal parts
x_train, x_test, y_train, y_test = train_test_split(
    x_transformed, y_transformed, test_size=0.5, random_state=42)

n_clauses = [20, 50, 100, 250, 500, 1000, 2000]
t_params = list(range(10, 101, 10))
s_params = list(range(1, 11))

results = {}

for n_clause in n_clauses:
    for t in t_params:
        for s in s_params:
            tm = MultiClassTsetlinMachine(n_clause, t, s, weighted_clauses=True)

            start_train = time()
            tm.fit(x_train, y_train, epochs=1)
            stop_train = time()

            start_test = time()
            y_pred = tm.predict(x_test)
            stop_test = time()

            accuracy = f1_score(y_test, y_pred, average='macro')

            params = '{n_clauses = %d, t = %d, s = %d}' % (n_clause, t, s)
            results[params] = accuracy

            print("%.2f%% for {n_clauses = %d, t = %d, s = %d} Training: %.2fs Testing: %.2fs" % (accuracy, n_clause, t, s, stop_train - start_train, stop_test - start_test))

print('\nBest parameter set found:')
print(max(results, key=results.get))
