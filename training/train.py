from sklearn import naive_bayes, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

# Define X and Y from reading the input file
data = pd.read_csv('../output-goodshots-processed.csv').as_matrix()

X = data[:, :-1]
Y = data[:, -1:]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

Y_train = np.reshape(Y_train, (len(Y_train)))
Y_test = np.reshape(Y_test, (len(Y_test)))

# clf = naive_bayes.GaussianNB()
clf = SVC(kernel='poly')
# clf = DecisionTreeClassifier()
# clf = LinearSVC()
# clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

print metrics.accuracy_score(Y_test, clf.predict(X_test))

from sklearn.externals import joblib
joblib.dump(clf, 'nb_model.pkl')