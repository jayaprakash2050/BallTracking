from sklearn.naive_bayes import GaussianNB

# Define X and Y from reading the input file

clf = GuassianNB()
clf.fit(X, Y)

from sklearn.externals import joblib
joblib.dump(clf, 'nb_model.pkl')
