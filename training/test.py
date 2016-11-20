from sklearn.svm import LinearSVC, SVC
from sklearn.externals import joblib

class NBPredictor(object):

    def __init__(self):
        self.clf = joblib.load('nb_model.pkl')

    def predict(self, X):
        return self.clf.predict(X)

