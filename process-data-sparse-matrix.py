from scipy.sparse import *
import pickle
from sklearn import naive_bayes, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import random

def main():
    dataset = []
    result = []
    with open('new-output.csv', 'r') as raw_file:
        processed_file = open('output-goodshots-processed-x.csv', 'w')
        processed_labels = open('output-goodshots-processed-y.csv', 'w')
        sparse_matrix = [[0 for x in xrange(640)] for y in xrange(400)]
        row = []
        for line in raw_file:
            cols = line.split(',')
            if len(cols) == 2:
                cols[0] = int(float(cols[0].strip()))
                cols[1] = int(float(cols[1].strip()))
                sparse_matrix[cols[1]][cols[1]] = 1
            elif len(cols) == 1:
                if cols[0].find(':') == -1 and len(cols[0].strip()) != 0:
                        dataset.append(coo_matrix(sparse_matrix).tocsr())
                        result.append(cols[0])
                        sparse_matrix = [[0 for x in xrange(640)] for y in xrange(400)]

    X = dataset
    Y = result
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=random.randint(0,100)
    )

    Y_train = np.reshape(Y_train, (len(Y_train)))
    Y_test = np.reshape(Y_test, (len(Y_test)))

    # clf = naive_bayes.GaussianNB()
    clf = SVC(kernel='linear')
    # clf = DecisionTreeClassifier()
    # clf = LinearSVC()
    # clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)

    print metrics.accuracy_score(Y_test, clf.predict(X_test))
    print metrics.confusion_matrix(Y_test, clf.predict(X_test))
    print clf.predict(X_test)
    print Y_tes

if __name__ == '__main__':
    main()
