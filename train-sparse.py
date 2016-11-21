import scipy.sparse as sps
import pickle
from sklearn import naive_bayes, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random

def process_file(filename, test=False):
    dataset = []
    result = []
    with open(filename, 'r') as raw_file:
        shape = (640, 400)
        rows, cols = 640, 400
        sparse_matrix = sps.coo_matrix((rows, cols))
        row = []
        col = []
        for line in raw_file:
            val = line.split(',')
            if len(val) == 2:
                val[0] = int(float(val[0].strip()))
                val[1] = int(float(val[1].strip()))
                row.append(val[0])
                col.append(val[1])
            elif len(val) == 1:
                if val[0].find(':') == -1 and len(val[0].strip()) != 0:
                        r = np.array(row)
                        c = np.array(col)
                        d = np.ones((len(r),))
                        sparse_matrix = sparse_matrix + sps.coo_matrix((d, (r, c)), shape=(rows, cols))
                        dataset.append(sparse_matrix.toarray().flatten())
                        if 'cover' in val[0]:
                            #print 'one'
                            result.append(0)
                        else:
                            result.append(1)
                        sparse_matrix = sps.coo_matrix((rows, cols))
                        row = []
                        col = []
        if test == True:
            r = np.array(row)
            c = np.array(col)
            d = np.ones((len(r),))
            sparse_matrix = sparse_matrix + sps.coo_matrix((d, (r, c)), shape=(rows, cols))
            dataset.append(sparse_matrix.toarray().flatten())                
            sparse_matrix = sps.coo_matrix((rows, cols))
                
    return (dataset, result)

def main():
    X, Y = process_file('new-output.csv')
    
    print len(X), len(Y)
    print len(X[0]), len(X[1])
    #print X[0]
    #print X[0].getnnz(), X[1].getnnz()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=random.randint(0,100)
    )

    #X_train = np.reshape(X_train, (len(Y_train)))
    #X_test = np.reshape(X_test, (len(Y_test)))

    # clf = naive_bayes.GaussianNB()
    clf = SVC(kernel='linear')
    # clf = DecisionTreeClassifier()
    # clf = LinearSVC()
    # clf = RandomForestClassifier()
    clf.fit(X_train, Y_train)
    print 'prediction with train data'
    print metrics.accuracy_score(Y_train, clf.predict(X_train))
    print 'prediction with test data'
    print metrics.accuracy_score(Y_test, clf.predict(X_test))
    #print 'confusion matrix'
    #print metrics.confusion_matrix(Y_test, clf.predict(X_test))
    print 'Testing with new data'
    A, B = process_file('ball-3.csv', True)
    print A
    print 'Cover' if clf.predict(A) == 0 else 'Straight'
    #print clf.predict(X_test)
    #print Y_test

if __name__ == '__main__':
    main()
