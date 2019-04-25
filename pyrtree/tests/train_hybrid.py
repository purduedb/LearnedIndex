import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import (RBFSampler, Nystroem)
from sklearn import preprocessing
import argparse

parser = argparse.ArgumentParser(description="Train hybrid model")
parser.add_argument('-i', '--input', type=str,
                    default='datasetGauss.csv', help='Input CSV File')
parser.add_argument('-c', '--components', type=int,
                    default=100, help='No. of Components for RBF approximator')
parser.add_argument('-s', '--scale', action='store_true',
                    help='Normalize input to 0 mean & 1 std.dev')


args = parser.parse_args()

def getData():
    rows = ['ptx', 'pty', 'a','b','c','d','e']
    dataframe = pd.read_csv(args.input, header=None,
                    error_bad_lines=False, delimiter=':', names=rows)

    pts = dataframe.loc[:,'ptx':'pty'].values#.as_matrix()
    rbf_feature = RBFSampler(gamma=7.5, n_components=args.components)
#    nys_feature = Nystroem(kernel='rbf', gamma=5.5, n_components=100)
    X_rbf = rbf_feature.fit_transform(pts)
#    print(rbf_feature.__dict__['random_weights_'].shape)
#    print(rbf_feature.__dict__['random_offset_'].shape)
    new_dim = pts[:, 0]*pts[:, 0] + pts[:, 1]*pts[:, 1]
    pts = np.column_stack((pts, pts[:,0]*pts[:,0], pts[:,1]*pts[:,1], pts[:,0]*pts[:,1], new_dim))
    if args.scale:
        pts = preprocessing.scale(pts)

    paths = dataframe.loc[:, 'a':].values#as_matrix()
    paths = paths.astype(int)

    return pts, X_rbf, paths


class Node(object):
    def __init__(self, level):
        if level == 0:
#            self.classifier = svm.SVC()
#            self.classifier = MLPClassifier(hidden_layer_sizes=(5,3,3,), alpha=1e-7, max_iter=1000)
#            self.classifier = SGDClassifier(alpha=0.0, learning_rate="constant", eta0=1.0)
            self.classifier = svm.LinearSVC(dual=False, tol=1e-10, penalty='l2', max_iter=20000)
        else:
            self.classifier = svm.LinearSVC(dual=False,tol=1e-10, penalty='l2', max_iter=10000 )
#        self.classifier = svm.LinearSVC(dual=False)
#        self.classifier = MLPClassifier(hidden_layer_sizes=(3,), alpha=1e-7, max_iter=10000)
        self.children = []

def train(pts, paths, level):
    Av = np.unique(paths[:, level])
    Av = [x for x in Av if x >= 0]
    if len(Av) == 0:
        return

    root = Node(level)
    classifier = root.classifier
    Y = paths[:, level]
    if level == 0:
        classifier.fit(X_rbf, Y)
        y_pred = classifier.predict(X_rbf)
    else:
        classifier.fit(pts, Y)
        y_pred = classifier.predict(pts)
    print("Train: Level:{} Acc:{}".format(level, float((Y == y_pred).sum())/len(paths)))

    for branch in Av:
        Y = paths[:, level] == branch
        root.children.append(train(pts[Y], paths[Y], level+1))

    return root

def test_tree(pts, paths, root, level):
    Av  = np.unique(paths[:, level])
    Av = [x for x in Av if x >= 0]
    if len(Av) == 0 or len(paths) == 0:
        return 0

    classifier = root.classifier
    Y = paths[:, level]

    curPts = pts if level > 0 else X_rbf
    y_pred = classifier.predict(curPts)

    wrongCt = (Y != y_pred).sum()

    correct = paths[Y == y_pred]
    X = pts[Y == y_pred]

    for branch in Av:
        childPath = correct[:, level] == branch
        wrongCt += test_tree(X[childPath], correct[childPath], root.children[branch], level+1)

    return wrongCt


pts, X_rbf, paths = getData()
root = train(pts, paths, 0)
wrong = test_tree(pts, paths, root, 0)
print(wrong)
print(float(len(paths)-wrong)/len(paths))
