# coding=utf-8

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


def stacking(x_train, y_train, x_test, cv):

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    clfs = [
        MultinomialNB(),
        SGDClassifier(loss="log"),  # predict_proba are not available for loss='hinge'
        LogisticRegression(),
        DecisionTreeClassifier(),
    ]

    skf = StratifiedKFold(n_splits=cv)
    skf_dataset = list(skf.split(x_train, y_train))

    # y_count: the kinds of labels
    y_count = len(set(y_train))

    # blend_train is the probabilities that every clf predicts every label (i.e. y) for every sample
    # it is used to train the clfs in the second layer
    blend_train = np.zeros((x_train.shape[0], len(clfs) * y_count))

    # blend_test is used as the input of the clfs in the second layer to predict the labels of x_test
    blend_test = np.zeros((x_test.shape[0], len(clfs) * y_count))

    for j, clf in enumerate(clfs):
        print j, clf

        # blend_test_j: the probabilities that j-th clf predicts every label for x_test
        blend_test_j = np.zeros((x_test.shape[0], len(skf_dataset) * y_count))
        for k, (train_idx, test_idx) in enumerate(skf_dataset):
            x_train_k = x_train[train_idx]
            y_train_k = y_train[train_idx]
            x_train_holdout = x_train[test_idx]

            clf.fit(x_train_k, y_train_k)
            blend_train[test_idx, j*y_count: (j+1)*y_count] = clf.predict_proba(x_train_holdout)
            blend_test_j[:, k*y_count: (k+1)*y_count] = clf.predict_proba(x_test)

        # because there are len(skf_dataset) blend_test_j for x_test, it needs to calculated the mean value
        blend_test_j_mean = np.zeros((x_test.shape[0], y_count))

        # indices: supposed y_count = 3, indices would be [0, 3, 6]
        # it is used to find the corresponding probabilities of the same label, and calculate the mean
        indices = np.arange(len(skf_dataset)) * y_count
        for c in xrange(y_count):
            blend_test_j_mean[:, c] = blend_test_j[:, indices].mean(1)
            indices += 1
        blend_test[:, j*y_count: (j+1)*y_count] = blend_test_j_mean

    clf = LogisticRegression()
    clf.fit(blend_train, y_train)
    y_pred = clf.predict(blend_test)
    y_pred = le.inverse_transform(y_pred)

    return y_pred
