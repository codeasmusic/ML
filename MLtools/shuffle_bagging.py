# coding=utf-8

import numpy as np
from scipy.sparse import vstack
from scipy import stats

from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder


def shuffle_bagging(x_train, y_train, x_test, bagging_params):
    estimators = bagging_params['n_estimators']
    max_features = bagging_params['max_features']

    # It needs to encoding labels from string to int
    # Because bagging.estimators_ give the transformed int label instead of the original labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    bagging_clfs = train_bagging_clfs(x_train, y_train, estimators, max_features)

    estimator_idx = 0
    y_preds = np.empty((estimators*len(bagging_clfs), x_test.shape[0]), dtype=str)
    for bagging in bagging_clfs:
        for (estimator, feature) in zip(bagging.estimators_, bagging.estimators_features_):
            y_preds[estimator_idx] = estimator.predict(x_test[:, feature])
            estimator_idx += 1

    # voting
    y_pred = stats.mode(y_preds.astype(np.int))[0][0]
    y_pred = le.inverse_transform(y_pred)
    return y_pred

    # This is the old code without using LabelEncoder
    # estimator_idx = 0
    # for bagging in bagging_clfs:
    #     for (estimator, feature) in zip(bagging.estimators_, bagging.estimators_features_):
    #         y_preds[estimator_idx] = bagging.classes_[estimator.predict(x_test[:, feature])]
    #         estimator_idx += 1


def train_bagging_clfs(x_train, y_train, estimators, max_features):
    # seg data into several units, e.g. 4, data => [d1, d2, d3, d4]
    # then combinations: [d1, d2], [d1, d3], [d1, d4], [d2, d3], [d2, d4], [d3, d4]
    # so there are six shuffle dataset to train six BaggingClassifiers
    sample_seglist = get_sample_seglist(x_train.shape[0], seg_unit=4)

    seglist_len = len(sample_seglist)
    print sample_seglist

    bagging_clfs = []
    for j in xrange(seglist_len - 2):
        x1_start = sample_seglist[j]
        x1_end = sample_seglist[j + 1]
        x1 = x_train[x1_start: x1_end]
        y1 = y_train[x1_start: x1_end]

        k = j + 1
        while k < seglist_len - 1:
            x2_start = sample_seglist[k]
            x2_end = sample_seglist[k + 1]
            x2 = x_train[x2_start: x2_end]
            y2 = y_train[x2_start: x2_end]

            # x_merge is csr_matrix here, so use the sparse.vstack
            x_merge = vstack((x1, x2))
            y_merge = np.hstack((y1, y2))
            k += 1

            new_bag = get_bagging(x_merge, y_merge, estimators, max_features)
            bagging_clfs.append(new_bag)

    return bagging_clfs


def get_bagging(x_train, y_train, estimators, max_features):
    base = SGDClassifier(penalty="l1")
    clf = BaggingClassifier(base_estimator=base, n_estimators=estimators,
                            max_features=max_features, n_jobs=-1)

    clf.fit(x_train, y_train)
    return clf


def get_sample_seglist(samples_num, seg_unit):
    seglist = [0]
    unit_num = ((seg_unit - (samples_num % seg_unit)) + samples_num) / seg_unit

    num = unit_num
    while num < samples_num:
        seglist.append(num)
        num += unit_num

    if num != samples_num:
        seglist.append(samples_num)
    return seglist


