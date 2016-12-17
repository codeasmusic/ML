# coding=utf-8
import sys
sys.path.append("..")

import warnings
warnings.filterwarnings("ignore")

import tools
import preprocess
import posthandle
import zero_one_features as zero_one
from feature_df_tf import df_ovo_keywords as df

import numpy as np
from scipy import stats
from scipy.sparse import vstack, hstack
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB


# encoding label into index starting from 0,
# otherwise predict_proba (base.py) meet error
def label_encode(labe_list):
    label_set = set()

    for label in labe_list:
        if label not in label_set:
            label_set.add(label)
    sorted_list = sorted(label_set)

    count = 0
    label_enc = {}
    label_dec = {}
    for label in sorted_list:
        label_enc[label] = count
        label_dec[count] = label
        count += 1

    enc_label_list = []
    for label in labe_list:
        enc_label_list.append(label_enc[label])

    # use numpy array, otherwise cross-validate (stacker.py) meet error
    enc_label_list = np.asarray(enc_label_list)
    return enc_label_list, label_dec


def label_decode(labe_list, label_dec):
    true_label_list = []

    for label in labe_list:
        true_label_list.append(label_dec[label])
    return true_label_list


def get_bound_list(sample_count):
    seg_unit = 4
    bound_list = [0]
    quarter = ((seg_unit - (sample_count % seg_unit)) + sample_count) / seg_unit
    count = quarter

    while count < sample_count:
        bound_list.append(count)
        count += quarter

    if count != sample_count:
        bound_list.append(sample_count)
    return bound_list


def data_shffle_pred(fea_train, labe_list, fea_test, params):
    bounds = get_bound_list(fea_train.shape[0])
    bounds_len = len(bounds)
    print bounds

    all_preds = []
    for m in xrange(bounds_len - 2):
        quar1_start = bounds[m]
        quar1_end = bounds[m + 1]
        quar1 = fea_train[quar1_start: quar1_end]
        labels1 = labe_list[quar1_start: quar1_end]

        n = m + 1
        while n < bounds_len - 1:
            quar2_start = bounds[n]
            quar2_end = bounds[n + 1]
            quar2 = fea_train[quar2_start: quar2_end]
            labels2 = labe_list[quar2_start: quar2_end]

            quar_merge = vstack((quar1, quar2))
            labels_merge = np.hstack((labels1, labels2))
            n += 1

            print "shuffle: ", len(all_preds), ", quar_merge: ", quar_merge.shape
            pred = stacking_pred(quar_merge, labels_merge, fea_test, params)
            all_preds.append(pred)

    all_preds = np.asarray(all_preds)
    print all_preds.shape

    final_pred = stats.mode(all_preds.astype(np.int))[0][0]
    return final_pred


def stacking_pred(fea_train, labe_list, fea_test, params):
    clfs = [
        LogisticRegression(n_jobs=-1),
        LogisticRegression(penalty="l1", n_jobs=-1),
        SGDClassifier(penalty="l1", loss="log"),
        SGDClassifier(penalty="l1", loss="modified_huber"),
        DecisionTreeClassifier(),
        MultinomialNB(),
        BaggingClassifier(base_estimator=SGDClassifier(penalty="l1", loss="log"),
                          n_estimators=params["estimators"],
                          max_samples=params["max_sample"],
                          max_features=params["max_feature"],
                          bootstrap=True, n_jobs=-1)]

    skf = StratifiedKFold(n_splits=params["cv"])
    skf_dataset = list(skf.split(fea_train, labe_list))

    label_count = len(set(labe_list))
    blend_train = np.zeros((fea_train.shape[0], len(clfs) * label_count))
    blend_test = np.zeros((fea_test.shape[0], len(clfs) * label_count))

    for j, clf in enumerate(clfs):
        print j, clf

        blend_test_j = np.zeros((fea_test.shape[0], len(skf_dataset) * label_count))
        for k, (train, test) in enumerate(skf_dataset):
            feature_train_k = fea_train[train]
            label_list_train_k = labe_list[train]
            feature_train_holdout = fea_train[test]

            clf.fit(feature_train_k, label_list_train_k)
            blend_train[test, j * label_count: (j + 1) * label_count] = clf.predict_proba(feature_train_holdout)
            blend_test_j[:, k * label_count: (k + 1) * label_count] = clf.predict_proba(fea_test)

        blend_test_j_mean = np.zeros((fea_test.shape[0], label_count))
        indices = np.arange(len(skf_dataset)) * label_count
        for c in xrange(label_count):
            blend_test_j_mean[:, c] = blend_test_j[:, indices].mean(1)
            indices += 1
        blend_test[:, j * label_count: (j + 1) * label_count] = blend_test_j_mean

    clf = LogisticRegression(penalty=params["penal"])
    clf.fit(blend_train, labe_list)
    pred = clf.predict(blend_test)

    return pred


if __name__ == "__main__":

    # dir_path = "../../offline/"
    dir_path = "../../Data/"
    target_train_list = [
        dir_path + "processed_train/train_age_apns_bigram.csv",
        dir_path + "processed_train/train_gender_apns_bigram.csv",
        dir_path + "processed_train/train_edu_apns_bigram.csv"
    ]

    # target_train = dir_path + "processed_train/train_all_apns_bigram.csv"
    target_test = dir_path + "processed_test/TEST_segfile_apns_bigram.csv"

    label_name = ["age", "gender", "edu"]    # "age", "gender", "edu"
    label_index = [1, 2, 3]  # 1, 2, 3
    keywords_cntlist = [7000, 14000, 7000]
    estimators_list = [100, 100, 200]
    max_samples_list = [1.0, 1.0, 1.0]

    max_feature_list = [0.4, 0.3, 0.25]
    penal = ["l1", "l2", "l1"]

    for i in xrange(len(label_name)):
        labe_name = label_name[i]
        labe_idx = label_index[i]
        keywords_count = keywords_cntlist[i]
        print labe_name, keywords_count

        target_train = target_train_list[i]
        label_list = preprocess.read_labels(target_train, labe_idx)

        labels_words_df = df.get_df_rates(target_train, labe_idx, label_list)
        labels_words_tf = df.get_tf_rates(target_train, labe_idx, label_list)
        labels_words_df_tf = df.get_df_tf_rates(labels_words_df, labels_words_tf)
        keyword_list = df.get_df_ovo_keywords(labels_words_df_tf, label_list, keywords_count)
        print "keywords: ", len(keyword_list)

        query_idx = 4
        feature_train = zero_one.zero_one_features(target_train, query_idx, keyword_list)
        query_idx = 1
        feature_test = zero_one.zero_one_features(target_test, query_idx, keyword_list)

        # tools.save_data(feature_train, dir_path + "bin/feature_train_" + str(keywords_count) + ".bin")
        # tools.save_data(feature_test, dir_path + "bin/feature_test_" + str(keywords_count) + ".bin")
        # feature_train = tools.load_data(dir_path + "bin/feature_train_" + str(keywords_count) + ".bin")
        # feature_test = tools.load_data(dir_path + "bin/feature_test_" + str(keywords_count) + ".bin")

        label_list, label_dec_map = label_encode(label_list)
        parameters = {"estimators": estimators_list[i], "max_sample": max_samples_list[i],
                      "max_feature": max_feature_list[i], "penal": penal[i], "cv": 5}

        pred_labels = data_shffle_pred(feature_train, label_list, feature_test, parameters)

        predictions = label_decode(pred_labels, label_dec_map)
        predict_file = dir_path + "predictions/predict_" + labe_name + ".csv"
        posthandle.output_labels(target_test, predictions, predict_file)


