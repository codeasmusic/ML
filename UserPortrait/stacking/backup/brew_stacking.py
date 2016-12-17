# coding=utf-8
import sys
sys.path.append("..")

import tools
import preprocess
import posthandle
import zero_one_features as zero_one
from feature_df_tf import df_ovo_keywords as df

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import MultinomialNB

from brew.base import Ensemble
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier


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



if __name__ == "__main__":

    # dir_path = "../../offline/"
    dir_path = "../../Data/"
    target_train_list = [
        dir_path + "processed_train/train_age_apns_bigram.csv",
        dir_path + "processed_train/train_gender_apns_bigram.csv",
        dir_path + "processed_train/train_edu_apns_bigram.csv"]

    # target_train = dir_path + "processed_train/train_all_apns.csv"
    target_test = dir_path + "processed_test/TEST_segfile_apns_bigram.csv"

    label_name = ["age", "gender", "edu"]    # "age", "gender", "edu"
    label_index = [1, 2, 3]  # 1, 2, 3
    keywords_cntlist = [7000, 14000, 7000]
    estimators_list = [100, 100, 200]
    max_samples_list = [1.0, 1.0, 1.0]

    max_feature_list = [0.4, 0.3, 0.25]

    for i in xrange(len(label_name)):
        if i == 1:
            continue

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

        print "begin stacking ..."
        label_list, label_dec_map = label_encode(label_list)

        estimators = estimators_list[i]
        max_sample = max_samples_list[i]
        max_feature = max_feature_list[i]

        clfs = [LogisticRegression(),
                LogisticRegression(penalty="l1"),
                SGDClassifier(penalty="l1", loss="log"),
                DecisionTreeClassifier(),
                SGDClassifier(penalty="l1", loss="modified_huber"),  # use "log", "modified_huber"
                MultinomialNB(),
                BaggingClassifier(base_estimator=SGDClassifier(penalty="l1", loss="log"),
                                  n_estimators=estimators,
                                  max_samples=max_sample,
                                  max_features=max_feature,
                                  bootstrap=True, n_jobs=-1)
                ]

        layer_1 = Ensemble(clfs)
        layer_2 = Ensemble([LogisticRegression(penalty="l1")])

        stack = EnsembleStack(cv=3)
        stack.add_layer(layer_1)
        stack.add_layer(layer_2)

        sclf = EnsembleStackClassifier(stack)
        sclf.fit(feature_train, label_list)
        pred = sclf.predict(feature_test)

        pred = label_decode(pred, label_dec_map)
        predict_file = dir_path + "predictions/predict_" + labe_name + ".csv"
        posthandle.output_labels(target_test, pred, predict_file)


