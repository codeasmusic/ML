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



if __name__ == "__main__":

    # dir_path = "../../offline/"
    dir_path = "../../Data/"
    target_train_list = [
        dir_path + "processed_train/train_age_apns_bigram.csv",
        # dir_path + "processed_train/biclass_separate/age_biclass.csv",

        dir_path + "processed_train/train_gender_apns_bigram.csv",

        dir_path + "processed_train/train_edu_apns_bigram.csv"
        # dir_path + "processed_train/biclass_separate/edu_biclass.csv"
    ]
    #
    # # target_train = dir_path + "processed_train/train_all_apns_bigram.csv"
    target_test = dir_path + "processed_test/TEST_segfile_apns_bigram.csv"

    # train_flag = "_1_2"
    # test_flag = "_one"
    #
    # target_train = dir_path + "processed_train/biclass_separate/age_biclass" + train_flag + ".csv"
    # target_test = dir_path + "predictions/biclass_separate/biclass_age" + test_flag + ".csv"


    label_name = ["age", "gender", "edu"]    # "age", "gender", "edu"
    label_index = [1, 2, 3]  # 1, 2, 3
    keywords_cntlist = [14000, 14000, 14000]
    estimators_list = [100, 100, 200]
    max_samples_list = [1.0, 1.0, 1.0]

    max_feature_list = [0.4, 0.3, 0.25]
    penal = ["l1", "l2", "l1"]

    for i in xrange(len(label_name)):
        if i != 2:
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
        # tools.save_data(feature_train, dir_path + "bin/feature_train_" + str(keywords_count) + ".bin")
        # feature_train = tools.load_data(dir_path + "bin/feature_train_" + str(keywords_count) + ".bin")

        query_idx = 1
        feature_test = zero_one.zero_one_features(target_test, query_idx, keyword_list)
        # tools.save_data(feature_test, dir_path + "bin/feature_test_" + str(keywords_count) + ".bin")
        # feature_test = tools.load_data(dir_path + "bin/feature_test_" + str(keywords_count) + ".bin")

        label_list, label_dec_map = label_encode(label_list)

        estimators = estimators_list[i]
        max_sample = max_samples_list[i]
        max_feature = max_feature_list[i]

        clfs = [
            LogisticRegression(n_jobs=-1),
            LogisticRegression(penalty="l1", n_jobs=-1),
            SGDClassifier(penalty="l1", loss="log"),
            SGDClassifier(penalty="l1", loss="modified_huber"),
            DecisionTreeClassifier(),
            MultinomialNB(),
            BaggingClassifier(base_estimator=SGDClassifier(penalty="l1", loss="log"),
                              n_estimators=estimators,
                              max_samples=max_sample,
                              max_features=max_feature,
                              bootstrap=True, n_jobs=-1),
        ]

        skf = StratifiedKFold(n_splits=5)
        skf_dataset = list(skf.split(feature_train, label_list))

        label_count = len(set(label_list))
        blend_train = np.zeros((feature_train.shape[0], len(clfs) * label_count))
        blend_test = np.zeros((feature_test.shape[0], len(clfs) * label_count))

        for j, clf in enumerate(clfs):
            print j, clf

            blend_test_j = np.zeros((feature_test.shape[0], len(skf_dataset) * label_count))
            for k, (train, test) in enumerate(skf_dataset):
                feature_train_k = feature_train[train]
                label_list_train_k = label_list[train]
                feature_train_holdout = feature_train[test]
                # label_list_test_k = label_list[test]

                clf.fit(feature_train_k, label_list_train_k)
                blend_train[test, j*label_count: (j+1)*label_count] = clf.predict_proba(feature_train_holdout)
                blend_test_j[:, k*label_count: (k+1)*label_count] = clf.predict_proba(feature_test)

            blend_test_j_mean = np.zeros((feature_test.shape[0], label_count))
            indices = np.arange(len(skf_dataset)) * label_count
            for c in xrange(label_count):
                blend_test_j_mean[:, c] = blend_test_j[:, indices].mean(1)
                indices += 1
            blend_test[:, j*label_count: (j+1)*label_count] = blend_test_j_mean

        clf = LogisticRegression(penalty=penal[i])
        clf.fit(blend_train, label_list)
        pred = clf.predict(blend_test)

        pred = label_decode(pred, label_dec_map)
        predict_file = dir_path + "predictions/predict_" + labe_name + ".csv"
        # predict_file = dir_path + "predictions/biclass_separate/biclass_" + labe_name + ".csv"
        # predict_file = dir_path + "predictions/biclass_separate/biclass_" + labe_name + train_flag + ".csv"
        posthandle.output_labels(target_test, pred, predict_file)


