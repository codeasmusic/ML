# coding=utf-8
import sys
sys.path.append("..")

import tools
import preprocess
import posthandle
import zero_one_features as zero_one
from feature_df_tf import df_ovo_keywords as df
from feature_low_dims import word_dict as wd
from feature_low_dims import word2vec as w2v

import sklearn
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize


def label_encode(labe_list):
    # encoding label into index starting from 0,
    # otherwise predict_proba (base.py) meet error

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


tra_text_idx = 4
tes_text_idx = 1
def high_dims_features(tra_set, tes_set, labe_idx, labe_list, topk_words):
    words_df = df.get_df_rates(tra_set, labe_idx, labe_list)
    words_tf = df.get_tf_rates(tra_set, labe_idx, labe_list)
    words_df_tf = df.get_df_tf_rates(words_df, words_tf)

    keyword_list = df.get_df_ovo_keywords(words_df_tf, labe_list, topk_words)
    print "keywords: ", len(keyword_list)


    high_tra = zero_one.zero_one_features(tra_set, tra_text_idx, keyword_list)
    high_tes = zero_one.zero_one_features(tes_set, tes_text_idx, keyword_list)
    return high_tra, high_tes


def low_dims_features(tra_set, tes_set, labe_idx, labe_list):
    pos_list = [{"n", "nr", "nrfg", "nrt", "ns", "nt", "nz"},
                {"v", "vg", "vd", "vn", "vi"},
                {"a", "ad", "ag", "an"}]

    words_set1 = wd.get_words_set(tra_set, tra_text_idx, min_tf=10)
    rate_map = wd.get_rate_map(tra_set, labe_idx, labe_list, words_set1)
    rate_train = wd.get_tf_or_rate_feature(tra_set, tra_text_idx, rate_map, pos_list)
    rate_train = normalize(rate_train)

    tf_map = wd.get_tf_map(tra_set, labe_idx, labe_list, words_set1)
    tf_train = wd.get_tf_or_rate_feature(tra_set, tra_text_idx, tf_map, pos_list)
    tf_train = normalize(tf_train)

    vector_file = "../../Data/sogou_dataset/vector_sohu_100.csv"
    words_set2 = wd.get_words_set(tra_set, tra_text_idx, min_tf=0)
    vectors_map, vector_len = w2v.get_vectors_map(vector_file, words_set2)
    word2vec_train = w2v.get_word2vec_feature(tra_set, tra_text_idx, vectors_map, vector_len)
    word2vec_train = normalize(word2vec_train)


    rate_test = wd.get_tf_or_rate_feature(tes_set, tes_text_idx, rate_map, pos_list)
    rate_test = normalize(rate_test)

    tf_test = wd.get_tf_or_rate_feature(tes_set, tes_text_idx, tf_map, pos_list)
    tf_test = normalize(tf_test)

    word2vec_test = w2v.get_word2vec_feature(tes_set, tes_text_idx, vectors_map, vector_len)
    word2vec_test = normalize(word2vec_test)

    low_tra = np.hstack((rate_train, tf_train, word2vec_train))
    low_tes = np.hstack((rate_test, tf_test, word2vec_test))
    return low_tra, low_tes


def blend_features(clfs, feature_tra, feature_tes, labe_list, cv):
    skf = StratifiedKFold(n_splits=cv)
    skf_dataset = list(skf.split(feature_tra, labe_list))

    labe_count = len(set(labe_list))
    blend_tra = np.zeros((feature_tra.shape[0], len(clfs) * labe_count))
    blend_tes = np.zeros((feature_tes.shape[0], len(clfs) * labe_count))

    for j, clf in enumerate(clfs):
        print j, clf
        blend_test_j = np.zeros((feature_tes.shape[0], len(skf_dataset) * labe_count))

        for k, (train, test) in enumerate(skf_dataset):
            # print "Fold", k
            feature_tra_k = feature_tra[train]
            labels_tra_k = labe_list[train]
            feature_tra_holdout = feature_tra[test]
            # label_list_test_k = label_list[test]

            clf.fit(feature_tra_k, labels_tra_k)
            blend_tra[test, j * labe_count: (j + 1) * labe_count] = clf.predict_proba(feature_tra_holdout)
            blend_test_j[:, k * labe_count: (k + 1) * labe_count] = clf.predict_proba(feature_tes)

        blend_test_j_mean = np.zeros((feature_tes.shape[0], labe_count))
        indices = np.arange(len(skf_dataset)) * labe_count
        for c in xrange(labe_count):
            blend_test_j_mean[:, c] = blend_test_j[:, indices].mean(1)
            indices += 1
        blend_tes[:, j * labe_count: (j + 1) * labe_count] = blend_test_j_mean

    return blend_tra, blend_tes



if __name__ == "__main__":

    dir_path = "../../offline/"
    # dir_path = "../../Data/"
    # target_train_list = [
    #     dir_path + "processed_train/train_age_apns_bigram.csv",
    #     dir_path + "processed_train/train_gender_apns.csv",
    #     dir_path + "processed_train/train_edu_apns_bigram.csv"
    # ]

    high_train_set = dir_path + "processed_train/train_all_apns.csv"
    high_test_set = dir_path + "processed_test/TEST_segfile_apns.csv"

    low_train_set = dir_path + "processed_train/train_all_ap.csv"
    low_test_set = dir_path + "processed_test/TEST_segfile_ap.csv"


    label_name = ["age", "gender", "edu"]    # "age", "gender", "edu"
    label_index = [1, 2, 3]  # 1, 2, 3
    topk_list = [7000, 14000, 7000]
    estimators_list = [100, 100, 200]
    max_samples_list = [1.0, 1.0, 1.0]
    max_feature_list = [0.4, 0.3, 0.25]

    for i in xrange(len(label_name)):
        if i != 0:
            continue

        labe_name = label_name[i]
        labe_index = label_index[i]
        topk = topk_list[i]
        print labe_name, topk

        label_list = preprocess.read_labels(high_train_set, labe_index)
        # high_train, high_test = high_dims_features(high_train_set, high_test_set, labe_index, label_list, topk)
        # tools.save_data(high_train, dir_path + "bin/high_train.bin")
        # tools.save_data(high_test, dir_path + "bin/high_test.bin")
        # high_train = tools.load_data(dir_path + "bin/high_train.bin")
        # high_test = tools.load_data(dir_path + "bin/high_test.bin")
        #
        # feature_train1 = high_train
        # feature_test1 = high_test

        low_train, low_test = low_dims_features(low_train_set, low_test_set, labe_index, label_list)
        tools.save_data(low_train, dir_path + "bin/low_train.bin")
        tools.save_data(low_test, dir_path + "bin/low_test.bin")
        # low_train = tools.load_data(dir_path + "bin/low_train.bin")
        # low_test = tools.load_data(dir_path + "bin/low_test.bin")

        feature_train1 = low_train
        feature_test1 = low_test

        # clfs1 = [
        #     LogisticRegression(),
        #     LogisticRegression(penalty="l1"),
        #     SGDClassifier(penalty="l1", loss="log"),
        #     SGDClassifier(penalty="l1", loss="modified_huber"),
        #     DecisionTreeClassifier(),
        #     MultinomialNB(),
        #     BaggingClassifier(base_estimator=SGDClassifier(penalty="l1", loss="log"),
        #                       n_estimators=estimators_list[i],
        #                       max_samples=max_samples_list[i],
        #                       max_features=max_feature_list[i],
        #                       bootstrap=True, n_jobs=-1)]

        clfs1 = [
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=100, n_jobs=-1, criterion='entropy'),
            GradientBoostingClassifier()
        ]

        label_list, label_dec_map = label_encode(label_list)
        blend_train, blend_test = blend_features(clfs1, feature_train1, feature_test1, label_list, cv=3)

        # feature_train2 = np.hstack((blend_train, low_train))
        # feature_test2 = np.hstack((blend_test, low_test))

        feature_train2 = blend_train
        feature_test2 = blend_test

        clf2 = LogisticRegression()
        clf2.fit(feature_train2, label_list)
        pred_labels = clf2.predict(feature_test2)

        pred_labels = label_decode(pred_labels, label_dec_map)
        predict_file = dir_path + "predictions/predict_" + labe_name + ".csv"
        posthandle.output_labels(high_test_set, pred_labels, predict_file)


