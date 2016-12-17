# coding=utf-8
import sys
sys.path.append("..")

import tools
import posthandle
import preprocess
import word_dict as wd
import word2vec as w2v
import pos_freq as pf
import categories as cg

import numpy as np
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from scipy import stats

from xgboost.sklearn import XGBClassifier

def get_y_actual(input_file, labe_index):
    y_actual = []

    infile = open(input_file)
    for line in infile:
        line_parts = line.strip().split(" ")
        y = line_parts[labe_index]
        y_actual.append(y)
    return y_actual


if __name__ == "__main__":

    # dir_path = "../../offline/"
    dir_path = "../../Data/"
    target_train_list = [
        dir_path + "processed_train/train_age_ap.csv",
        dir_path + "processed_train/train_gender_ap.csv",
        dir_path + "processed_train/train_edu_ap.csv"]

    # target_train = dir_path + "processed_train/train_all_ap.csv"
    # target_test = dir_path + "processed_test/TEST_segfile_ap.csv"
    target_test = dir_path + "processed_test/TEST_segfile_complete_pos.csv"

    label_name = ["age", "gender", "edu"]  # "age", "gender", "edu"
    label_indices = [1, 2, 3]  # 1, 2, 3

    pos_list = [{"n", "nr", "nrfg", "nrt", "ns", "nt", "nz"},
                {"v", "vg", "vd", "vn", "vi"},
                {"a", "ad", "ag", "an"}]

    for i in xrange(len(label_name)):
        if i != 1:
            continue
        label_nam = label_name[i]
        label_idx = label_indices[i]
        print label_nam

        target_train = target_train_list[i]
        label_list = preprocess.read_labels(target_train, label_idx)
        # test_answer = dir_path + "processed_test/test_answer.csv"
        # y_actual = get_y_actual(test_answer, label_idx)

        # -----------------------------------------train--------------------------------------------

        text_index = 4
        words_set1 = wd.get_words_set(target_train, text_index, min_tf=10)
        rate_map = wd.get_rate_map(target_train, label_idx, label_list, words_set1)
        rate_train = wd.get_tf_or_rate_feature(target_train, text_index, rate_map, pos_list)
        rate_train = normalize(rate_train)
        # # feature_train = rate_train
        #
        tf_map = wd.get_tf_map(target_train, label_idx, label_list, words_set1)
        tf_train = wd.get_tf_or_rate_feature(target_train, text_index, tf_map, pos_list)
        tf_train = normalize(tf_train)
        # # feature_train = tf_train
        #
        vector_file = "../../Data/sogou_dataset/vector_100.csv"
        words_set2 = wd.get_words_set(target_train, text_index, min_tf=0)
        vectors_map, vector_len = w2v.get_vectors_map(vector_file, words_set2)
        word2vec_train = w2v.get_word2vec_feature(target_train, text_index, vectors_map, vector_len)
        word2vec_train = normalize(word2vec_train)
        # feature_train = word2vec_train

        # category_file = "../../Data/scws/categories_level1.csv"
        # category_tra_file = "../../Data/scws/categories_train_firstline.csv"  # categories_train_firstline
        # uid_labels = cg.get_uid_labels(target_train)
        # category_list = cg.get_categories(category_file)
        # uid_categories_map = cg.get_uid_categories_map(category_tra_file, category_list, level=1)
        # tools.save_data(uid_categories_map, dir_path + "bin/uid_categories_map_level2.bin")

        # cg_train = cg.get_categories_feature(uid_categories_map, target_train)
        # cg_train = normalize(cg_train)
        # # feature_train = cg_train


        # # # ------------------------------------------test--------------------------------------------
        text_index = 1
        rate_test = wd.get_tf_or_rate_feature(target_test, text_index, rate_map, pos_list)
        rate_test = normalize(rate_test)
        # # feature_test = rate_test
        #
        tf_test = wd.get_tf_or_rate_feature(target_test, text_index, tf_map, pos_list)
        tf_test = normalize(tf_test)
        # # feature_test = tf_test
        #
        word2vec_test = w2v.get_word2vec_feature(target_test, text_index, vectors_map, vector_len)
        word2vec_test = normalize(word2vec_test)
        # # feature_test = word2vec_test

        # cg_test = cg.get_categories_feature(uid_categories_map, target_test)
        # cg_test = normalize(cg_test)
        # # feature_test = cg_test

        feature_train = np.hstack((rate_train, tf_train, word2vec_train))
        feature_test = np.hstack((rate_test, tf_test, word2vec_test))
        print feature_train.shape

        tools.save_data(feature_train, dir_path + "bin/feature_train.bin")
        tools.save_data(feature_test, dir_path + "bin/feature_test.bin")
        # feature_train = tools.load_data(dir_path + "bin/feature_train.bin")
        # feature_test = tools.load_data(dir_path + "bin/feature_test.bin")

        # sk = SelectPercentile(f_classif, percentile=95)
        # feature_train = sk.fit_transform(feature_train, label_list)
        # feature_test = sk.transform(feature_test)

        le = LabelEncoder()
        label_list = le.fit_transform(label_list)

        clfs = [
            XGBClassifier(n_estimators=50, min_child_weight=0.9),
            RandomForestClassifier(n_estimators=200, n_jobs=-1),
            LogisticRegression(),
            SVC(gamma=0.5),
        ]

        y_preds = np.zeros((len(clfs), len(feature_test)))
        for j, clf in enumerate(clfs):
            clf.fit(feature_train, label_list)
            y_pred = clf.predict(feature_test)
            y_preds[j] = y_pred

        y_pred = stats.mode(y_preds.astype(np.int32))[0][0]
        y_pred = le.inverse_transform(y_pred)
        predict_file = dir_path + "predictions/predict_" + label_nam + ".csv"
        posthandle.output_labels(target_test, y_pred, predict_file)

        # y_pred = le.inverse_transform(y_pred)
        # print accuracy_score(y_actual, y_pred)


        # clfs = [
        #     XGBClassifier(n_estimators=50),
        #     RandomForestClassifier(n_estimators=100, n_jobs=-1),
        #     LogisticRegression(),
        #     SVC(gamma=0.5, probability=True),
        # ]
        #
        # skf = StratifiedKFold(n_splits=3)
        # skf_dataset = list(skf.split(feature_train, label_list))
        #
        # label_count = len(set(label_list))
        # blend_train = np.zeros((feature_train.shape[0], len(clfs) * label_count))
        # blend_test = np.zeros((feature_test.shape[0], len(clfs) * label_count))
        #
        # for j, clf in enumerate(clfs):
        #     print j, clf
        #
        #     blend_test_j = np.zeros((feature_test.shape[0], len(skf_dataset) * label_count))
        #     for k, (train, test) in enumerate(skf_dataset):
        #         feature_train_k = feature_train[train]
        #         label_list_train_k = label_list[train]
        #         feature_train_holdout = feature_train[test]
        #         # label_list_test_k = label_list[test]
        #
        #         clf.fit(feature_train_k, label_list_train_k)
        #         blend_train[test, j * label_count: (j + 1) * label_count] = clf.predict_proba(feature_train_holdout)
        #         blend_test_j[:, k * label_count: (k + 1) * label_count] = clf.predict_proba(feature_test)
        #
        #     blend_test_j_mean = np.zeros((feature_test.shape[0], label_count))
        #     indices = np.arange(len(skf_dataset)) * label_count
        #     for c in xrange(label_count):
        #         blend_test_j_mean[:, c] = blend_test_j[:, indices].mean(1)
        #         indices += 1
        #     blend_test[:, j * label_count: (j + 1) * label_count] = blend_test_j_mean
        #
        # tools.save_data(blend_train, dir_path + "bin/blend_train_" + label_nam + ".bin")
        # tools.save_data(blend_test, dir_path + "bin/blend_test_" + label_nam + ".bin")
        # blend_train = tools.load_data(dir_path + "bin/blend_train_" + label_nam + ".bin")
        # blend_test = tools.load_data(dir_path + "bin/blend_test_" + label_nam + ".bin")

        # clf = LogisticRegression(penalty="l2")
        # clf = LinearSVC(penalty="l1", dual=False)
        # clf = SVC(gamma=0.5)
        # clf.fit(blend_train, label_list)
        # y_pred = clf.predict(blend_test)
        #
        # y_pred = le.inverse_transform(y_pred)
        # print accuracy_score(y_actual, y_pred)







