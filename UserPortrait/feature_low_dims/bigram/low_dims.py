# coding=utf-8
import sys
sys.path.append("..")

import tools
import predict
import preprocess
import word_dict as wd
import word2vec as w2v
import pos_freq as pf
import categories as cg

import numpy as np
from sklearn.preprocessing import normalize, scale, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score

from xgboost.sklearn import XGBClassifier



def get_y_list(input_file, labe_index):
    y_list = []

    infile = open(input_file)
    for line in infile:
        line_parts = line.strip().split(" ")
        y = line_parts[labe_index]
        y_list.append(y)
    return y_list


def select_feature_thresh(fitted_clf, fea_train, fea_test, y_actual):
    # from matplotlib import pyplot
    # print clf.feature_importances_
    # pyplot.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
    # pyplot.show()

    thresholds = list(set(fitted_clf.feature_importances_))
    thresholds = [x for x in thresholds if x > 0.]
    thresholds = sorted(thresholds)
    print thresholds

    for thresh in thresholds:
        selection = SelectFromModel(fitted_clf, threshold=thresh, prefit=True)
        select_feature_train = selection.transform(fea_train)

        select_clf = XGBClassifier()
        select_clf.fit(select_feature_train, label_list)

        select_feature_test = selection.transform(fea_test)
        y_pred = select_clf.predict(select_feature_test)
        accuracy = accuracy_score(y_actual, y_pred)
        print("Thresh=%.4f, Features: %d, Accuracy: %.4f" % (thresh, select_feature_train.shape[1], accuracy))


def select_feature(fea_train, fea_test, thresh):
    select_clf = XGBClassifier()
    selection = SelectFromModel(select_clf, threshold=thresh)
    select_train = selection.fit_transform(fea_train, label_list)
    select_test = selection.transform(fea_test)
    print select_train.shape
    return select_train, select_test





if __name__ == "__main__":

    dir_path = "../../offline/"
    # dir_path = "../../Data/"
    # target_train_list = [
    #     dir_path + "processed_train/train_age_apns_bigram.csv",
    #     dir_path + "processed_train/train_gender_apns_bigram.csv",
    #     dir_path + "processed_train/train_edu_apns_bigram.csv"]

    target_train = dir_path + "processed_train/train_all_ap_bigram.csv"
    target_test = dir_path + "processed_test/TEST_segfile_ap.csv"

    label_name = ["age", "gender", "edu"]  # "age", "gender", "edu"
    label_indices = [1, 2, 3]  # 1, 2, 3

    pos_list = [{"n", "nr", "nrfg", "nrt", "ns", "nt", "nz"},
                {"v", "vg", "vd", "vn", "vi"},
                {"a", "ad", "ag", "an"},
                {"n-n"},
                {"n-v", "v-n"},
                {"a-n", "n-a"},
                ]

    # pos_list_bigram = ["n", "v", "a"]


    for i in xrange(len(label_name)):
        if i != 0:
            continue

        label_nam = label_name[i]
        label_idx = label_indices[i]
        print label_nam

        # target_train = target_train_list[i]
        label_list = preprocess.read_labels(target_train, label_idx)
        y_list = get_y_list(dir_path + "processed_test/test_answer.csv", label_idx)
        # # -----------------------------------------train--------------------------------------------
        text_index = 4

        # bigram_pos_set = {"n-n", "n-v", "v-n", "a-n", "n-a"}
        words_set1 = wd.get_words_set_bigram(target_train, text_index, {}, min_tf=10)
        rate_map = wd.get_rate_map(target_train, label_idx, label_list, words_set1)
        rate_train = wd.get_tf_or_rate_feature(target_train, text_index, rate_map, pos_list)
        rate_train = normalize(rate_train)
        # feature_train = rate_train

        # rate_map = wd.get_rate_map(target_train, label_idx, label_list, words_set1)
        # rate_train_bigram = wd.get_tf_or_rate_feature_bigram(target_train, text_index, rate_map, pos_list)
        # rate_train_bigram = normalize(rate_train_bigram)
        # # feature_train = rate_train_bigram

        words_set1 = wd.get_words_set_bigram(target_train, text_index, {}, min_tf=10)
        tf_map = wd.get_tf_map(target_train, label_idx, label_list, words_set1)
        tf_train = wd.get_tf_or_rate_feature(target_train, text_index, tf_map, pos_list)
        tf_train = normalize(tf_train)
        # feature_train = tf_train

        vector_file = "../../Data/sogou_dataset/vector_100.csv"
        words_set2 = wd.get_words_set(target_train, text_index, min_tf=0)
        vectors_map, vector_len = w2v.get_vectors_map(vector_file, words_set2)
        word2vec_train = w2v.get_word2vec_feature(target_train, text_index, vectors_map, vector_len)
        word2vec_train = normalize(word2vec_train)
        # feature_train = word2vec_train
        #
        #
        # category_file = "../../Data/scws/categories_level1.csv"
        # category_tra_file = "../../Data/scws/categories_train_firstline.csv"  # categories_train_firstline
        # uid_labels = cg.get_uid_labels(target_train)
        # category_list = cg.get_categories(category_file)
        # uid_categories_map = cg.get_uid_categories_map(category_tra_file, category_list, level=1)
        # # tools.save_data(uid_categories_map, dir_path + "bin/uid_categories_map_level2.bin")
        #
        # cg_train = cg.get_categories_feature(uid_categories_map, target_train)
        # cg_train = normalize(cg_train)
        # # feature_train = cg_train



        # # ------------------------------------------test--------------------------------------------
        text_index = 1
        rate_test = wd.get_tf_or_rate_feature(target_test, text_index, rate_map, pos_list)
        rate_test = normalize(rate_test)
        # feature_test = rate_test

        # rate_test_bigram = wd.get_tf_or_rate_feature_bigram(target_test, text_index, rate_map, pos_list)
        # rate_test_bigram = normalize(rate_test_bigram)
        # feature_test = rate_test_bigram

        tf_test = wd.get_tf_or_rate_feature(target_test, text_index, tf_map, pos_list)
        tf_test = normalize(tf_test)
        feature_test = tf_test

        word2vec_test = w2v.get_word2vec_feature(target_test, text_index, vectors_map, vector_len)
        word2vec_test = normalize(word2vec_test)
        # feature_test = word2vec_test
        #
        # cg_test = cg.get_categories_feature(uid_categories_map, target_test)
        # cg_test = normalize(cg_test)
        # # feature_test = cg_test


        # select_rate_train, select_rate_test = select_feature(rate_train, rate_test, thresh=0.006)
        # select_tf_train, select_tf_test = select_feature(tf_train, tf_test, thresh=0.0075)
        # select_w2v_train, select_w2v_test = select_feature(word2vec_train, word2vec_test, thresh=0.005)
        #
        # feature_train = np.hstack((select_rate_train, select_tf_train))
        # feature_test = np.hstack((select_rate_test, select_tf_test))

        # feature_train = np.hstack((rate_train, tf_train, word2vec_train, cg_train))
        # feature_test = np.hstack((rate_test, tf_test, word2vec_test, cg_test))

        feature_train = np.hstack((rate_train, tf_train, word2vec_train))
        feature_test = np.hstack((rate_test, tf_test, word2vec_test))

        # tools.save_data(feature_train, dir_path + "bin/feature_train.bin")
        # tools.save_data(feature_test, dir_path + "bin/feature_test.bin")
        # feature_train = tools.load_data(dir_path + "bin/feature_train.bin")
        # feature_test = tools.load_data(dir_path + "bin/feature_test.bin")
        print feature_train.shape

        # sk = SelectPercentile(f_classif, percentile=95)
        # feature_train = sk.fit_transform(feature_train, label_list)
        # feature_test = sk.transform(feature_test)


        clf = XGBClassifier()
        clf.fit(feature_train, label_list)
        y_pred = clf.predict(feature_test)
        accuracy = accuracy_score(y_list, y_pred)
        print "accuracy: ", accuracy


        # clf.fit(feature_train, label_list)
        # # predict_file = dir_path + "predictions/predict_" + label_nam + ".csv"
        # # predict.predict(target_test, feature_test, clf, predict_file)
        #
        # select_feature_thresh(clf, feature_train, feature_test, y_list)


