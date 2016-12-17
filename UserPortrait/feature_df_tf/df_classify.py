# coding=utf-8
import sys
sys.path.append("..")

import tools
import preprocess
import predict
import model_selection as ms

import df_ovo_keywords as df
import zero_one_features as zero_one
import tf_features as tf
import tf_idf_features as tf_idf
import time

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier






if __name__ == "__main__":

    dir_path = "../../Offline/"
    # dir_path = "../../Data/"

    target_train = dir_path + "processed_train/train_all_apns.csv"
    target_test = dir_path + "processed_test/TEST_segfile_apns.csv"


    labels = ["age"]    # "age", "gender", "edu"
    labels_index = [1]  # 1, 2, 3
    keywords_cntlist = [2000]   # 7000, 15000, 7000
    estimators_list = [100]     # 100, 100, 200
    max_feature_list = [0.1]   # 0.4, 0.3, 0.25

    for i in xrange(len(labels)):
        label = labels[i]
        label_idx = labels_index[i]

        keywords_count = keywords_cntlist[i]
        print label, keywords_count
        # target_train = dir_path + "processed_train/train_" + label + "_bigram.csv"

        label_list = preprocess.read_labels(target_train, label_idx)

        labels_words_df = df.get_df_rates(target_train, label_idx)
        labels_words_tf = df.get_tf_rates(target_train, label_idx)
        labels_words_df_tf = df.get_df_tf_rates(labels_words_df, labels_words_tf)
        # tools.save_data(labels_words_df_tf, dir_path + "bin/labels_words_df_tf.bin")
        # labels_words_df_tf = tools.load_data(dir_path + "bin/labels_words_df_tf.bin")

        keyword_list = df.get_df_ovo_keywords(labels_words_df_tf, label_idx, keywords_count)
        print "keywords: ", len(keyword_list)

        query_idx = 4
        feature_train = zero_one.zero_one_features(target_train, query_idx, keyword_list)
        # feature_train = tf.generate_tf_features(target_train, query_idx, keyword_list)
        # feature_train = tf_idf.generate_tf_idf_features(target_train, query_idx, keyword_list)
        # feature_train = normalize(feature_train, norm="max")
        feature_train = csr_matrix(feature_train)

        query_idx = 1
        feature_test = zero_one.zero_one_features(target_test, query_idx, keyword_list)
        # feature_test = tf.generate_tf_features(target_test, query_idx, keyword_list)
        # feature_test = tf_idf.generate_tf_idf_features(target_test, query_idx, keyword_list)
        # feature_test = normalize(feature_test, norm="max")
        feature_test = csr_matrix(feature_test)
        #
        # tools.save_data(feature_train, dir_path + "bin/feature_train_" + str(keywords_count) + ".bin")
        # tools.save_data(feature_test, dir_path + "bin/feature_test_" + str(keywords_count) + ".bin")
        # feature_train = tools.load_data(dir_path + "bin/feature_train_" + str(keywords_count) + ".bin")
        # feature_test = tools.load_data(dir_path + "bin/feature_test_" + str(keywords_count) + ".bin")

        predict_file = dir_path + "predictions/predict_" + label + "_" + str(keywords_count) + ".csv"

        sgd = SGDClassifier(penalty="l1")
        estimators = estimators_list[i]
        max_sample = 1.0
        max_feature = max_feature_list[i]
        print label, estimators, max_sample, max_feature

        bag = BaggingClassifier(base_estimator=sgd, n_estimators=estimators,
                                max_samples=max_sample, max_features=max_feature,
                                bootstrap=True)

        bag.fit(feature_train, label_list)
        predict.predict(target_test, feature_test, bag, predict_file)

        # svc = LinearSVC(dual=False)
        # svc.fit(feature_train, label_list)
        # predict.predict(target_test, feature_test, svc, predict_file)


        # ---------------------------------------------------------------------------------------
        # params = {
        #     # "base_estimator": [SGDClassifier(loss="hinge", penalty="l1"),
        #     #                    SGDClassifier(loss="hinge", penalty="l2"),
        #     #                    SGDClassifier(loss="log", penalty="l1"),
        #     #                    SGDClassifier(loss="log", penalty="l2")],
        #     #  SGDClassifier(loss="hinge", penalty="l2")
        #     #
        #     "base_estimator": [SGDClassifier(penalty="l1", alpha=0.0001),
        #                        SGDClassifier(penalty="l1", alpha=0.00001),
        #                        SGDClassifier(penalty="l1", alpha=0.000001),
        #                        SGDClassifier(penalty="l1", alpha=0.001),
        #                        SGDClassifier(penalty="l1", alpha=0.01),
        #                        SGDClassifier(penalty="l1", alpha=0.1)],  # 1e-5
        #     #      # SGDClassifier(penalty="l1", l1_ratio=0.1),
        #     #      # SGDClassifier(penalty="l1", l1_ratio=0.2),
        #     #      # SGDClassifier(penalty="l1", l1_ratio=0.25)
        #
        #     # "n_estimators": [10, 30, 50, 100, 150, 200, 250, 300, 400, 500],  # 500
        #     # "max_samples": [0.1, 0.3, 0.5, 0.8, 1.0],   # 0.5
        #     # "max_features": [0.1, 0.3, 0.5, 0.8, 1.0],    # 1.0
        #     # "bootstrap": [True, False],   # True
        #     # "bootstrap_features": [True, False],  # True
        #     # "oob_score": [True, False],   # True
        # }
        # grid_clf = ms.grid_search_clf(BaggingClassifier(n_jobs=1, n_estimators=500, max_samples=0.5,
        #                                                 max_features=1.0, bootstrap=True),
        #                               params, cross_num=3, jobs=1)
        # grid_clf.fit(feature_train, label_list)
        # ms.print_best_params(grid_clf, params)




