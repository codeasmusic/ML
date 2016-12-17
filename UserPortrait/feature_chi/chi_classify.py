# coding=utf-8
import sys
sys.path.append("..")

import tools
import preprocess
import predict
import model_selection as ms

import zero_one_features as zero_one
import tf_features as tf
import tf_idf_features as tf_idf
import chi_features

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import chi2



if __name__ == "__main__":

    dir_path = "../../offline/"
    # dir_path = "../../Data/"
    target_train = dir_path + "processed_train/train_all_apns.csv"
    target_test = dir_path + "processed_test/TEST_segfile_apns.csv"

    # target_train = dir_path + "processed_train/train_all_bigram.csv"
    # target_test = dir_path + "processed_test/TEST_segfile_bigram.csv"
    min_tf = 10

    labels = ["age"]    # "age", "gender", "edu"
    labels_index = [1]  # 1, 2, 3
    keywords_cntlist = [30000]   # 7000, 15000, 7000
    estimators_list = [100]     # 100, 100, 200
    max_feature_list = [0.4]   # 0.4, 0.3, 0.25

    for i in xrange(len(labels)):
        label = labels[i]
        label_idx = labels_index[i]

        keywords_count = keywords_cntlist[i]
        print label, keywords_count
        # target_train = dir_path + "processed_train/train_" + label + "_bigram.csv"
        label_list = preprocess.read_labels(target_train, label_idx)

        word_chi_score = chi_features.cal_chi_score(target_train, 4, min_tf, label_list, chi2)
        # tools.save_data(word_chi_score, dir_path + "bin/word_chi_score.bin")
        # word_chi_score = tools.load_data(dir_path + "bin/word_chi_score.bin")
        keyword_list = chi_features.chi2_keywords(word_chi_score, keywords_count)

        query_idx = 4
        feature_train = zero_one.zero_one_features(target_train, query_idx, keyword_list)
        # feature_train = tf.generate_tf_features(target_train, query_idx, keyword_list)
        # feature_train = tf_idf.generate_tf_idf_features(target_train, query_idx, keyword_list)
        # feature_train = normalize(feature_train, norm="l2")
        feature_train = csr_matrix(feature_train)

        query_idx = 1
        feature_test = zero_one.zero_one_features(target_test, query_idx, keyword_list)
        # feature_test = tf.generate_tf_features(target_test, query_idx, keyword_list)
        # feature_test = tf_idf.generate_tf_idf_features(target_test, query_idx, keyword_list)
        # feature_test = normalize(feature_test, norm="l2")
        feature_test = csr_matrix(feature_test)

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




