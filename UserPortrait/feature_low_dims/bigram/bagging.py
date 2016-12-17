# coding=utf-8
import sys
sys.path.append("..")

import tools
import preprocess
import predict
import word_dict as wd
import word2vec as w2v

from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import normalize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

import numpy as np
from scipy import stats


def get_bagging_list(fea_train, label_train, est_num, sample_ratio, feature_ratio):
    bounds = get_bound_list(fea_train.shape[0])
    bounds_len = len(bounds)
    print bounds

    bag_cnt = 0
    bag_clf_list = []
    for j in xrange(bounds_len - 2):
        quar1_start = bounds[j]
        quar1_end = bounds[j + 1]
        quar1 = fea_train[quar1_start: quar1_end]
        labels1 = label_train[quar1_start: quar1_end]

        k = j + 1
        while k < bounds_len - 1:
            quar2_start = bounds[k]
            quar2_end = bounds[k + 1]
            quar2 = fea_train[quar2_start: quar2_end]
            labels2 = label_train[quar2_start: quar2_end]

            quar_merge = np.vstack((quar1, quar2))
            labels_merge = np.hstack((labels1, labels2))
            k += 1

            print bag_cnt
            bag_cnt += 1

            # print_labels_stat(labels_merge)
            new_bag = get_bagging(quar_merge, labels_merge, est_num, sample_ratio, feature_ratio)
            bag_clf_list.append(new_bag)


    return bag_clf_list


def get_bagging(new_trainset, new_labels, est_num, sample_ratio, feature_ratio):
    # est = SGDClassifier(penalty="l2")
    est = LogisticRegression()

    bag_clf = BaggingClassifier(base_estimator=est, n_estimators=est_num,
                                max_samples=sample_ratio, max_features=feature_ratio,
                                bootstrap=True, n_jobs=-1)
    bag_clf.fit(new_trainset, new_labels)
    return bag_clf


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


def output_labels(test_file, label_predict, outfile_name):
    infile = open(test_file)
    uid_list = []
    for line in infile:
        line_parts = line.split("\t")
        uid_list.append(line_parts[0])
    infile.close()

    outfile = open(outfile_name, "w")
    for p in xrange(len(label_predict)):
        outfile.write(uid_list[p] + " " + str(label_predict[p]) + "\n")
    outfile.close()



if __name__ == "__main__":

    dir_path = "../../offline/"
    # dir_path = "../../Data/"
    # target_train_list = [
    #     dir_path + "processed_train/train_age_apns_bigram.csv",
    #     dir_path + "processed_train/train_gender_apns_bigram.csv",
    #     dir_path + "processed_train/train_edu_apns_bigram.csv"]

    target_train = dir_path + "processed_train/train_all_ap.csv"
    target_test = dir_path + "processed_test/TEST_segfile_ap.csv"

    label_name = ["age", "gender", "edu"]  # "age", "gender", "edu"
    label_indices = [1, 2, 3]  # 1, 2, 3

    estimators_list = [100, 100, 200]
    max_samples_list = [1.0, 1.0, 1.0]
    max_feature_list = [0.4, 0.3, 0.25]

    pos_list = [{"n", "nr", "nrfg", "nrt", "ns", "nt", "nz"},
                {"v", "vg", "vd", "vn", "vi"},
                {"a", "ad", "ag", "an"}]

    for i in xrange(len(label_name)):
        if i != 0:
            continue

        label_nam = label_name[i]
        label_idx = label_indices[i]

        estimators = estimators_list[i]
        max_sample = max_samples_list[i]
        max_feature = max_feature_list[i]
        print label_nam, estimators, max_sample, max_feature

        # target_train = target_train_list[i]
        label_list = preprocess.read_labels(target_train, label_idx)


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

        vector_file = "../../Data/sogou_dataset/vector_sohu_100.csv"
        words_set2 = wd.get_words_set(target_train, text_index, min_tf=0)
        vectors_map, vector_len = w2v.get_vectors_map(vector_file, words_set2)
        word2vec_train = w2v.get_word2vec_feature(target_train, text_index, vectors_map, vector_len)
        word2vec_train = normalize(word2vec_train)
        # feature_train = word2vec_train


        # ------------------------------------------test--------------------------------------------
        text_index = 1
        rate_test = wd.get_tf_or_rate_feature(target_test, text_index, rate_map, pos_list)
        rate_test = normalize(rate_test)
        # # feature_test = rate_test
        #
        tf_test = wd.get_tf_or_rate_feature(target_test, text_index, tf_map, pos_list)
        tf_test = normalize(tf_test)
        # # feature_test = tf_test

        word2vec_test = w2v.get_word2vec_feature(target_test, text_index, vectors_map, vector_len)
        word2vec_test = normalize(word2vec_test)
        # feature_test = word2vec_test

        feature_train = np.hstack((rate_train, tf_train, word2vec_train))
        feature_test = np.hstack((rate_test, tf_test, word2vec_test))
        print feature_train.shape


        # -----------------------------------------bagging-----------------------------------------

        estimators = estimators_list[i]
        max_sample = max_samples_list[i]
        max_feature = max_feature_list[i]
        print label_nam, estimators, max_sample, max_feature

        bag_list = get_bagging_list(feature_train, label_list, estimators, max_sample, max_feature)

        count = 0
        predict_labels = np.empty((estimators * len(bag_list), feature_test.shape[0]), dtype=str)
        for bag in bag_list:
            for (estimator, feature) in zip(bag.estimators_, bag.estimators_features_):
                predict_labels[count] = bag.classes_[estimator.predict(feature_test[:, feature])]
                count += 1

        voting_labels = stats.mode(predict_labels.astype(np.int))[0][0]

        predict_file = dir_path + "predictions/predict_" + label_nam + ".csv"
        output_labels(target_test, voting_labels, predict_file)

