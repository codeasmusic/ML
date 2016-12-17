# coding=utf-8
import sys
sys.path.append("..")

import tools
import preprocess
import predict
import model_selection as ms

import zero_one_features as zero_one
import tf_features as tf
import df_ovo_keywords as df
import tf_idf_features as tf_idf

from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.preprocessing import normalize
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import BaggingClassifier
import numpy as np
import random



def get_bagging_list(fea_train, label_train, est_num, sample_ratio, feature_ratio):
    bounds = get_bound_list(fea_train.shape[0])
    bounds_len = len(bounds)
    print bounds

    count = 0
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

            quar_merge = vstack((quar1, quar2))
            labels_merge = np.hstack((labels1, labels2))
            k += 1

            print count
            count += 1

            # print_labels_stat(labels_merge)
            new_bag = get_bagging(quar_merge, labels_merge, est_num, sample_ratio, feature_ratio)
            bag_clf_list.append(new_bag)


    return bag_clf_list


def get_bagging(new_trainset, new_labels, est_num, sample_ratio, feature_ratio):
    est = SGDClassifier(penalty="l1")

    bag_clf = BaggingClassifier(base_estimator=est, n_estimators=est_num,
                                max_samples=sample_ratio, max_features=feature_ratio,
                                bootstrap=True, n_jobs=-1)
    bag_clf.fit(new_trainset, new_labels)
    return bag_clf


def print_labels_stat(labe_list):
    print "-------------label stat--------------"

    label_map = {}
    for label in labe_list:
        if label not in label_map:
            label_map[label] = 1
        else:
            label_map[label] += 1

    sorted_label_map = sorted(label_map.items(), key=lambda x: x[0])
    for label, count in sorted_label_map:
        print label, count


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


def voting(pred_labels):
    voting_labels = []

    for m in xrange(len(pred_labels[0])):
        votes = {}
        for label in pred_labels[:, m]:
            if label not in votes:
                votes[label] = 1
            else:
                votes[label] += 1
        sorted_vote = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        voting_labels.append(sorted_vote[0][0])
    return voting_labels


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

    # dir_path = "../../offline/"
    dir_path = "../../Data/"
    target_train_list = [
        dir_path + "processed_train/train_age_apns_bigram.csv",
        dir_path + "processed_train/train_gender_apns_bigram.csv",
        dir_path + "processed_train/train_edu_apns_bigram.csv"]

    # target_train = dir_path + "processed_train/train_all_apns_bigram.csv"
    target_test = dir_path + "processed_test/TEST_segfile_apns_bigram.csv"

    label_name = ["age", "gender", "edu"]    # "age", "gender", "edu"
    label_index = [1, 2, 3]  # 1, 2, 3
    keywords_cntlist = [6500, 14000, 6500]
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
        # tools.save_data(labels_words_df_tf, dir_path + "bin/labels_words_df_tf.bin")
        # labels_words_df_tf = tools.load_data(dir_path + "bin/labels_words_df_tf.bin")

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


        estimators = estimators_list[i]
        max_sample = max_samples_list[i]
        max_feature = max_feature_list[i]
        print labe_name, estimators, max_sample, max_feature

        bag_list = get_bagging_list(feature_train, label_list, estimators, max_sample, max_feature)

        predict_labels = np.empty((estimators*len(bag_list), feature_test.shape[0]), dtype=str)
        cnt = 0
        for bag in bag_list:
            for (estimator, feature) in zip(bag.estimators_, bag.estimators_features_):
                predict_labels[cnt] = bag.classes_[estimator.predict(feature_test[:, feature])]
                cnt += 1

        vote_labels = voting(predict_labels)

        predict_file = dir_path + "predictions/predict_" + labe_name + ".csv"
        output_labels(target_test, vote_labels, predict_file)

