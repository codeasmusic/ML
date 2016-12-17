# coding = utf-8
import os
import sys
sys.path.append("..")

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.sparse import csr_matrix
import numpy as np

import tools
import preprocess


def chi2_keywords(sorted_word_score, topk):
    cnt = 0
    feature_words = []
    for (word, score) in sorted_word_score:
        feature_words.append(word)
        cnt += 1
        if cnt >= topk:
            break
    return feature_words


def cal_chi_score(dataset, text_index, min_tf, labels, score_func):
    word_list = get_word_list(dataset, text_index, min_tf)
    print "words larger than min_tf: ", len(word_list)

    X = generate_X_for_chi(dataset, text_index, word_list)

    sk = SelectKBest(score_func, k="all")
    sk.fit_transform(X, labels)
    score_list = sk.scores_

    word_score = {}
    for i in xrange(len(word_list)):
        word_score[word_list[i]] = score_list[i]
    sorted_word_score = sorted(word_score.items(), key=lambda x: x[1], reverse=True)

    return sorted_word_score


def generate_X_for_chi(dataset, text_index, word_list):
    length = len(word_list)
    X = np.zeros((get_lines(dataset), length))

    cnt = 0
    infile = open(dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        curr_words = line_parts[text_index].split(" ")
        tf_map = {}

        for word in curr_words:
            if word == "":
                continue
            if word not in tf_map:
                tf_map[word] = 1
            else:
                tf_map[word] += 1

        feature = np.zeros(length)
        for i in xrange(length):
            word = word_list[i]
            if word in tf_map:
                feature[i] = tf_map[word]

        X[cnt] = feature
        cnt += 1

    return csr_matrix(X)


def get_word_list(dataset, text_index, min_tf):
    tf_map = tools.cal_corpus_tf(dataset, text_index)
    word_list = []

    for word, freq in tf_map.items():
        if freq >= min_tf:
            word_list.append(word)
    return word_list


def get_lines(data_set):
    cnt = 0
    infile = open(data_set)

    for line in infile:
        cnt += 1
    infile.close()
    return cnt



# if __name__ == "__main__":
#     dir_path = "../../Offline/"
#     target_train = dir_path + "processed_train/train_all_bigram_demo.csv"
#
#     label = "age"
#     label_idx = 1
#     label_list = preprocess.read_labels(target_train, label_idx)
#
#     topk = 36
#     text_idx = 4
#     min_tf_value = 10
#     bin_file = dir_path + "bin/chi2_sorted_score.bin"
#     feature_words = chi2_keywords(target_train, text_idx, min_tf_value, label_list,
#                                   chi2, topk, bin_file)
#
#     cnt = 0
#     outfile = open("feature_words.csv", "w")
#     for word in feature_words:
#         outfile.write(word + "\t")
#         cnt += 1
#         if cnt % 10 == 0:
#             outfile.write("\n")
#     outfile.close()
#
#     # corpus_tf = tools.cal_corpus_tf(target_train, 4)
#     # sorted_corpus_tf = sorted(corpus_tf.items(), key=lambda x: x[1])
#     # print len(corpus_tf)
#     #
#     # while True:
#     #     cnt = 0
#     #     lessk = int(raw_input("tf less than:\t"))
#     #     for (w, tf) in sorted_corpus_tf:
#     #         if tf > lessk:
#     #             print cnt
#     #             break
#     #         else:
#     #             cnt += 1

