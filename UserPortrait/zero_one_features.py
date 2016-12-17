# coding=utf-8
import numpy as np
from scipy.sparse import csr_matrix, vstack


def zero_one_features(data_set, text_index, keywords_list):
    print "generate zero-one features ..."

    length = len(keywords_list)
    feature_list = []

    infile = open(data_set)
    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        wordset = set(text)

        feature = np.zeros(length)
        for k in xrange(length):
            if keywords_list[k] in wordset:
                feature[k] = 1
        feature_list.append(csr_matrix(feature))

    feature_list = vstack(feature_list, format="csr")
    return feature_list


def get_lines(data_set):
    cnt = 0
    infile = open(data_set)

    for line in infile:
        cnt += 1
    infile.close()
    return cnt