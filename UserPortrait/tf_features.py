# coding=utf-8
import numpy as np


def generate_tf_features(data_set, text_index, keywords_list):
    print "generate tf features ..."

    length = len(keywords_list)
    feature_list = np.zeros((get_lines(data_set), length))
    wordset = set(keywords_list)

    cnt = 0
    infile = open(data_set)
    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        tf_map = {}
        for word in text:
            if word not in wordset:
                continue
            if word in tf_map:
                tf_map[word] += 1
            else:
                tf_map[word] = 1

        feature = np.zeros(length)
        for j in xrange(length):
            keyword = keywords_list[j]
            if keyword in tf_map:
                feature[j] = tf_map[keyword]

        feature_list[cnt] = feature
        cnt += 1

    return feature_list


def get_lines(data_set):
    cnt = 0
    infile = open(data_set)

    for line in infile:
        cnt += 1
    infile.close()
    return cnt
