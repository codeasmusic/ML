# coding=utf-8
import numpy as np
import word_dict as wd
import tools


def get_word2vec_feature(dataset, text_index, vectors_map, vector_len):
    line_num = 0
    # stopword_set = tools.get_stop_words("../../offline/stopwords.csv")
    feature_list = np.zeros((tools.get_lines(dataset), vector_len))

    infile = open(dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        feature = np.zeros(vector_len)
        for word_pos in text:
            word = word_pos.split(",")[0]
            if word in vectors_map:
                feature += vectors_map[word]
        feature_list[line_num] = feature
        line_num += 1
    return feature_list


def get_vectors_map(vector_file, words_set):
    flag = True
    vector_len = 0
    vectors_map = {}
    words_set = wd.remove_pos(words_set)

    infile = open(vector_file)
    for line in infile:
        line_parts = line.strip().split(" ")
        if flag:
            flag = False
            vector_len = int(line_parts[1])
            continue

        word = line_parts[0]
        if word not in words_set:
            continue

        vector = np.asarray(line_parts[1:], dtype=np.float)
        vectors_map[word] = vector

    print "word2vec length: ", vector_len
    return vectors_map, vector_len