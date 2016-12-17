# coding=utf-8
import numpy as np
import tools


def get_eng_feature(dataset, text_index):
    print "eng feature ..."
    infile = open(dataset)
    feature_list = []

    line_num = 0
    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        eng_count = 0
        for word_pos in text:
            word_pos_part = word_pos.split(",")
            pos = word_pos_part[1]

            if pos == "eng":
                eng_count += 1
        feature_list.append([eng_count])
    infile.close()

    feature_list = np.asarray(feature_list)
    feature_list = (feature_list - feature_list.min()) * 1.0 / (feature_list.max() - feature_list.min())

    print feature_list[:30]
    return feature_list


# eng_train = pf.get_eng_feature(target_train, text_index)
# eng_test = pf.get_eng_feature(target_test, text_index)

def get_pos_freq(dataset, text_index):
    print "get pos freq feature ..."

    infile = open(dataset)
    pos_freq_map = {}

    import string
    for letter in string.lowercase:
        pos_freq_map[letter] = 0
    dims = len(pos_freq_map)

    line_cnt = 0
    feature_list = np.zeros((tools.get_lines(dataset), dims))

    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        for word_pos in text:
            word_pos_part = word_pos.split(",")
            pos = word_pos_part[1]
            if pos == "":
                pos = "x"
            else:
                pos = pos[0]
            pos_freq_map[pos] += 1

        feature = np.zeros(dims)
        pos_freq_tuple = sorted(pos_freq_map.items(), key=lambda x: x[0])
        for k in xrange(len(pos_freq_tuple)):
            feature[k] = pos_freq_tuple[k][1]

        feature_list[line_cnt] = feature
        line_cnt += 1
    return feature_list
