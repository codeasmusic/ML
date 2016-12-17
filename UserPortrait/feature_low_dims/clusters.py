# coding=utf-8
import numpy as np
import tools
import word_dict as wd

def get_cluster_feature(dataset, text_index, cluster_map, cluster_cnt):

    line_num = 0
    feature_list = np.zeros((tools.get_lines(dataset), cluster_cnt))
    # stopword_set = tools.get_stop_words("../../offline/stopwords.csv")

    infile = open(dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        cnt_map = {}
        for word_pos in text:
            word = word_pos.split(",")[0]
            if word not in cluster_map:
                continue

            clus = cluster_map[word]
            if clus in cnt_map:
                cnt_map[clus] += 1
            else:
                cnt_map[clus] = 1

        feature = np.zeros(cluster_cnt)
        for k in xrange(cluster_cnt):
            if k in cnt_map:
                feature[k] = cnt_map[k]

        feature_list[line_num] = feature
        line_num += 1

    print "nonzero-columns: ", len(np.where(feature_list.any(axis=0))[0])
    return feature_list



def get_cluster_map(classes_file, words_set):
    cluster_map = {}
    cluster_set = set()
    words_set = wd.remove_pos(words_set)

    infile = open(classes_file)
    for line in infile:
        line_parts = line.strip().split(" ")
        word = line_parts[0]
        cluster = int(line_parts[1])
        if cluster not in cluster_set:
            cluster_set.add(cluster)

        if word not in words_set:
            continue
        cluster_map[word] = cluster

    return cluster_map, len(cluster_set)


# classes_file = "../../Data/sogou_dataset/classes_100.csv"
# words_set = wd.get_words_set(target_train, text_index, min_tf=10)
# cluster_map, cluster_cnt = clusters.get_cluster_map(classes_file, words_set)
# cluster_train = clusters.get_cluster_feature(target_train, text_index, cluster_map, cluster_cnt)
# cluster_train = normalize(cluster_train)
# # feature_train = cluster_train


# cluster_test = clusters.get_cluster_feature(target_test, text_index, cluster_map, cluster_cnt)
# cluster_test = normalize(cluster_test)
# # feature_test = cluster_test