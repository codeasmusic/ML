# coding = utf-8
import math


def generate_tf_idf_features(data_set, text_index, keywords_list):
    print "generate tfidf features ..."

    feature_list = []
    infile = open(data_set)
    idf_map = cal_idf(data_set, text_index)

    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")
        tf_map = cal_user_tf(text)

        feature = []
        for keyword in keywords_list:
            if keyword not in tf_map:
                feature.append(0.0)
            else:
                feature.append(tf_map[keyword] * idf_map[keyword])
        feature_list.append(feature)
    return feature_list


def cal_idf(train_set, text_index):
    infile = open(train_set)
    idf_map = {}
    doc_num = 0

    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")
        doc_num += 1

        visit_set = set()
        for word in text:
            if word in visit_set:
                continue
            visit_set.add(word)

            if word not in idf_map:
                idf_map[word] = 1
            else:
                idf_map[word] += 1
    infile.close()

    doc_num *= 1.0
    for word, cnt in idf_map.items():
        idf_map[word] = math.log(doc_num / cnt)
    return idf_map


def cal_user_tf(word_list):
    tf_map = {}
    for word in word_list:
        if word == "":
            continue

        if word not in tf_map:
            tf_map[word] = 1
        else:
            tf_map[word] += 1
    return tf_map
