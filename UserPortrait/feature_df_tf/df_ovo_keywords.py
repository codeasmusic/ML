# coding=utf-8
import sys
sys.path.append("..")
import os
import tools


def get_df_rates(train_input_file, label_index, label_list):
    print "generating df rate ..."

    labels = set(label_list)
    labels_docs_map = get_init_map(labels, 0)
    labels_words_docs_map = get_init_map(labels, {})

    # if label_index == 2:
    #     labels_docs_map = {"1": 0, "2": 0}
    #     labels_words_docs_map = {"1": {}, "2": {}}
    # else:
    #     labels_docs_map = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
    #     labels_words_docs_map = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}, "6": {}}

    infile = open(train_input_file)
    for line in infile:
        line_parts = line.strip().split("\t")
        label = line_parts[label_index]
        text = line_parts[4].split(" ")

        visit_set = set()
        for word in text:
            if word in visit_set:
                continue
            else:
                visit_set.add(word)

            if word in labels_words_docs_map[label]:
                labels_words_docs_map[label][word] += 1
            else:
                labels_words_docs_map[label][word] = 1
        labels_docs_map[label] += 1

    # write_double_map(labels_docs_map, labels_words_docs_map, "dat/df.csv")

    for label, words_docs in labels_words_docs_map.items():
        label_doc_num = labels_docs_map[label] * 1.0
        if label_doc_num == 0:
            raise ValueError("label: " + label + ", doc_num is 0.")

        for word, doc in words_docs.items():
            words_docs[word] = doc / label_doc_num

    return labels_words_docs_map


def get_tf_rates(train_input_file, label_index, label_list):
    print "generating tf rate ..."

    labels = set(label_list)
    labels_total_map = get_init_map(labels, 0)
    labels_wordcnt_map = get_init_map(labels, {})

    # if label_index == 2:
    #     labels_total_map = {"1": 0, "2": 0}
    #     labels_wordcnt_map = {"1": {}, "2": {}}
    # else:
    #     labels_total_map = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0}
    #     labels_wordcnt_map = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}, "6": {}}

    infile = open(train_input_file)
    for line in infile:
        line_parts = line.strip().split("\t")
        label = line_parts[label_index]
        text = line_parts[4].split(" ")

        for word in text:
            if word in labels_wordcnt_map[label]:
                labels_wordcnt_map[label][word] += 1
            else:
                labels_wordcnt_map[label][word] = 1
            labels_total_map[label] += 1

    # write_double_map(labels_total_map, labels_wordcnt_map, "dat/tf.csv")

    for label, wordcnt in labels_wordcnt_map.items():
        label_total = labels_total_map[label] * 1.0
        if label_total == 0:
            raise ValueError("label: " + label + ", word total num is 0.")

        for word, cnt in wordcnt.items():
            wordcnt[word] = cnt / label_total

    return labels_wordcnt_map


def get_df_tf_rates(labels_words_docs_map, labels_wordcnt_map):
    import math
    for label, words_docs in labels_words_docs_map.items():
        words_tf = labels_wordcnt_map[label]

        for word, doc in words_docs.items():
            words_docs[word] *= math.sqrt(words_tf[word])
    return labels_words_docs_map


def write_double_map(uni_map, double_map, outfile_name):
    outfile = open(outfile_name, "w")

    for label in uni_map:
        outfile.write("-------" + label + "\t" + str(uni_map[label]) + "\n")
        umap = double_map[label]

        cnt = 1
        sorted_tuple = sorted(umap.items(), key=lambda x: x[1], reverse=True)
        for (key, value) in sorted_tuple:
            outfile.write(key + "," + str(value) + "\t")
            if cnt % 10 == 0:
                outfile.write("\n")
            cnt += 1

        outfile.write("\n\n")
    outfile.close()



def get_df_ovo_keywords(labels_words_docs_map, label_list, topk):
    label_set_list = sorted(set(label_list))
    print label_set_list

    keywords_set = set()
    for target_label in label_set_list:
        for other_label in label_set_list:
            if other_label == target_label:
                continue

            target_map = labels_words_docs_map[target_label]
            other_map = labels_words_docs_map[other_label]
            topk_words_tuple = get_topk_words(target_map, other_map, topk)

            for (word, docs) in topk_words_tuple:
                if word not in keywords_set:
                    keywords_set.add(word)
                    # outfile.write(word + " ")
        # outfile.write("\n")
    # outfile.close()

    keywords_list = []
    for word in keywords_set:
        keywords_list.append(word)

    # tools.save_data(keywords_list, bin_file)
    return keywords_list


def get_init_map(labels, init_value):
    init_map = {}

    for label in labels:
        if type(init_value) == dict:
            init_map[label] = init_value.copy()
        else:
            init_map[label] = init_value
    return init_map


def get_topk_words(target_words_docs, other_words_docs, topk):
    df_ovo_map = {}

    for word, doc in target_words_docs.items():
        if word not in other_words_docs:
            df_ovo_map[word] = doc
        else:
            df_ovo_map[word] = doc - other_words_docs[word]

    sorted_tuple = sorted(df_ovo_map.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_tuple) > topk:
        sorted_tuple = sorted_tuple[:topk]

    return sorted_tuple


def write_keywords(keywords, outfile_name):
    outfile = open(outfile_name, "w")
    cnt = 1
    for w in keywords:
        outfile.write(w + "\t")
        if cnt % 100 == 0:
            outfile.write("\n")
        cnt += 1
    outfile.close()



# if __name__=="__main__":
#     dir_path = "../../offline/"
#     train_file = dir_path + "processed_train/train_uid_label_query_age_demo.csv"
#
#     label_idx = 1
#     bin_file1 = dir_path + "bin/labels_words_docs_map.bin"
#
#     labels_words_docs = get_document_rates(train_file, label_idx, bin_file1)
#
#     bin_file2 = dir_path + "bin/df_ovo_keywords.bin"
#     get_df_ovo_keywords(labels_words_docs, label_idx, 10, bin_file2)

