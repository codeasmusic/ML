# coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append("..")

import os
import tools
import numpy as np


def get_categories_feature(uid_category_map, dataset):
    feature_list = []

    infile = open(dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        uid = line_parts[0]

        if uid in uid_category_map:
            feature_list.append(uid_category_map[uid])
        else:
            print "uid not in uid_category_map: ", uid
    infile.close()
    print "user has features: ", len(feature_list)
    return np.asarray(feature_list)


def zero_one_categories_feature(uid_category_map, dataset, topk):
    feature_list = []

    infile = open(dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        uid = line_parts[0]

        if uid in uid_category_map:
            freq_list = uid_category_map[uid]
            topk_set = sorted(freq_list, reverse=True)[:topk]

            zero_one_list = []
            for f in freq_list:
                if f in topk_set:
                    zero_one_list.append(1)
                else:
                    zero_one_list.append(0)
            feature_list.append(zero_one_list)
        else:
            print "uid not in uid_category_map: ", uid
    infile.close()
    print "user has features: ", len(feature_list)
    return np.asarray(feature_list)


def get_uid_categories_map(category_dataset, categories_list, level):
    uid_feature_map = {}
    category_cnt_map = get_init_category_map(categories_list)

    prev_uid = ""
    infile = open(category_dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        if len(line_parts) == 1:
            if prev_uid == "":
                prev_uid = line_parts[0]
                continue

            feature = []
            for c in categories_list:
                feature.append(category_cnt_map[c])
            uid_feature_map[prev_uid] = feature

            category_cnt_map = get_init_category_map(categories_list)
            prev_uid = line_parts[0]
        else:
            category_parts = line_parts[1].split("::")
            if level <= len(category_parts):
                category = "::".join(category_parts[:level])
                if category in category_cnt_map:
                    category_cnt_map[category] += 1
                else:
                    if category != "~~":
                        print category
            else:
                print line
    infile.close()

    feature = []
    for c in categories_list:
        feature.append(category_cnt_map[c])
    uid_feature_map[prev_uid] = feature

    print "users has category: ", len(uid_feature_map)
    return uid_feature_map


def get_init_category_map(categories_list):
    category_cnt_map = {}
    for c in categories_list:
        category_cnt_map[c] = 0
    return category_cnt_map


def get_all_categories(categories_file, outfile_name, level):
    infile = open(categories_file)
    outfile = open(outfile_name, "a")

    bin_path = "../Data/bin/categories_set.bin"
    if os.path.isfile(bin_path):
        categories_set = tools.load_data(bin_path)
    else:
        categories_set = set()

    count = len(categories_set)
    for line in infile:
        line_parts = line.strip().split("\t")
        if len(line_parts) <= 2:
            continue

        for c in line_parts[1:]:
            if c == "" or c.isspace():
                continue

            c = c.split("[")[0].split("::")
            if len(c) >= level:
                c_name = "::".join(c[:level]).strip()
                if c_name not in categories_set:
                    categories_set.add(c_name)
                    outfile.write(str(count) + ":" + c_name + "\n")
                    count += 1
    infile.close()
    outfile.close()

    print len(categories_set)
    tools.save_data(categories_set, bin_path)


def extract_firstline_category(categories_file, outfile_name):
    infile = open(categories_file)
    outfile = open(outfile_name, "w")

    for line in infile:
        line_parts = line.strip().split("\t")
        if len(line_parts) == 1:
            outfile.write(line)
            continue

        c_name = line_parts[1].split("[")[0].strip()
        outfile.write(line_parts[0] + "\t" + c_name + "\n")
    infile.close()
    outfile.close()


def get_uid_labels(input_file):
    infile = open(input_file)
    uid_labels_map = {}

    for line in infile:
        line_parts = line.split("\t")
        uid = line_parts[0]
        age = line_parts[1]
        gender = line_parts[2]
        edu = line_parts[3]

        uid_labels_map[uid] = {"age": age, "gender": gender, "edu": edu}
    return uid_labels_map


def get_categories(categories_file):
    categories_list = []
    infile = open(categories_file)
    for line in infile:
        line_parts = line.strip().split(":")
        categories_list.append(line_parts[1])
    infile.close()
    return sorted(categories_list)


def init_labels_to_categories(categories_file, label_list):
    categories_list = get_categories(categories_file)
    label_to_categories = {}
    for label in label_list:
        label_to_categories[label] = {}
        for c in categories_list:
            label_to_categories[label][c] = 0
        label_to_categories[label]["others"] = 0
    return label_to_categories


def get_labels_to_categories(category_train_file, uid_labels_map, label_name, init_categories_map, level):
    uid = ""
    category_map = {}

    infile = open(category_train_file)
    for line in infile:
        line_parts = line.strip().split("\t")
        if len(line_parts) == 1:
            uid = line_parts[0]
            try:
                label = uid_labels_map[uid][label_name]
                category_map = init_categories_map[label]
            except KeyError:
                print "uid keyerror: ", line
                continue
        else:
            if uid == "":
                print "uid is null: ", line
                break

            category_parts = line_parts[1].split("::")
            if level <= len(category_parts):
                category = "::".join(category_parts[:level])
                if category in category_map:
                    category_map[category] += 1
                else:
                    category_map["others"] += 1
    return init_categories_map


def draw_label_to_category(labels_to_categories_map, label_name):
    for label, category_map in labels_to_categories_map.items():
        sorted_tuple = sorted(category_map.items(), key=lambda x: x[0])
        x_list = xrange(len(sorted_tuple))
        xticks, y_list = zip(*sorted_tuple)

        configs = {"title": label_name + ": " + label,
                   "xlabel": "categories",
                   "ylabel": "query count",
                   "xticks": xticks}
        tools.histogram([x_list, y_list], configs)


# if __name__ == "__main__":
#     dir_path = "../../Data/"
#
#     # category_file = dir_path + "scws/categories_test_clean.csv"
#     # # category_file = dir_path + "scws/demo.csv"
#     #
#     # # output_file = dir_path + "scws/categories.csv"
#     # # get_all_categories(category_file, output_file, level=3)
#     #
#     # output_file = dir_path + "scws/categories_test_firstline.csv"
#     # extract_firstline_category(category_file, output_file)
#
#     labe_name = "age"
#     train_set = dir_path + "processed_train/train_" + labe_name + "_apns_bigram.csv"
#     uid_labels = get_uid_labels(train_set)
#
#     # labe_list = ["1", "2", "3", "4", "5", "6"]
#     category_file = dir_path + "scws/categories_level1.csv"
#     # init_category_map = init_labels_to_categories(category_file, labe_list)
#
#     category_tra_file = dir_path + "scws/categories_train_firstline.csv"
#     # label_to_category = get_labels_to_categories(category_tra_file, uid_labels,
#     #                                              labe_name, init_category_map, level=1)
#     # draw_label_to_category(label_to_category, labe_name)
#
#     category_list = get_categories(category_file)
#     get_uid_categories_map(category_tra_file, category_list, level=1)


