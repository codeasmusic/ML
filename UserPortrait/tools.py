# coding=utf-8
import jieba
import jieba.posseg as pseg

import pickle
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def histogram(data, configs):
    x = data[0]
    y = data[1]
    bar_width = 0.5

    if 'xticks' in configs:
        xticks = configs['xticks']
        index = np.arange(len(xticks))
        plt.bar(index, y, bar_width)    # bar_width: the width of every bar

        # make the xticks locate in the center of each bar
        plt.xticks(index + bar_width/2, xticks, rotation="vertical")

    else:
        plt.bar(x, y)

    for v1, v2 in zip(x, y):
        # text: show value on each bar
        plt.text(float(v1) + bar_width/2, v2 + 0.5, "%d"%v2, ha="center", va="bottom")

    plt.title(configs['title'])
    plt.xlabel(configs['xlabel'])
    plt.ylabel(configs['ylabel'])
    plt.show()


def jieba_pseg(input_file, text_index, outfile_name, stopword_set):
    infile = open(input_file)
    outfile = open(outfile_name, "w")
    pos_set = {"n", "nr", "ns", "nt", "nz", "v", "vn"}

    cnt = 0
    for line in infile:
        line_parts = line.strip().split("\t")
        text = " ".join(line_parts[text_index:])
        word_pos = pseg.cut(text)

        seg_text = ""
        for word, pos in word_pos:
            word = word.encode('utf-8')
            if word in stopword_set:
                continue

            if pos in pos_set:
                seg_text += word + " "
            if pos == "eng" and len(word) > 1:
                seg_text += word + " "

        outfile.write("\t".join(line_parts[:text_index]) + "\t"
                      + seg_text.strip() + "\n")    # jieba分词的结果是unicode编码，需要转为utf-8

        cnt += 1
        if cnt % 1000 == 0:
            print cnt
    infile.close()
    outfile.close()


def jieba_pseg_complete(input_file, text_index, outfile_name):
    print input_file

    infile = open(input_file)
    outfile = open(outfile_name, "w")

    cnt = 0
    for line in infile:
        line_parts = line.strip().split("\t")
        text = " ".join(line_parts[text_index:])
        word_pos = pseg.cut(text)

        seg_text = ""
        for word, pos in word_pos:
            if word == "" or word.isspace():
                continue
            seg_text += word + "," + pos + " "

        outfile.write("\t".join(line_parts[:text_index]) + "\t"
                      + seg_text.encode('utf-8').strip() + "\n")
        cnt += 1
        if cnt % 1000 == 0:
            print cnt

    infile.close()
    outfile.close()


def jieba_cut_hold_stopwords(input_file, text_index, outfile_name):
    infile = open(input_file)
    outfile = open(outfile_name, "w")

    cnt = 0
    for line in infile:
        line_parts = line.strip().split("\t")
        text = " ".join(line_parts[text_index:])
        seglist = jieba.cut(text)

        seg_text = ""
        for word in seglist:
            word = word.encode('utf-8')
            if word == "" or word.isspace():
                continue
            seg_text += word + " "

        outfile.write("\t".join(line_parts[:text_index]) + "\t" + seg_text.strip() + "\n")

        cnt += 1
        if cnt % 1000 == 0:
            print cnt
    infile.close()
    outfile.close()


def jieba_cut(input_file, text_index, outfile_name, stopword_set):
    infile = open(input_file)
    outfile = open(outfile_name, "w")

    cnt = 0
    for line in infile:
        line_parts = line.strip().split("\t")
        text = " ".join(line_parts[text_index:])
        seglist = jieba.cut(text)

        seg_text = ""
        for word in seglist:
            word = word.encode('utf-8')
            if word in stopword_set or word == "" or word.isspace():
                continue
            seg_text += word + " "

        outfile.write("\t".join(line_parts[:text_index]) + "\t" + seg_text.strip() + "\n")

        cnt += 1
        if cnt % 1000 == 0:
            print cnt
    infile.close()
    outfile.close()


def cal_corpus_tf(dataset, text_index):
    infile = open(dataset)
    tf_map = {}

    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        for word in text:
            if word == "":
                continue

            if word not in tf_map:
                tf_map[word] = 1
            else:
                tf_map[word] += 1

    infile.close()
    return tf_map


def get_words_set(dataset, text_index, min_tf):
    tf_map = cal_corpus_tf(dataset, text_index)
    words_set = set()
    for (word, tf) in tf_map.items():
        if tf > min_tf:
            words_set.add(word)

    print "words_set, before:", len(tf_map), ", after:", len(words_set)
    return words_set


def get_init_map(label_list, init_value):
    init_map = {}
    for label in label_list:
        if label not in init_map:
            if init_value == {}:
                init_map[label] = init_value.copy()
            else:
                init_map[label] = init_value
    return init_map


def get_lines(dataset):
    cnt = 0
    infile = open(dataset)

    for line in infile:
        cnt += 1
    infile.close()
    return cnt


def get_stop_words(stopword_file):

    infile = open(stopword_file)
    stop_words = set()
    for line in infile:
        w = line.strip()
        if w not in stop_words:
            stop_words.add(w)
    infile.close()
    return stop_words


def save_data(data, save_file):
    print "save data ..."
    pickle.dump(data, open(save_file, "wb"))


def load_data(data_file):
    print "load data ..."
    return pickle.load(open(data_file, "rb"))


def write_list(mylist, outfile_name):
    cnt = 0
    outfile = open(outfile_name, "w")

    for value in mylist:
        outfile.write(str(value) + "\t")
        cnt += 1
        if cnt % 10 == 0:
            outfile.write("\n")
    outfile.close()


def write_double_list(double_list, outfile_name):
    outfile = open(outfile_name, "w")

    for single_list in double_list:
        for value in single_list:
            outfile.write(str(value) + "\t")
        outfile.write("\n")
    outfile.close()


def jieba_cut_w2v(input_file, text_index, outfile_name):
    infile = open(input_file)
    outfile = open(outfile_name, "a")

    cnt = 0
    for line in infile:
        line_parts = line.strip().split("\t")

        for text in line_parts[text_index:]:
            seglist = jieba.cut(text)
            outfile.write((" ".join(seglist)).encode("utf-8") + "\n")

        cnt += 1
        if cnt % 10000 == 0:
            print cnt
    infile.close()
    outfile.close()