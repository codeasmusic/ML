# coding=utf-8
import numpy as np
import tools



def get_tf_or_rate_feature(dataset, text_index, tf_or_rate_map, pos_list):
    infile = open(dataset)
    length = len(pos_list)
    x_length = xrange(length)

    tf_rate_list = tf_or_rate_map["list"]
    word_index_map = tf_or_rate_map["index"]
    unit_size = tf_or_rate_map["size"]

    stopword_set = tools.get_stop_words("../../offline/stopwords.csv")
    feature_list = np.zeros((tools.get_lines(dataset), (length + 1) * unit_size))

    cnt = 0
    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        feature = np.zeros((length + 1, unit_size))
        for word_pos in text:
            if word_pos not in word_index_map:
                continue

            wp = word_pos.split(",")
            word = wp[0]
            pos = wp[1]
            if word == "" or word.isspace() or word in stopword_set:
                continue

            flag = False
            word_index = word_index_map[word_pos]
            for k in x_length:
                if pos in pos_list[k]:
                    feature[k] += tf_rate_list[(word_index * unit_size): ((word_index + 1) * unit_size)]
                    flag = True
                    break
            if not flag:
                feature[-1] += tf_rate_list[(word_index * unit_size): ((word_index + 1) * unit_size)]
        feature_list[cnt] = np.reshape(feature, feature.shape[0] * feature.shape[1])
        cnt += 1
    return feature_list



def get_rate_map(dataset, label_index, label_list, words_set):
    print "get rate map ..."

    wordrate_label_map = get_wordrate_label_map(dataset, label_index, label_list, words_set)

    words_list = list(words_set)
    labels = sorted(list(set(label_list)))

    rate_map = transform_tf_or_rate_map(wordrate_label_map, words_list, labels)
    return rate_map


def get_wordrate_label_map(dataset, label_index, label_list, words_set):
    wordrate_label_map = tools.get_init_map(label_list, {})

    infile = open(dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        label = line_parts[label_index]
        text = line_parts[4].split(" ")

        ratecnt = 0
        visit_set = set()
        wordrate_map = wordrate_label_map[label]

        for word in text:
            if word not in words_set:
                continue

            if word not in visit_set:
                ratecnt += 1
                visit_set.add(word)
                if word in wordrate_map:
                    wordrate_map[word] += 1
                else:
                    wordrate_map[word] = 1
    infile.close()

    word_labelcnt_map = {}
    for word in words_set:
        word_labelcnt_map[word] = 0
        for label, wordrate_map in wordrate_label_map.items():
            if word in wordrate_map:
                word_labelcnt_map[word] += wordrate_map[word]

    for label, wordrate_map in wordrate_label_map.items():
        for word, rate in wordrate_map.items():
            wordrate_map[word] = rate * 1.0 / word_labelcnt_map[word]

    return wordrate_label_map



def get_tf_map(dataset, label_index, label_list, words_set):
    print "get tf map ..."

    wordtf_label_map = get_wordtf_label_map(dataset, label_index, label_list, words_set)

    words_list = list(words_set)
    labels = sorted(list(set(label_list)))

    tf_map = transform_tf_or_rate_map(wordtf_label_map, words_list, labels)
    return tf_map


def get_wordtf_label_map(dataset, label_index, label_list, words_set):
    wordcnt_label_map = tools.get_init_map(label_list, 0)
    wordtf_label_map = tools.get_init_map(label_list, {})

    infile = open(dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        label = line_parts[label_index]
        text = line_parts[4].split(" ")

        wordcnt = 0
        wordtf_map = wordtf_label_map[label]
        for word in text:
            if word not in words_set:
                continue
            wordcnt += 1

            if word in wordtf_map:
                wordtf_map[word] += 1
            else:
                wordtf_map[word] = 1
        wordcnt_label_map[label] += wordcnt
    infile.close()

    for label, wordtf_map in wordtf_label_map.items():
        wordcnt = wordcnt_label_map[label]
        for word, tf in wordtf_map.items():
            wordtf_map[word] = tf * 1.0 / wordcnt

    return wordtf_label_map



def transform_tf_or_rate_map(wordrate_label_map, words_list, labels):
    cnt = 0
    index = 0
    unit_size = len(labels)
    rate_list = np.zeros(len(words_list) * unit_size)
    word_index_map = {}

    for word in words_list:
        for label in labels:
            if word in wordrate_label_map[label]:
                rate_list[cnt] = wordrate_label_map[label][word]
            cnt += 1
        word_index_map[word] = index
        index += 1
    return {"list": rate_list, "index": word_index_map, "size": unit_size}


def get_words_set(dataset, text_index, min_tf):
    tf_map = tools.cal_corpus_tf(dataset, text_index)
    words_set = set()
    for (word, tf) in tf_map.items():
        if tf > min_tf:
            words_set.add(word)

    print "words_set, before:", len(tf_map), ", after:", len(words_set)
    return words_set


def remove_pos(words_set):
    new_words_set = set()

    for word_pos in words_set:
        parts = word_pos.split(",")
        new_words_set.add(parts[0])
    return new_words_set

