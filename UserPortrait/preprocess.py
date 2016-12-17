# coding=utf-8
import tools
import jieba
import jieba.posseg as pseg
import numpy as np


def remove_missing(input_file, label_index, outfile_name, all_no_miss):
    miss_cnt = 0
    outfile = open(outfile_name, "w")

    infile = open(input_file)
    for line in infile:
        line_parts = line.strip().split('\t')

        if all_no_miss:
            if line_parts[1] == '0' or line_parts[2] == '0' or line_parts[3] == '0':
                miss_cnt += 1
            else:
                outfile.write(line)
        else:
            label = line_parts[label_index]
            if label == '0':
                miss_cnt += 1
            else:
                outfile.write(line)

    infile.close()
    outfile.close()
    print "users containing missing value: ", miss_cnt


def read_labels(train_set, label_index):
    infile = open(train_set)
    label_list = []

    for line in infile:
        line_parts = line.strip().split("\t")
        label_list.append(line_parts[label_index])
    infile.close()

    return np.asarray(label_list)


def generate_bigrams(train_set, selected_seg_file, stop_words, text_index, outfile_name):
    print "generate bigrams ..."
    selected_words = get_selected_words(selected_seg_file, text_index)

    cnt = 1
    bigrams_map = {}
    out_pos = {"m", "o", "p", "r", "u", "w", "wp", "x"}

    infile = open(train_set)
    for line in infile:
        line_parts = line.strip().split("\t")
        uid = line_parts[0]

        bigrams = ""
        for text in line_parts[text_index:]:
            seglist = pseg.cut(text)
            word_pos_list = []
            for word, flag in seglist:
                word_pos_list.append((word.encode("utf-8"), flag))

            length = len(word_pos_list) - 1
            for i in xrange(length):
                word1 = word_pos_list[i][0]
                pos1 = word_pos_list[i][1]

                word2 = word_pos_list[i+1][0]
                pos2 = word_pos_list[i+1][1]

                if (word1 == "") or (word2 == "") or\
                        (pos1 in out_pos) or (pos2 in out_pos):
                    continue

                if (word1 in selected_words and word2 not in stop_words) or \
                        (word2 in selected_words and word1 not in stop_words):
                    bigrams += word1 + "-" + word2 + " "

        bigrams_map[uid] = bigrams
        if cnt % 1000 == 0:
            print cnt
        cnt += 1

    infile.close()

    append_bigrams(selected_seg_file, bigrams_map, outfile_name)


def generate_bigrams_apns(train_set, selected_seg_file, text_index, outfile_name):
    print "generate bigrams ..."
    selected_words = get_selected_words(selected_seg_file, text_index)

    cnt = 1
    bigrams_map = {}
    infile = open(train_set)
    for line in infile:
        line_parts = line.strip().split("\t")
        uid = line_parts[0]

        bigrams = ""
        for text in line_parts[text_index:]:
            seglist = jieba.cut(text)
            wordlist = []
            for w in seglist:
                wordlist.append(w.encode("utf-8"))

            length = len(wordlist) - 1
            for i in xrange(length):
                word1 = wordlist[i]
                word2 = wordlist[i+1]

                if word1 in selected_words and word2 in selected_words:
                    bigrams += word1 + "-" + word2 + " "

        bigrams_map[uid] = bigrams
        if cnt % 1000 == 0:
            print cnt
        cnt += 1

    infile.close()

    append_bigrams(selected_seg_file, bigrams_map, outfile_name)


def generate_bigrams_ap(train_file, ap_file, text_index, stopword_set, outfile_name):
    print "generate bigrams ..."

    infile1 = open(ap_file)
    infile2 = open(train_file)
    outfile = open(outfile_name, "w")

    count = 0
    for (line1, line2) in zip(infile1, infile2):
        line_parts1 = line1.strip().split("\t")
        text1 = line_parts1[text_index].split(" ")

        line_parts2 = line2.strip().split("\t")
        text2 = line_parts2[text_index:]

        bigrams = []
        for j in xrange(len(text1)-1):
            wp1 = text1[j]
            word_pos1 = wp1.split(",")
            word1 = word_pos1[0]
            if word1 in stopword_set or word1 == "" or word1.isspace():
                continue

            wp2 = text1[j + 1]
            word_pos2 = wp2.split(",")
            word2 = word_pos2[0]
            if word2 in stopword_set or word2 == "" or word2.isspace():
                continue

            for query in text2:
                if word1 in query and word2 in query:
                    bigrams.append(word1 + "-" + word2 + "," + word_pos1[1] + "-" + word_pos2[1])
                    break
        newline = line1.strip() + " " + " ".join(bigrams) + "\n"
        outfile.write(newline)

        count += 1
        if count % 5000 == 0:
            print count

    infile1.close()
    outfile.close()


def get_selected_words(selected_seg_file, text_index):

    infile = open(selected_seg_file)
    selected_words = set()
    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        for w in text:
            if w not in selected_words and w != "":
                selected_words.add(w)
    infile.close()
    return selected_words


def append_bigrams(seg_file, bigrams_map, outfile_name):
    infile = open(seg_file)
    outfile = open(outfile_name, "w")

    for line in infile:
        line_parts = line.strip().split("\t")
        uid = line_parts[0]

        if uid not in bigrams_map:
            print "bigrams_map has no uid: ", uid
            break

        outfile.write(line.strip() + " " + bigrams_map[uid] + "\n")
    infile.close()
    outfile.close()


def remove_rare(dataset, label_index, remove_set, outfile_name):
    infile = open(dataset)
    outfile = open(outfile_name, "w")

    for line in infile:
        line_parts = line.strip().split("\t")
        label = line_parts[label_index]

        if label in remove_set:
            continue

        outfile.write(line)
    infile.close()
    outfile.close()


def combine_biclasses(dataset, label_index, outfile_name):
    infile = open(dataset)
    outfile = open(outfile_name, "w")

    for line in infile:
        line_parts = line.split("\t")
        label = line_parts[label_index]

        new_label = "1"
        if label == "3" or label == "4":
            new_label = "2"
        if label == "5" or label == "6":
            new_label = "3"

        outfile.write("\t".join(line_parts[:label_index]
                                + [new_label] + line_parts[label_index+1:]))
    infile.close()
    outfile.close()


def extract_train_classes(dataset, label_index, labels_set, outfile_name):
    infile = open(dataset)
    outfile = open(outfile_name, "w")

    for line in infile:
        line_parts = line.split("\t")
        label = line_parts[label_index]

        if label in labels_set:
            outfile.write(line)
    infile.close()
    outfile.close()


def extract_test_classes(dataset, pred_file, labels_set, outfile_name):
    infile = open(pred_file)
    uid_set = set()

    for line in infile:
        line_parts = line.strip().split(" ")
        uid = line_parts[0]
        label = line_parts[1]

        if label in labels_set:
            uid_set.add(uid)
    infile.close()

    infile = open(dataset)
    outfile = open(outfile_name, "w")
    for line in infile:
        line_parts = line.split("\t")
        uid = line_parts[0]

        if uid in uid_set:
            outfile.write(line)
    infile.close()
    outfile.close()


def gen_ovo_dataset(dataset, label_index, label_name):
    label_values = ["1", "2", "3", "4", "5", "6"]

    file_path = dataset.split("/")
    out_path = "/".join(file_path[:-1]) + "/ovo_separate/"

    for v1 in label_values:
        for v2 in label_values:
            if v1 == v2 or v1 > v2:
                continue

            labels_set = {v1, v2}
            outfile_name = out_path + label_name + "_" + v1 + "_" + v2 + ".csv"
            extract_train_classes(dataset, label_index, labels_set, outfile_name)


def group_text(dataset):
    group_text_map = {}

    infile = open(dataset)
    for line in infile:
        line_parts = line.strip().split("\t")
        age = line_parts[1]
        gender = line_parts[2]
        edu = line_parts[3]

        if age == "0" or gender == "0" or edu == "0":
            continue

        group = gender + "-" + age + "-" + edu
        if group in group_text_map:
            group_text_map[group] += line_parts[4:]
        else:
            group_text_map[group] = line_parts[4:]
    infile.close()

    # sorted_tuple = sorted(group_text_map.items(), key=lambda x: x[0])
    # for group, text_list in sorted_tuple:
    #     print group, len(text_list)

    return group_text_map


def generate_new_users(group_text_map, unit_count, outfile_name):
    outfile = open(outfile_name, "w")

    uid = 0
    for group, texts in group_text_map.items():
        group_parts = group.split("-")
        gender = group_parts[0]
        age = group_parts[1]
        edu = group_parts[2]

        texts = np.asarray(texts)
        rand_indices = np.random.permutation(texts.size)
        rand_texts = texts[rand_indices]

        end_idx = unit_count
        while end_idx < rand_texts.size:
            outfile.write("user_" + str(uid) + "\t" + age + "\t" + gender + "\t" + edu + "\t"
                          + "\t".join(rand_texts[end_idx-unit_count: end_idx]) + "\n")
            end_idx += unit_count
            uid += 1

        if (end_idx - rand_texts.size) < (unit_count / 2):
            outfile.write("user_" + str(uid) + "\t" + age + "\t" + gender + "\t" + edu + "\t"
                          + "\t".join(rand_texts[end_idx - unit_count: rand_texts.size]) + "\n")
            end_idx += unit_count
            uid += 1
    outfile.close()


def dataset_shuffle(dataset, outfile_name):
    infile = open(dataset)

    texts = []
    for line in infile:
        texts.append(line)
    infile.close()

    texts = np.asarray(texts)
    rand_indices = np.random.permutation(texts.size)
    rand_texts = texts[rand_indices]

    outfile = open(outfile_name, "w")
    for line in rand_texts:
        outfile.write(line)
    outfile.close()


def merge_two_files(file1, file2):
    infile = open(file1)
    outfile = open(file2, "a")

    for line in infile:
        outfile.write(line)
    infile.close()
    outfile.close()


def trans_ap_to_apns(ap_file, text_index, stopword_set, outfile_name):
    infile = open(ap_file)
    outfile = open(outfile_name, "w")

    for line in infile:
        line_parts = line.strip().split("\t")
        text = line_parts[text_index].split(" ")

        word_list = []
        for word_pos in text:
            wp = word_pos.split(",")
            word = wp[0]
            if word == "" or word in stopword_set:
                continue
            word_list.append(word)
        outfile.write("\t".join(line_parts[:text_index]) + "\t" + " ".join(word_list) + "\n")
    infile.close()
    outfile.close()



if __name__ == "__main__":

    dir_path = "../Data/"
    # dir_path = "../Offline/"

    # stopwords_file = dir_path + "stopwords.csv"
    # stopwords_set = tools.get_stop_words(stopwords_file)

    # --------------------------generate bigram for ap---------------------------
    # trainfile = dir_path + "user_tag_query.10W.TRAIN"
    # apfile = dir_path + "processed_train/TRAIN_segfile_complete_pos.csv"
    # output_file = dir_path + "processed_train/TRAIN_segfile_complete_pos_bigram.csv"
    #
    # text_idx = 4
    # generate_bigrams_ap(trainfile, apfile, text_idx, stopwords_set, output_file)


    # -------------------------transform apfile to apns--------------------------
    # apfile = dir_path + "processed_train/train_all_ap_demo.csv"
    # apnsfile = dir_path + "processed_train/train_all_apns_demo.csv"
    # trans_ap_to_apns(apfile, 4, stopwords_set, apnsfile)

    # apfile = dir_path + "processed_test/TEST_segfile_ap_demo.csv"
    # apnsfile = dir_path + "processed_test/TEST_segfile_apns_demo.csv"
    # trans_ap_to_apns(apfile, 1, stopwords_set, apnsfile)


    # -----------------------------generate dataset------------------------------
    # train_file = dir_path + "user_tag_query.10W.TRAIN"
    # # group_texts = group_text(train_file)
    # #
    # # unit_cnt = 125
    # new_dataset = dir_path + "user_tag_query.10W_new.TRAIN"
    # # generate_new_users(group_texts, unit_cnt, new_dataset)
    #
    # merge_two_files(train_file, new_dataset)
    #
    # new_merge_dataset = dir_path + "user_tag_query.20W.TRAIN"
    # dataset_shuffle(new_dataset, new_merge_dataset)

    # ---------------------------generate combine files--------------------------
    # label_idx = 1
    # label_name = "age"
    # train_file = dir_path + "processed_train/train_" + label_name + "_apns_bigram.csv"
    #
    # # # output_file = dir_path + "processed_train/biclass_separate/" + label_name + "_biclass.csv"
    # # # combine_biclasses(train_file, label_idx, output_file)
    # #
    # label_set = {"1"}
    # test_file = dir_path + "processed_test/TEST_segfile_apns_bigram.csv"
    # init_pred_file = dir_path + "predictions/biclass_separate/biclass_" + label_name + ".csv"
    # output_file = dir_path + "predictions/biclass_separate/biclass_" + label_name + "_one.csv"
    # extract_test_classes(test_file, init_pred_file, label_set, output_file)
    #
    # label_set = {"2"}
    # test_file = dir_path + "processed_test/TEST_segfile_apns_bigram.csv"
    # init_pred_file = dir_path + "predictions/biclass_separate/biclass_" + label_name + ".csv"
    # output_file = dir_path + "predictions/biclass_separate/biclass_" + label_name + "_two.csv"
    # extract_test_classes(test_file, init_pred_file, label_set, output_file)
    #
    # label_set = {"3"}
    # test_file = dir_path + "processed_test/TEST_segfile_apns_bigram.csv"
    # init_pred_file = dir_path + "predictions/biclass_separate/biclass_" + label_name + ".csv"
    # output_file = dir_path + "predictions/biclass_separate/biclass_" + label_name + "_three.csv"
    # extract_test_classes(test_file, init_pred_file, label_set, output_file)


    # label_set = {"1", "2"}
    # output_file = dir_path + "processed_train/biclass_separate/" + label_name + "_biclass_1_2.csv"
    # extract_train_classes(train_file, label_idx, label_set, output_file)
    #
    # label_set = {"3", "4"}
    # output_file = dir_path + "processed_train/biclass_separate/" + label_name + "_biclass_3_4.csv"
    # extract_train_classes(train_file, label_idx, label_set, output_file)
    #
    # label_set = {"5", "6"}
    # output_file = dir_path + "processed_train/biclass_separate/" + label_name + "_biclass_5_6.csv"
    # extract_train_classes(train_file, label_idx, label_set, output_file)


    # ---------------------------generate ovo dataset----------------------------
    # label_idx = 3
    # label_nam = "edu"
    # train_file = dir_path + "processed_train/train_" + label_nam + "_apns_bigram.csv"
    #
    # gen_ovo_dataset(train_file, label_idx, label_nam)


    # -------------------------------word segment--------------------------------
    # train_file = dir_path + "user_tag_query.2W.TRAIN"
    # train_seg_file = dir_path + "processed_train/TRAIN_segfile.csv"
    # train_text_index = 4
    # tools.jieba_pseg(train_file, train_text_index, train_seg_file, stopwords_set)
    #
    # test_file = dir_path + "user_tag_query.2W.TEST"
    # test_seg_file = dir_path + "processed_test/TEST_segfile.csv"
    # test_text_index = 1
    # tools.jieba_pseg(test_file, test_text_index, test_seg_file, stopwords_set)


    # -----------------------------generate bigram--------------------------------
    # train_text_index = 4
    # train_file = dir_path + "user_tag_query.20W.TRAIN"
    # complete_train_seg_file = dir_path + "processed_train/TRAIN_segfile_complete_pos_2.csv"
    # tools.jieba_pseg_complete(train_file, train_text_index, complete_train_seg_file)

    # test_text_index = 1
    # test_file = dir_path + "user_tag_query.10W.TEST"
    # complete_test_seg_file = dir_path + "processed_test/TEST_segfile_complete_pos_2.csv"
    # tools.jieba_pseg_complete(test_file, test_text_index, complete_test_seg_file)
    #
    #
    # train_seg_file = dir_path + "processed_train/TRAIN_segfile.csv"
    # train_bigram_file = dir_path + "processed_train/TRAIN_segfile_bigram.csv"
    # generate_bigrams(train_file, train_seg_file, stopwords_set, train_text_index, train_bigram_file)
    #
    # test_seg_file = dir_path + "processed_test/TEST_segfile.csv"
    # test_bigram_file = dir_path + "processed_test/TEST_segfile_bigram.csv"
    # generate_bigrams(test_file, test_seg_file, stopwords_set, test_text_index, test_bigram_file)

    # train_seg_file = dir_path + "processed_train/train_age_apns.csv"
    # train_bigram_file = dir_path + "processed_train/train_age_apns_bigram.csv"
    # generate_bigrams_apns(train_file, train_seg_file, train_text_index, train_bigram_file)
    #
    # train_seg_file = dir_path + "processed_train/TRAIN_segfile_apns.csv"
    # train_bigram_file = dir_path + "processed_train/TRAIN_segfile_apns_bigram.csv"
    # generate_bigrams_apns(train_file, train_seg_file, train_text_index, train_bigram_file)
    #
    # test_seg_file = dir_path + "processed_test/TEST_segfile_apns.csv"
    # test_bigram_file = dir_path + "processed_test/TEST_segfile_apns_bigram.csv"
    # generate_bigrams_apns(test_file, test_seg_file, test_text_index, test_bigram_file)


    # -------------------------dataset without missing---------------------------
    train_seg_file = dir_path + "processed_train/TRAIN_segfile_complete_pos.csv"
    #
    age_idx = 1
    train_age_file = dir_path + "processed_train/train_age_apns.csv"
    remove_missing(train_seg_file, age_idx, train_age_file, False)

    gender_idx = 2
    train_gender_file = dir_path + "processed_train/train_gender_apns.csv"
    remove_missing(train_seg_file, gender_idx, train_gender_file, False)

    edu_idx = 3
    train_edu_file = dir_path + "processed_train/train_edu_apns.csv"
    remove_missing(train_seg_file, edu_idx, train_edu_file, False)

    # train_all_file = dir_path + "processed_train/TRAIN_segfile_complete_pos.csv"
    # remove_missing(train_seg_file, None, train_all_file, all_no_miss=True)


    # -------------------------complete seg without pos---------------------------
    # train_file = dir_path + "user_tag_query.20W.TRAIN"
    # train_seg_file = dir_path + "processed_train/TRAIN_segfile_apns.csv"  # all pos no stopwords
    # train_text_index = 4
    # tools.jieba_cut(train_file, train_text_index, train_seg_file, stopwords_set)
    #
    # test_file = dir_path + "user_tag_query.10W.TEST"
    # test_seg_file = dir_path + "processed_test/TEST_segfile_apns.csv"
    # test_text_index = 1
    # tools.jieba_cut(test_file, test_text_index, test_seg_file, stopwords_set)


    # ---------------------------remove rare classes------------------------------
    # remove = {"1", "2"}
    # label_idx = 3
    # dir_path = "../Offline/"
    #
    # data_set = dir_path + "processed_train/train_all_apns.csv"
    # output = dir_path + "processed_train/train_edu_apns_no12.csv"
    #
    # remove_rare(data_set, label_idx, remove, output)

    # -----------------------generate file for word2vec--------------------------
    # train_file = dir_path + "user_tag_query.10W.TRAIN"
    # test_file = dir_path + "user_tag_query.10W.TEST"
    # output_file = dir_path + "sogou_dataset/word2vec_data.csv"
    #
    # # tools.jieba_cut_w2v(train_file, 4, outfile_name)
    # tools.jieba_cut_w2v(test_file, 1, output_file)
