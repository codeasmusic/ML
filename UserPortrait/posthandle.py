# coding=utf-8
import tools


def output_labels(test_set, label_predict, outfile_name):
    infile = open(test_set)
    uid_list = []
    for line in infile:
        line_parts = line.split("\t")
        uid_list.append(line_parts[0])
    infile.close()

    outfile = open(outfile_name, "w")
    for p in xrange(len(label_predict)):
        outfile.write(uid_list[p] + " " + str(label_predict[p]) + "\n")
    outfile.close()


def file2map(input_file):
    file_map = {}
    infile = open(input_file)

    for line in infile:
        line_parts = line.strip().split(" ")
        file_map[line_parts[0]] = line_parts[1]
    infile.close()

    return file_map


def combine3maps(target_file, tag, map1, map2, map3, outfile_name):
    infile = open(target_file)
    outfile = open(outfile_name, "w")

    for line in infile:
        line_parts = line.split(tag)
        uid = line_parts[0]

        outfile.write(uid + " " + map1[uid]
                      + " " + map2[uid]
                      + " " + map3[uid] + "\n")
    infile.close()
    outfile.close()


def separate_labels(target_file, num2label_bin_file, outfile_name):
    infile = open(target_file)
    outfile = open(outfile_name, "w")
    num2labels_map = tools.load_data(num2label_bin_file)

    cnt = 0
    for line in infile:
        line_parts = line.strip().split(" ")
        labels = num2labels_map[int(line_parts[1])]
        label_parts = labels.split("-")

        uid = line_parts[0]
        gender = int(label_parts[0])
        age = int(label_parts[1])
        edu = int(label_parts[2])

        if gender < 1 or gender > 2:
            print "exception: gender is wrong at ", cnt
        if age < 1 or age > 6:
            print "exception: age is wrong at ", cnt
        if edu < 1 or edu > 6:
            print "exception: edu is wrong at ", cnt

        outfile.write(uid + " " + str(age) + " " + str(gender) + " " + str(edu) + "\n")
        cnt += 1

    infile.close()
    outfile.close()



if __name__ == "__main__":

    file_path = "../Data/predictions/"
    # file_path = "../Offline/predictions/"

    # ------------------------------------------------------------------------------
    # predict_combine_file = file_path + "predict_all_combine.csv"
    #
    # num2labels_bin = "../Data/bin/num2label.bin"
    # prediction_file = file_path + "predict_all_combine_seg.csv"
    # separate_labels(predict_combine_file, num2labels_bin, prediction_file)


    # ------------------------------------------------------------------------------

    age_file = file_path + "predict_age.csv"
    gender_file = file_path + "predict_gender.csv"
    edu_file = file_path + "predict_edu.csv"
    #
    age_map = file2map(age_file)
    gender_map = file2map(gender_file)
    edu_map = file2map(edu_file)
    #

    tag = "\t"
    test_file = "../Data/processed_test/TEST_segfile_apns.csv"

    # tag = " "
    # test_file = "../Offline/test_answers.csv"

    prediction_file = file_path + "seg_prediction.csv"
    combine3maps(test_file, tag, age_map, gender_map, edu_map, prediction_file)

