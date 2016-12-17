# coding=utf-8
import sys
from datetime import datetime


def divide_dataset(online_train_file, ratio, off_train, off_test, off_answer):
    infile = open(online_train_file)
    outfile_train = open(off_train, "w")
    outfile_test = open(off_test, "w")
    outfile_answer = open(off_answer, "w")

    divide_cnt = calculate_divide_count(online_train_file, ratio)

    cnt = 0
    for line in infile:
        if cnt < divide_cnt:
            outfile_train.write(line)
        else:
            line_parts = line.strip().split("\t")
            outfile_answer.write(" ".join(line_parts[:4]) + "\n")
            outfile_test.write(line_parts[0] + "\t" + line_parts[4] + "\n")
        cnt += 1

    outfile_train.close()
    outfile_test.close()


def calculate_divide_count(online_train_file, ratio):
    infile = open(online_train_file)

    train_cnt = 0
    for line in infile:
        train_cnt += 1
    infile.close()

    return train_cnt * ratio


def calculate_offline_score(prediction, validate_file):
    infile = open(validate_file)
    validate_map = {}
    total_cnt = 0

    for line in infile:
        line_parts = line.strip().split(" ")
        uid = line_parts[0]
        age = line_parts[1]
        gender = line_parts[2]
        edu = line_parts[3]

        validate_map[uid] = {"age": age, "gender": gender, "edu": edu}
        total_cnt += 1
    infile.close()

    infile = open(prediction)
    correct_cnt = {"age": 0, "gender": 0, "edu": 0}

    for line in infile:
        line_parts = line.strip().split(" ")
        uid = line_parts[0]
        if uid not in validate_map:
            continue

        pred_age = line_parts[1]
        pred_gender = line_parts[2]
        pred_edu = line_parts[3]

        labels = validate_map[uid]
        if pred_age == labels["age"]:
            correct_cnt["age"] += 1
        if pred_gender == labels["gender"]:
            correct_cnt["gender"] += 1
        if pred_edu == labels["edu"]:
            correct_cnt["edu"] += 1
    infile.close()

    score = (correct_cnt["age"] + correct_cnt["gender"] + correct_cnt["edu"]) / (total_cnt * 3.0)
    print "score: ", score


def calculate_offline_single_score(prediction, validate_file, label_index):

    validate_map = {}
    infile = open(validate_file)
    for line in infile:
        line_parts = line.strip().split(" ")
        uid = line_parts[0]
        validate_map[uid] = line_parts[label_index]
    infile.close()

    total_cnt = 0
    correct_cnt = 0
    infile = open(prediction)
    now = datetime.now()
    err_outfile = open("../Offline/predictions/error_pred_"
                       + str(now.hour) + "-" + str(now.minute) + ".csv", "w")

    for line in infile:
        line_parts = line.strip().split(" ")
        uid = line_parts[0]
        if uid not in validate_map:
            print "uid not in test answer: ", uid
            continue

        total_cnt += 1
        pred_value = line_parts[1]
        if pred_value == validate_map[uid]:
            correct_cnt += 1
        else:
            err_outfile.write(uid + " " + validate_map[uid] + " " + pred_value + "\n")
    infile.close()

    score = correct_cnt / (total_cnt * 1.0)
    print "score: ", score



if __name__ == "__main__":

    online_path = "../Data/"
    offline_path = "../Offline/"

    # ----------------------------------divide dataset-------------------------------------
    # online_train = online_path + "processed_train/train_all_ap.csv"
    # offline_train = offline_path + "processed_train/train_all_ap.csv"
    # offline_test = offline_path + "processed_test/TEST_segfile_ap.csv"
    # offline_answer = offline_path + "processed_test/test_answer.csv"
    #
    # divide_dataset(online_train, 0.8, offline_train, offline_test, offline_answer)


    # ----------------------------------calculate score------------------------------------
    # prediction_file = offline_path + "predictions/seg_prediction.csv"
    # test_answer = offline_path + "processed_test/test_answer.csv"
    # calculate_offline_score(prediction_file, test_answers)


    # ------------------------------calculate single score---------------------------------
    label = ""
    label_idx = 0

    if len(sys.argv) == 2:
        pred_path = sys.argv[1]
        if "age" in pred_path:
            label = "age"
            label_idx = 1
        elif "gender" in pred_path:
            label = "gender"
            label_idx = 2
        elif "edu" in pred_path:
            label = "edu"
            label_idx = 3

        test_answer = offline_path + "processed_test/test_answer.csv"
        # test_answer = offline_path + "test_answer.csv"
        calculate_offline_single_score(sys.argv[1], test_answer, label_idx)
