# coding=utf-8
import tools


def user_cnt(input_file):
    infile = open(input_file)
    user_set = set()

    for line in infile:
        line_parts = line.strip().split('\t')
        user = line_parts[0]

        if user not in user_set:
            user_set.add(user)

    infile.close()
    print len(user_set)


def var_spread_histogram(input_file):
    age_map = {}
    gender_map = {}
    edu_map = {}

    infile = open(input_file)
    for line in infile:
        line_parts = line.strip().split('\t')
        age = line_parts[1]
        gender = line_parts[2]
        edu = line_parts[3]

        if age not in age_map:
            age_map[age] = 1
        else:
            age_map[age] += 1

        if gender not in gender_map:
            gender_map[gender] = 1
        else:
            gender_map[gender] += 1

        if edu not in edu_map:
            edu_map[edu] = 1
        else:
            edu_map[edu] += 1
    infile.close()

    # print "age_map:\n", age_map
    # print "gender_map:\n", gender_mapu
    # print "edu_map:\n", edu_map

    data1 = map2list(age_map)
    configs1 = {"title": "Age", "xlabel": "groups", "ylabel": "count",
                "xticks": (u"未知", "0-18", "19-23", "24-30", "31-40", "41-50", "51-999")}
    tools.histogram(data1, configs1)

    data2 = map2list(gender_map)
    configs2 = {"title": "Gender", "xlabel": "groups", "ylabel": "count",
                "xticks": (u"未知", u"男", u"女")}
    tools.histogram(data2, configs2)

    data3 = map2list(edu_map)
    configs3 = {"title": "Education", "xlabel": "groups", "ylabel": "count",
                "xticks": (u"未知", u"博士", u"硕士", u"大学生", u"高中", u"初中", u"小学")}
    tools.histogram(data3, configs3)


def var_combine_spread_histogram(input_file):
    infile = open(input_file)
    label_map = {}

    for line in infile:
        line_parts = line.strip().split("\t")
        label = int(line_parts[1])

        if label not in label_map:
            label_map[label] = 1
        else:
            label_map[label] += 1
    infile.close()

    print label_map
    data = map2list(label_map, 72)
    configs = {"title": "Combination label", "xlabel": "user groups", "ylabel": "number of users",
               "xticks": ("1-1-1", "1-1-2", "1-1-3", "1-1-4", "1-1-5", "1-1-6",
                          "1-2-1", "1-2-2", "1-2-3", "1-2-4", "1-2-5", "1-2-6",
                          "1-3-1", "1-3-2", "1-3-3", "1-3-4", "1-3-5", "1-3-6",
                          "1-4-1", "1-4-2", "1-4-3", "1-4-4", "1-4-5", "1-4-6",
                          "1-5-1", "1-5-2", "1-5-3", "1-5-4", "1-5-5", "1-5-6",
                          "1-6-1", "1-6-2", "1-6-3", "1-6-4", "1-6-5", "1-6-6",
                          "2-1-1", "2-1-2", "2-1-3", "2-1-4", "2-1-5", "2-1-6",
                          "2-2-1", "2-2-2", "2-2-3", "2-2-4", "2-2-5", "2-2-6",
                          "2-3-1", "2-3-2", "2-3-3", "2-3-4", "2-3-5", "2-3-6",
                          "2-4-1", "2-4-2", "2-4-3", "2-4-4", "2-4-5", "2-4-6",
                          "2-5-1", "2-5-2", "2-5-3", "2-5-4", "2-5-5", "2-5-6",
                          "2-6-1", "2-6-2", "2-6-3", "2-6-4", "2-6-5", "2-6-6")}
    tools.histogram(data, configs)


def map2list(input_map, xticks):
    x_list = []
    y_list = []

    sorted_tuple = sorted(input_map.items(), key=lambda x: x[0])

    cnt = 0
    for key, value in sorted_tuple:
        while key > cnt:
            x_list.append(cnt)
            y_list.append(0)
            cnt += 1
        cnt += 1
        x_list.append(key)
        y_list.append(value)

    while cnt < xticks:
        x_list.append(cnt)
        y_list.append(0)
        cnt += 1

    print len(x_list), x_list
    return x_list, y_list


def get_query_count(dataset, text_index, label_index):
    infile = open(dataset)

    interval_list = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 301]
    init_cnt_map = {}
    for interval in interval_list:
        init_cnt_map[interval] = 0
    label_cnt_map = get_init_map(get_init_list(label_index), init_cnt_map)

    for line in infile:
        line_parts = line.strip().split("\t")
        part = line_parts[:text_index]
        text = line_parts[text_index:]

        label = part[label_index]
        if label == "0":
            continue
        cnt_map = label_cnt_map[label]

        length = len(text)
        for interval in interval_list:
            if length < interval or interval == interval_list[-1]:
                cnt_map[interval] += 1
                break
    infile.close()

    label_name = {1: "Age", 2: "Gender", 3: "Edu"}
    xticks = ("50", "75", "100", "125", "150", "175", "200", "225", "250", "275", "300", ">300")
    for label, cnt_map in label_cnt_map.items():
        sorted_tuple = sorted(cnt_map.items(), key=lambda x: x[0])
        tools.histogram([xrange(len(sorted_tuple)), [y for v, y in sorted_tuple]],
                        configs={"title": label_name[label_index] + ": " + label,
                                 "xlabel": "query count of each user",
                                 "ylabel": "count", "xticks": xticks})


def get_init_map(label_list, init_value):
    init_map = {}

    for label in label_list:
        if label not in init_map:
            if type(init_value) == dict:
                init_map[label] = init_value.copy()
            else:
                init_map[label] = init_value
    return init_map


def get_init_list(label_index):
    if label_index == 2:
        return ["1", "2"]
    else:
        return ["1", "2", "3", "4", "5", "6"]



if __name__ == '__main__':
    dir_path = "../Data/"
    train_file = dir_path + "user_tag_query.10W.TRAIN"

    # user_cnt(train_file)
    # var_spread_histogram(train_file)

    # train_combine_file = "../Data/processed_train/train_uid_label_query_all_combine.csv"
    # var_combine_spread_histogram(train_combine_file)

    get_query_count(train_file, text_index=4, label_index=2)




