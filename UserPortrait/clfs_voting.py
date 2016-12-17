# coding=utf-8


def voting(predict_files, outfile_name):
    if len(predict_files) < 2:
        return

    uid_votings = {}
    for f in xrange(len(predict_files)):
        infile = open(predict_files[f])

        for line in infile:
            line_parts = line.strip().split(" ")
            uid = line_parts[0]
            label = line_parts[1]

            votes = 1
            if uid not in uid_votings:
                uid_votings[uid] = {label: votes}
            else:
                if label not in uid_votings[uid]:
                    uid_votings[uid][label] = votes
                else:
                    uid_votings[uid][label] += votes

    infile = open(predict_files[0])
    outfile = open(outfile_name, "w")

    for line in infile:
        line_parts = line.strip().split(" ")
        uid = line_parts[0]

        votings = uid_votings[uid]
        sort_votings = sorted(votings.items(), key=lambda x: x[1], reverse=True)

        outfile.write(uid + " " + sort_votings[0][0] + "\n")
        # if sort_votings[0][1] == sort_votings[1][1]:
        #     print uid, sort_votings[0][0], sort_votings[1][0], "count: ", sort_votings[1][1]

    infile.close()
    outfile.close()


if __name__ == "__main__":
    dir_path = "../Data/"
    pred_files = []

    label_values = ["1", "2", "3", "4", "5", "6"]
    for v1 in label_values:
        for v2 in label_values:
            if v1 == v2 or v1 > v2:
                continue
            target_train = dir_path + "predictions/edu/predict_edu_" + v1 + "_" + v2 + ".csv"
            pred_files.append(target_train)

    voting(pred_files, dir_path + "predictions/predict_edu_ovo.csv")

