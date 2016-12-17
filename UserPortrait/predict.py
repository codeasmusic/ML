# coding=utf-8


def predict(test_file, feature_test, clf_model, outfile_name):
    print "Begin predicting..."

    infile = open(test_file)
    uid_list = []

    for line in infile:
        line_parts = line.split("\t")
        uid_list.append(line_parts[0])
    infile.close()

    label_predict = clf_model.predict(feature_test)
    if len(uid_list) != len(label_predict):
        print len(uid_list), len(label_predict)
        print "user count is not equal to the number of predictions!"
        return

    outfile = open(outfile_name, "w")
    for i in xrange(len(label_predict)):
        outfile.write(uid_list[i] + " " + label_predict[i] + "\n")
    outfile.close()
