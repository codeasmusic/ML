from sklearn.model_selection import GridSearchCV
import time


def grid_search_clf(clf, params, cross_num, jobs):
    grid_search = GridSearchCV(clf, params, cv=cross_num, n_jobs=jobs, verbose=1)
    return grid_search


def print_best_params(grid_clf, parameters):
    outfile = open("result" + str(time.time()) + ".txt", "w")
    outfile.write("best score: " + str(grid_clf.best_score_) + "\n\n")

    best_params = grid_clf.best_params_
    for p in sorted(parameters.keys()):
        outfile.write(p + ": " + str(best_params[p]) + "\n")
    outfile.close()


def read_labels(train_file, label_index):
    infile = open(train_file)
    label_list = []

    for line in infile:
        line_parts = line.strip().split("\t")
        label_list.append(line_parts[label_index])
    infile.close()

    return label_list
