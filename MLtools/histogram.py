# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']    # show chinese
plt.rcParams['axes.unicode_minus'] = False  # show negative sign


# data = [x_list, y_list]
# x_list: index start from 0, e.g. [0, 1, 2, ...]
#
# configs = {'title': "",
#            'xlabel': "",
#            'ylabel': "",
#            'xticks': ("group1", "group2", "group3", ...)}


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
        plt.text(float(v1) + bar_width/2, v2 + 0.5, "%d" % v2, ha="center", va="bottom")

    plt.title(configs['title'])
    plt.xlabel(configs['xlabel'])
    plt.ylabel(configs['ylabel'])
    plt.show()