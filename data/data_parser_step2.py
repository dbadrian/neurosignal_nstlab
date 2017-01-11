#data preprocessing
import pandas as pd
import time
from datetime import datetime
from collections import OrderedDict
import os
import numpy as np
import csv
import itertools


def flattenListOfLists(lst):
    result = []
    for sublist in lst:
        result.extend(sublist)
    return result

path = "./"
data = pd.DataFrame.from_csv('training_data_raw.csv')

last_label = data.itertuples().next().Label
count = 0
cnts = []
for entry in data.itertuples():
    count += 1
    if entry.Label != last_label:
        last_label = entry.Label
        cnts.append(count)
        count = 0

minLimit = min(cnts)
trials = len(cnts)
print cnts
print "Min. Measurements: ", minLimit
print "Max. Measurements: ", max(cnts)
print "Trials: ", trials



# Okay, now only take the first N (min value) measurements
offset = 2
# blob = [[]] * 12
# measurement_matrix = []
measurement_matrix = np.zeros((trials+1, 12 * minLimit + 1))
last_label = data.itertuples().next().Label

limit_cnt = 0
c_trial = 0
for m, entry in enumerate(data.itertuples()):
    e = list(entry)

    if entry.Label != last_label:
        measurement_matrix[c_trial, -1] = 0 if (last_label == 'R') else 1

        last_label = entry.Label
        limit_cnt = 0
        c_trial += 1

    if limit_cnt < minLimit:
        for idx in range(0,12):
            measurement_matrix[c_trial, idx*minLimit + limit_cnt] = e[idx + offset]

    limit_cnt += 1


# print measurement_matrix[0][0][0:12]
# print len(measurement_matrix[0][0]) #[0:10]
# measurement_matrix = measurement_matrix.transpose()
np.savetxt("training_data_matrix.csv", measurement_matrix, delimiter=",")
# with open("training_data_matrix.csv", "wb") as f:
#     writer = csv.writer(f)
#     for entry in measurement_matrix:

#         # print list(itertools.chain(*entry))
#         # break
#         writer.writerow(flattenListOfLists(entry))
