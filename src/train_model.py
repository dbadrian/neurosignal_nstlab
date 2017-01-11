#/usr/bin/python
# -*- coding: utf-8 -*-


import argparse

import matplotlib.pyplot as plt
import numpy as np

import nst_filter as nfilt
import nst_feature as nfeat
from scipy import signal

from sklearn.model_selection import KFold
from sklearn import svm, linear_model

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis



# Global Constants, cause fuck it
fs_Hz = 128.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn_data', '-d', type=str, default='',
                        help='List of words',
                        required=True)
    # parser.add_argument('--fd_out', '-o', type=str, default='',
    #                     help='List of words',
    #                     required=True)
    args = parser.parse_args()


# Load Measurements
raw_data = np.loadtxt(args.fn_data, delimiter=',')

# Split Measurements
# the by one error is neglible....
ch_1 = raw_data[:, 0 * 2 * 1492:1492]
ch_2 = raw_data[:, 1 * 2 * 1492: 3 * 1492]
ch_3 = raw_data[:, 2 * 2 * 1492: 5 * 1492]
ch_4 = raw_data[:, 3 * 2 * 1492: 7 * 1492]
ch_5 = raw_data[:, 4 * 2 * 1492: 9 * 1492]
ch_6 = raw_data[:, 5 * 2 * 1492: 11 * 1492]
label = raw_data[:, 6 * 2 * 1492: 6 * 2 * 1492 + 1]
# print label

raw_chs = [ch_1, ch_2, ch_3, ch_4, ch_5, ch_6]
chs = [nfilt.butter_bandpass_filter(ch, 49, 51, fs_Hz, 5, 'bandstop') for ch in raw_chs]
chs = [nfilt.butter_pass_filter(ch, 55, fs_Hz, 5)  for ch in chs]

ch_alphas = [nfeat.extract_alpha(ch, fs_Hz) for ch in chs]
ch_betas = [nfeat.extract_beta(ch, fs_Hz) for ch in raw_chs]

# Extracting Features
ch_feat = nfeat.compose_features(chs, sde=False, fs_Hz=fs_Hz, pw=True, mean=True, var=False)
ch_alphas_feat = nfeat.compose_features(ch_alphas, sde=False, fs_Hz=fs_Hz, pw=True, mean=True, var=False)
ch_betas_feat = nfeat.compose_features(ch_betas, sde=False, fs_Hz=fs_Hz, pw=True, mean=True, var=False)

fm = None
for idx, feature_set in enumerate([ch_feat, ch_alphas_feat, ch_betas_feat]):
	if fm is None:
		fm = np.hstack(tuple(feature_set))
	else:
		ft = np.hstack(tuple(feature_set))
		fm = np.hstack((fm, ft))

print fm.shape


# Train Classifier
kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(fm)
print(kf)

scores = []
for idx, (train_index, test_index) in enumerate(kf.split(fm)):
	print "Split:", idx
	# print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = fm[train_index], fm[test_index]
	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	y_train, y_test = label[train_index], label[test_index]
	# clf = svm.SVC(C=1000000, gamma=0.000001, verbose=False).fit(X_train, np.ravel(y_train))
	# clf = svm.LinearSVC(C=10000, verbose=False).fit(X_train, np.ravel(y_train))
	# clf = linear_model.SGDClassifier(loss='modified_huber', penalty='elasticnet', verbose=False).fit(X_train, np.ravel(y_train))
	# clf = LinearDiscriminantAnalysis()
	clf = QuadraticDiscriminantAnalysis()
	clf.fit(X_train, np.ravel(y_train))
	scores.append(clf.score(X_test, y_test))
	print scores[-1]

print "avg. score:", np.mean(scores)

# Test Classifier
#
lat_limit = 1 # in seconds
window_size = int(lat_limit * fs_Hz)
print "Window Size", window_size

window_cuts = []
for trial in range(0, fm.shape[0]):
	#label
	# take window over all channels
	cuts = []
	for ch in [ch_1, ch_2, ch_3, ch_4, ch_5, ch_6]:
		print len(ch[trial])
		minlim = int(len(ch[trial])%window_size)
		parts = int(len(ch[trial,:-minlim])/window_size)
		print minlim, parts
		cuts.append(np.split(ch[trial,:-minlim], parts))

	for cut in cuts:
		print len(cut)

	break

	# 	if blob is None:
	# 		blob = np.split(ch[trial,:-minlim], window_size)
	# 	else:
	# 		tmp = np.split(ch[trial,:-minlim], window_size)
	# 		blob = [np.vstack((subarr, tmp[idx])) for idx, subarr in enumerate(blob)]

	# print blob.shape
	# break
	# chs = [nfilt.butter_bandpass_filter(ch, 49, 51, fs_Hz, 5, 'bandstop') for ch in raw_chs]
	# chs = [nfilt.butter_pass_filter(ch, 55, fs_Hz, 5)  for ch in chs]

	# ch_alphas = [nfeat.extract_alpha(ch, fs_Hz) for ch in raw_chs]
	# ch_betas = [nfeat.extract_beta(ch, fs_Hz) for ch in raw_chs]

	# # classify featurewindow
	# # create mock up data
	# print chs[0][trial].shape



# Plot
# t = np.linspace(0, 1492, 1492)
# plt.figure
# plt.plot(t[0:100], ch_test[0:100], 'b', alpha=0.75)
# plt.plot(t[0:100], y[0:100], 'k')
# plt.legend(('noisy signal', 'filtfilt'), loc='best')
# plt.grid(True)
# plt.show()


# Filtering
# create the 50 Hz notch (bandpass) filter
# y1 = nfilt.butter_pass_filter(ch_1[0, :], 20, fs_Hz, 5)
# y = nfilt.butter_pass_filter(ch_1, 20, fs_Hz, 5, axis=1)


# # # y = nfilt.butter_bandpass_filter(y, 49, 51, fs_Hz, 5, 'bandstop')
# t = np.linspace(0, 1492, 1492)
# plt.figure
# # plt.plot(t[0:100], ch_1[0, 0:100], 'b', alpha=0.75)
# # plt.plot(t[0:100], y1[0:100], 'k')
# plt.plot(ch_pw[0]/max(ch_pw[0]), 'r')
# plt.plot(label, 'b')

# plt.legend(('noisy signal', 'filtfilt'), loc='best')
# plt.grid(True)
# plt.show()
