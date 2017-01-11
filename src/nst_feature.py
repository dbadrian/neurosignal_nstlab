from scipy import signal
import numpy as np

import nst_filter as nf


def extract_alpha(data, fs_Hz):
    return nf.butter_bandpass_filter(data, 8, 12, fs_Hz, 5, 'band', axis=1)


def extract_beta(data, fs_Hz):
    return nf.butter_bandpass_filter(data, 13, 30, fs_Hz, 5, 'band', axis=1)


def compose_features(channels, sde=True, fs_Hz=0, pw=True, mean=True, var=True):
    feature_set = []

    # # power/sde
    if sde:
        ch_sde = [signal.welch(ch, fs_Hz, nperseg=4096, axis=1) for ch in channels]
        ch_mean_sde = [np.mean(ch[1], axis=1) for ch in ch_sde]
        feature_set.append(ch_mean_sde)

    if pw:
        ch_pw = [np.sqrt(np.mean(np.power(ch, 2), axis=1)) for ch in channels]
        feature_set.append(ch_pw)

    # # mean
    if mean:
        ch_mean = [np.mean(ch, axis=1) for ch in channels]
        feature_set.append(ch_mean)

    ## var
    if var:
        ch_var = [np.var(ch, axis=1) for ch in channels]
        feature_set.append(ch_var)

    composed_feature = []
    for idx, chan in enumerate(channels):
        feat = None

        for feature in feature_set:
            if feat is None:
                feat = feature[idx]
            else:
                feat = np.vstack((feat, feature[idx]))

        feat = feat.transpose()

        composed_feature.append(feat)

    return composed_feature