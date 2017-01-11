from scipy.signal import butter, filtfilt


def butter_pass(cut_freq, fs, order=10, btype='highpass'):
    nyq = 0.5 * fs
    cf = cut_freq / nyq
    b, a = butter(order, cf, btype=btype)
    return b, a


def butter_pass_filter(data, cut_freq, fs, order=10, btype='highpass', axis=0):
    b, a = butter_pass(cut_freq, fs, order=order, btype=btype)
    y = filtfilt(b, a, data, axis=axis)
    return y


def butter_bandpass(lowcut, highcut, fs, order=10, btype='band'):
    nyq = 0.5 * fs
    # nyq = fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype=btype)
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=10, btype='band', axis=0):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    y = filtfilt(b, a, data, axis=axis)
    return y
