import os
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# -- Augmentation functions --
def butter_highpass(cutoff_hz, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def highpass_butter(data, cutoff_hz, fs, order=1):
    b, a = butter_highpass(cutoff_hz, fs, order=order)
    return filtfilt(b, a, data)


def highpass_moving_average(data, win_size):
    kernel = np.ones(win_size) / win_size
    trend = np.convolve(data, kernel, mode='same')
    return data - trend


def augment_features(tr_events, cutoff_percentile=90, hp_cutoff=0.1, ma_window=12, smooth_sigma=1):
    # 1) Derivative (prepend first value)
    derivative = np.diff(tr_events, prepend=tr_events[0])

    # 2) Threshold-based continuous features
    threshold = np.percentile(tr_events, cutoff_percentile)
    thresh_cont = np.where(tr_events > threshold, tr_events, 0.0)
    thresh_cont2 = np.maximum(tr_events - threshold, 0.0)

    # 3) High-pass Butterworth filter (TR=1s)
    fs = 1.0
    hp_series = highpass_butter(tr_events, cutoff_hz=hp_cutoff, fs=fs)

    # 4) Moving-average detrend
    hp_ma = highpass_moving_average(tr_events, win_size=ma_window)

    # 5) Half-wave rectify derivative
    deriv_halfwave = np.clip(derivative, 0, None)

    # 6) Raw feature thresholding
    raw_th = np.clip(tr_events - threshold, 0, None)

    # 7) Smooth both continuous features
    smoothed_deriv_th = gaussian_filter1d(deriv_halfwave, smooth_sigma)
    smoothed_raw_th = gaussian_filter1d(raw_th, smooth_sigma) * 10

    return {
        'original': tr_events,
        'derivative': derivative,
        'thresh_cont': thresh_cont,
        'thresh_cont2': thresh_cont2,
        'hp_series': hp_series,
        'hp_ma': hp_ma,
        'deriv_halfwave': deriv_halfwave,
        'raw_th': raw_th,
        'smoothed_deriv_th': smoothed_deriv_th,
        'smoothed_raw_th': smoothed_raw_th
    }

# -- Batch processing settings --
BASE_DIR = 'data/ours/extracted_text_features'
FILENAME = 'final_outputs_eventboundary_log_prob_context_256_512.npz'

# -- Main script --
if __name__ == '__main__':
    for story in os.listdir(BASE_DIR):
        story_dir = os.path.join(BASE_DIR, story)
        if not os.path.isdir(story_dir):
            continue

        infile = os.path.join(story_dir, FILENAME)
        if not os.path.exists(infile):
            print(f"Skipping {story}: {FILENAME} not found")
            continue

        data = np.load(infile)
        if 'features' not in data:
            print(f"Skipping {infile}: no 'features' key")
            continue
        tr_events = data['features']

        augmented = augment_features(tr_events)
        # Save each augmented feature separately
        for key, arr in augmented.items():
            out_name = f"final_outputs_eventboundary_log_prob_{key}_context_256_512.npz"
            outfile = os.path.join(story_dir, out_name)
            # Save array under 'features' key for consistency
            np.savez_compressed(outfile, features=arr)
            print(f"Saved {key} -> {outfile}")
