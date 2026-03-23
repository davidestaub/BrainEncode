#!/usr/bin/env python3
import os
import numpy as np

# Base folder containing all the text‐feature subdirectories
BASE_DIR = 'data/ours/extracted_text_features'

# Filename pattern to look for
PATTERN = 'eventboundary_log_prob'
OUTPUT_NAME = 'final_outputs_ALL_STACKED_context_256_512.npz'

def stack_eventboundary_features(folder_path):
    print("we are in")
    """Load all matching .npz files in folder_path, stack their arrays side by side, and save."""
    # find all .npz files matching our pattern
    files = [f for f in os.listdir(folder_path)
             if f.endswith('.npz') and PATTERN in f and not 'STACKED' in f]

    if not files:
        print("nononononon",flush=True)
        return  # no matching files here, skip

    # sort for reproducible column order
    files.sort()
    print(files)

    files = ['final_outputs_eventboundary_log_prob_cs1_context_256_512.npz',
     'final_outputs_eventboundary_log_prob_deriv_halfwave_context_256_512.npz',
     'final_outputs_eventboundary_log_prob_hp_ma_context_256_512.npz',
     'final_outputs_eventboundary_log_prob_smoothed_raw_th_context_256_512.npz',
]

    arrays = []
    for fname in files:
        data = np.load(os.path.join(folder_path, fname))
        # assume each .npz contains exactly one array; extract it
        key = data.files[0]
        if key == 'features':
            print(key)
            arr = data[key]
            print(arr)
            # ensure 2D: if 1D, turn (N,) into (N,1)
            if arr.ndim == 1:
                arr = arr[:, None]
            arrays.append(arr)

    # verify all arrays have the same number of rows
    n_rows = arrays[0].shape[0]
    if any(a.shape[0] != n_rows for a in arrays):
        raise ValueError(f"Dimension mismatch in {folder_path}: "
                         f"row counts are {[a.shape[0] for a in arrays]}")

    # horizontally concatenate into shape (n_rows, total_columns)
    stacked = np.hstack(arrays)
    print(stacked.shape)

    # save
    out_path = os.path.join(folder_path, OUTPUT_NAME)
    np.savez(out_path, features=stacked)
    print(f"Saved stacked feature to: {out_path}")



if __name__ == "__main__":
    print("Hello",flush=True)
    print(str(os.walk(BASE_DIR)))
    for root, dirs, files in os.walk(BASE_DIR):
        # only process leaf directories (i.e. skip the BASE_DIR itself)
        if root == BASE_DIR:
            print("That is not good",flush=True)
            continue
        stack_eventboundary_features(root)