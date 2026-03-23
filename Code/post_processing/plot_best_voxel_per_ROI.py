#!/usr/bin/env python
# roi_max_voxel_index_pick_storymax.py
# -----------------------------------------------------------------------------
# For each metric × ROI:
#   1) Iterate over stories; within ROI∩mask find voxel of highest corr (max_corr).
#   2) Keep the single story+voxel that yields the overall highest max_corr.
#   3) Write CSV: metric, roi, masked_idx, story, max_corr
#   4) Plot the winning voxel time series from:
#        full_response_<STORY>.pkl  (solid grey)
#        full_prediction_<STORY>.pkl (dashed red; ×10 if much smaller range)
#      Title includes ROI, voxel index, max_corr, story.
#
# Files are expected in each metric folder (same place as zscored_correlations):
#   zscored_correlations_story_<story>.pkl
#   null_corrs_story_<story>.pkl
#   full_response_<story>.pkl
#   full_prediction_<story>.pkl
#
# Output:
#   maxidx_pickstory_summary_<metric>.csv
#   maxidx_pickstory_summary_all_metrics.csv
#   plots/<metric>/plot_<metric>__<roi>.png

import os, re, glob, pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn import image
from nilearn.masking import apply_mask
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# --- 0. CONFIG ----------------------------------------------------
STORIES = [
    "maupassant_hand", "die_pflanzen_des_dr", "die_maske_des_roten_todes",
    "der_fall_stretelli", "koenig_pest", "der_katechismus_der_familie_musgrave",
    "lebendig_begraben", "fuenf_apfelsinenkerne", "der_blaue_karfunkel",
    "ligeia", "das_manuskript_in_der_flasche", "die_schwarze_katze",
    "mord_in_sunningdale",
]

MASK_IMG     = nib.load("../../../../DATA/BrainEncode/anatomical_files/binarized_mask.nii.gz")
masker       = NiftiMasker(mask_img=MASK_IMG, standardize=None).fit()
REGION_MASKS = glob.glob(os.path.join("brain_masks", "*.nii*"))

# auto-amp threshold: if pred range < THR * resp range → scale pred by 10
PRED_SMALL_RANGE_THR = 0.2
PRED_SCALE_FACTOR    = 10.0

# --- 1. HELPERS ---------------------------------------------------
def metric_folders(root="../../../../DATA/BrainEncode/text_metrics_results/"):
    return [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d.startswith("text_") and "True" in d]

def metric_name(folder):
    m = re.match(r"text_(.*?)_chunklen40", folder)
    return m.group(1) if m else folder

def recon_corr(zfile):
    with open(zfile, "rb") as f:
        z = pickle.load(f)
    with open(zfile.replace("zscored_correlations_","null_corrs_"), "rb") as f:
        null = pickle.load(f)
    return z * null.std() + null.mean()

def write_csv_with_sep_hint(df, fname, sep=";"):
    with open(fname, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"sep={sep}\n")
        df.to_csv(fh, index=False, sep=sep)

def align_to_corr_space(img, corr_img):
    if img.shape != corr_img.shape or not np.allclose(img.affine, corr_img.affine):
        return image.resample_to_img(img, corr_img, interpolation="nearest")
    return img

# map (i,j,k) -> masked 1D index (same ordering as masker/applied mask)
mask_bool  = MASK_IMG.get_fdata().astype(bool)
n_mask_vox = int(mask_bool.sum())

label_data = np.zeros(mask_bool.shape, dtype=np.int32)
label_data[mask_bool] = np.arange(n_mask_vox, dtype=np.int32)
label_img  = image.new_img_like(MASK_IMG, label_data)

p_to_j = apply_mask(label_img, MASK_IMG).astype(np.int64)  # vector pos p -> j label
j_to_p = np.empty(n_mask_vox, dtype=np.int64)
j_to_p[p_to_j] = np.arange(n_mask_vox, dtype=np.int64)

def ijk_to_masked_index(ijk_tuple):
    j = int(label_data[ijk_tuple])
    return int(j_to_p[j])

def load_voxel_timeseries(pkl_path, masked_idx, n_mask_vox):
    """Return 1D array (T,) for the given masked voxel index.
       Supports arrays shaped (V,T) or (T,V)."""
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    arr = np.array(obj)
    if arr.ndim == 2:
        if arr.shape[0] == n_mask_vox:
            ts = arr[masked_idx, :]
        elif arr.shape[1] == n_mask_vox:
            ts = arr[:, masked_idx]
        else:
            raise ValueError(f"{os.path.basename(pkl_path)} has shape {arr.shape}, "
                             f"neither dim equals #mask voxels ({n_mask_vox}).")
        return np.asarray(ts).astype(float)
    raise ValueError(f"{os.path.basename(pkl_path)} not understood (ndim={arr.ndim}); "
                     "expecting a 2D array with one dimension = #mask voxels.")

def maybe_scale_prediction(pred, resp, thr=PRED_SMALL_RANGE_THR, factor=PRED_SCALE_FACTOR):
    r_pred = np.ptp(pred)  # max-min
    r_resp = np.ptp(resp)
    if r_resp <= 0:
        return pred, False
    if r_pred < thr * r_resp:
        return pred * factor, True
    return pred, False

def save_plot(out_png, resp_ts, pred_ts, voxel_idx, roi_name, max_corr, story):
    plt.figure(figsize=(10, 4))
    # response in solid grey; prediction dashed red
    plt.plot(resp_ts, color="grey", linewidth=0.8, label="Response")
    plt.plot(pred_ts, color="red", linewidth=0.8, label="Prediction")
    plt.xlabel("Time")
    plt.ylabel("Signal")
    title = f"{roi_name} | idx={voxel_idx} | max_corr={max_corr:.3f} | story={story}"
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()

# --- 2. MAIN ------------------------------------------------------
all_rows = []

for folder in metric_folders():
    mname = metric_name(folder)
    print(f">> Metric: {mname}")

    # ensure plot dir
    plot_dir = os.path.join("plots", mname)
    os.makedirs(plot_dir, exist_ok=True)

    corr_template_img = None
    best_per_roi = {}  # roi_name -> dict(masked_idx, story, max_corr)

    # 2.1 pick best story+voxel per ROI
    for story in STORIES:
        zfile = os.path.join(folder, f"zscored_correlations_story_{story}.pkl")
        print("zfile:",zfile)
        if not os.path.exists(zfile):
            print(f"   - {story:<25} : missing z-file, skipping")
            continue

        corr_vec = recon_corr(zfile)  # (V,)

        if corr_template_img is None:
            corr_img4 = masker.inverse_transform(corr_vec.reshape(1, -1))
            corr_template_img = image.index_img(corr_img4, 0)

        corr_img4 = masker.inverse_transform(corr_vec.reshape(1, -1))
        corr_img  = image.index_img(corr_img4, 0)
        data3d    = corr_img.get_fdata()

        for mask_path in REGION_MASKS:
            if not 'binarized_mask.nii.gz' in mask_path:

                roi_name = os.path.splitext(os.path.basename(mask_path))[0]
                roi_img  = nib.load(mask_path)
                roi_img  = align_to_corr_space(roi_img, corr_template_img)

                roi_mask   = roi_img.get_fdata().astype(bool)
                inter_mask = roi_mask & mask_bool
                if not inter_mask.any():
                    continue

                roi_vals = data3d[inter_mask]
                imax     = int(np.argmax(roi_vals))
                max_corr = float(roi_vals[imax])

                if (roi_name not in best_per_roi) or (max_corr > best_per_roi[roi_name]["max_corr"]):
                    inter_coords = np.column_stack(np.where(inter_mask))
                    ijk = tuple(int(x) for x in inter_coords[imax])
                    masked_idx = ijk_to_masked_index(ijk)

                    best_per_roi[roi_name] = dict(
                        masked_idx=int(masked_idx),
                        story=story,
                        max_corr=max_corr,
                    )

    # 2.2 write CSV rows and make plots
    rows = []
    for roi_name, info in best_per_roi.items():
        masked_idx = info["masked_idx"]
        story      = info["story"]
        max_corr   = info["max_corr"]

        # load full response/prediction for that story from the SAME metric folder
        pred_pkl = os.path.join(folder, f"full_prediction_{story}.pkl")
        resp_pkl = os.path.join(folder, f"full_response_{story}.pkl")

        try:
            resp_ts = load_voxel_timeseries(resp_pkl, masked_idx, n_mask_vox)
            pred_ts = load_voxel_timeseries(pred_pkl, masked_idx, n_mask_vox)
            if resp_ts is not None and pred_ts is not None:
                pred_plot, scaled = maybe_scale_prediction(pred_ts, resp_ts)
                if scaled:
                    # annotate in legend by modifying the label in save_plot call if desired;
                    # for simplicity, add suffix to title
                    pass
                out_png = os.path.join(plot_dir, f"plot_{mname}__{roi_name}.png")
                save_plot(out_png, resp_ts, pred_plot, masked_idx, roi_name, max_corr, story)
        except Exception as e:
            print(f"   ! Plotting failed for {mname} / {roi_name} / {story}: {e}")

        rows.append(dict(
            metric=mname,
            roi=roi_name,
            masked_idx=masked_idx,
            story=story,
            max_corr=max_corr
        ))

    if rows:
        df_m = pd.DataFrame(rows).sort_values(["roi"]).reset_index(drop=True)
        outname = f"{mname}maxidx_pickstory_summary.csv"
        write_csv_with_sep_hint(df_m, outname)
        print(f"   OK: {outname}  ({len(df_m)} rows)")
        all_rows.extend(rows)
    else:
        print("   (No ROI rows written for this metric)")

# Combined CSV
if all_rows:
    df_all = pd.DataFrame(all_rows).sort_values(["metric","roi"]).reset_index(drop=True)
    write_csv_with_sep_hint(df_all, "maxidx_pickstory_summary_all_metrics.csv")
    print(f"\nALL OK: maxidx_pickstory_summary_all_metrics.csv  (rows: {len(df_all)})")
else:
    print("\nNothing written: no z-score files found or no ROI overlaps.")
