#!/usr/bin/env python
# masked_correlation_pipeline.py  –  ROI‑wise correlation summaries
# -----------------------------------------------------------------
# Outputs: summary_<metric>.csv  +  summary_all_metrics.csv
# Each row = metric × story × ROI with max/mean r and voxel counts.

import os, re, glob, pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn import image

# ─── 0. CONFIG ────────────────────────────────────────────────────
STORIES = [
    "maupassant_hand", "die_pflanzen_des_dr", "die_maske_des_roten_todes",
    "der_fall_stretelli", "koenig_pest", "der_katechismus_der_familie_musgrave",
    "lebendig_begraben", "fuenf_apfelsinenkerne", "der_blaue_karfunkel",
    "ligeia", "das_manuskript_in_der_flasche", "die_schwarze_katze",
    "mord_in_sunningdale",
]

MASK_IMG     = nib.load("../../../../DATA/BrainEncode/anatomical_files/binarized_mask.nii.gz")
masker       = NiftiMasker(mask_img=MASK_IMG, standardize="zscore_sample").fit()
REGION_MASKS = glob.glob(os.path.join("../../../../DATA/BrainEncode/MASKS", "*.nii*"))
THR = (0.1, 0.2, 0.3)                     # thresholds for voxel counts

# ─── 1. HELPERS ───────────────────────────────────────────────────
def metric_folders(root="../../../../DATA/BrainEncode/text_metrics_results/"):
    return [d for d in os.listdir(root)
            if d.startswith("text_") and "True" in d]

def metric_name(folder):
    m = re.match(r"text_(.*?)_chunklen40", folder)
    #print(m.group(1))
    return m.group(1) if m else folder

def recon_corr(zfile):
    with open(zfile, "rb") as f:          z = pickle.load(f)
    with open(zfile.replace("zscored_correlations_","null_corrs_"), "rb") as f:
        null = pickle.load(f)
    return z * null.std() + null.mean()

def write_csv_with_sep_hint(df, fname, sep=";"):
    """Write CSV preceded by 'sep=<sep>' so Excel auto‑detects the delimiter."""
    with open(fname, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"sep={sep}\n")
        df.to_csv(fh, index=False, sep=sep)

# ─── 2. MAIN ──────────────────────────────────────────────────────
all_rows = []

for folder in metric_folders():
    mname = metric_name(folder)
    print(f"⏩ Metric: {mname}")
    rows = []

    for story in STORIES:
        zfile = os.path.join("../../../../DATA/BrainEncode/text_metrics_results/", folder, f"zscored_correlations_story_{story}.pkl")
        print(zfile)
        if not os.path.exists(zfile):
            print(f"   • {story:<25} – missing z‑file")
            continue

        corr_vec  = recon_corr(zfile)
        print(corr_vec.shape)
        corr_img4 = masker.inverse_transform(corr_vec.reshape(1, -1))
        corr_img  = image.index_img(corr_img4, 0)          # 3‑D

        for mask_path in REGION_MASKS:
            roi_name = os.path.splitext(os.path.basename(mask_path))[0]
            roi_img  = nib.load(mask_path)
            if roi_img.shape != corr_img.shape:
                roi_img = image.resample_to_img(roi_img, corr_img, interpolation="nearest")

            roi_mask = roi_img.get_fdata().astype(bool)
            n_voxels = int(roi_mask.sum())
            data     = corr_img.get_fdata()[roi_mask]
            if data.size == 0:                # empty ROI
                continue

            row = dict(metric=mname, story=story, roi=roi_name,
                       max_corr=float(np.max(data)),
                       mean_corr=float(np.mean(data)), n_voxels = n_voxels)

            for t in THR:
                row[f"vox_gt_{t}"] = int(np.sum(data >= t))
            rows.append(row)

    # per‑metric CSV ------------------------------------------------
    if rows:
        df_m = pd.DataFrame(rows)
        write_csv_with_sep_hint(df_m, f"neg_summary_{mname}.csv")
        print(f"   ✔ summary_{mname}.csv  ({len(df_m)} rows)")
        all_rows.extend(rows)

# master CSV --------------------------------------------------------
if all_rows:
    df_all = pd.DataFrame(all_rows)
    write_csv_with_sep_hint(df_all, "pos_summary_all_metrics.csv")
    print(f"\n✅ summary_all_metrics.csv  (rows: {len(df_all)})")
else:
    print("\n⚠ No data written – no z‑score files found.")
    print(all_rows)
