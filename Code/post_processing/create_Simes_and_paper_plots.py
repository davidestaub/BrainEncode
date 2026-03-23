#!/usr/bin/env python
# roi_fdr_all_stories.py  —  FIXED Simes across voxels (per story × ROI)
# ------------------------------------------------------------
# Processes *every story* in:
#   text_metrics_results/<in_folder>/
#       zscored_correlations_story_<story>.pkl
#       null_corrs_story_<story>.pkl
#
# Stats per story:
# - Reconstruct voxelwise r from z + null
# - Empirical two-tailed p per voxel (+(1)/(N+1)), vectorized
# - Voxelwise whole-brain FDR (BH)
# - Within-ROI voxel FDR (BH)
# - ROI-level omnibus p via Simes (over voxel p's in the ROI)  [FIXED]
# - FWE across ROIs (Holm–Bonferroni + Bonferroni)
# - Magnitude & sign summaries: mean_abs_r, median_abs_r, n_sig_pos/neg
#
# Plots per story (toggles):
# - Unthresholded anatomical mosaic
# - Thresholded anatomical mosaic (|r| ≥ THRESH)
# - Per-ROI r-map, violin plots (optional)
# - Whole-brain violin of r (optional)
#
# Across stories:
# - ROI Simes count heatmap (paper-ready)
# - Across-stories combined p (Simes across stories) + BH-FDR over voxels
# - Mean r masked by across-stories significance
# - Consistency map: #stories where voxel was WB-FDR significant
# - Global violin pooling all voxels across all stories (optional)
#
# Outputs:
#   text_metrics_results/<in_folder>/<story>/[...]  per-story
#   text_metrics_results/<in_folder>/out/summary_all_stories__roi_family.csv
#   text_metrics_results/<in_folder>/out/roi_simes_count_map.{png,pdf}
#   ... plus across-stories NIfTI/PDFs if enabled
# ------------------------------------------------------------

import os
import glob
import pickle
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn import image, plotting
from nilearn.masking import apply_mask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#p#lt.style.use('dark_background')

from statsmodels.stats.multitest import multipletests

# ---------------- CONFIG ----------------
HERE = os.path.abspath(os.path.dirname(__file__))

# Path to the results folder containing zscored/null pkls for this metric/run
in_folder = '../../../../DATA/BrainEncode/text_metrics_results/text_punctuation_chunklen40_alphas_-6to0_50_nboots_20_use_corr_True_MEAN_HIP_R'

in_folder = '../../../../DATA/BrainEncode/text_metrics_results/text_context_drift_DETREND_STABILIZE0_RHO03_USE_PCA0_layer32_chunklen40_alphas_-6to0_50_nboots_100_use_corr_True_MEAN_HIP_R'

try:
    RHO= in_folder.split('RHO')[1].split('_')[0]
    print(RHO)
except Exception as e:
    RHO = ''


# Brain mask + ROI masks (keep these in the same space)
MASK_IMG_PATH     = os.path.join(HERE, "../../../../DATA/BrainEncode/anatomical_files", "binarized_mask.nii.gz")
REGION_MASKS_GLOB = os.path.join(HERE, "../../../../DATA/BrainEncode/MASKS", "*.nii*")

# Background (optional)
T1_NII        = os.path.join(HERE, "../../../../DATA/BrainEncode/anatomical_files", "Robert_T1_brain_in_func_space_space.nii.gz")
T1_BRAIN_NII  = os.path.join(HERE, "../../../../DATA/BrainEncode/anatomical_files", "Robert_T1_brain_in_func_space_space.nii.gz")

# Plot scale and threshold
VMIN, VMAX = -0.3, 0.3
THRESH = 0.10

# ---- Plot toggles ----
DO_PLOTS_PER_STORY        = True
DO_PER_ROI_ANAT_PLOTS     = False
DO_PER_ROI_VIOLIN_PLOTS   = False
DO_WHOLE_BRAIN_VIOLIN     = True
DO_PLOT_ROI_COUNTS        = True
DO_PLOT_ACROSS_STORIES    = True
DO_GLOBAL_ALL_VOX_VIOLIN  = True

# Across-stories settings
ACROSS_STORIES_ALPHA   = 0.05  # BH-FDR alpha across voxels on combined p

# Paper style
APPLY_PAPER_STYLE = True
SAVE_PDF_FIGS     = True

if APPLY_PAPER_STYLE:
    matplotlib.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# ---------------- helpers ----------------
def write_csv_with_sep_hint(df, fname, sep=";"):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"sep={sep}\n")
        df.to_csv(fh, index=False, sep=sep)

def align_to_corr_space(img, corr_img):
    if img.shape != corr_img.shape or not np.allclose(img.affine, corr_img.affine):
        return image.resample_to_img(img, corr_img, interpolation="nearest")
    return img

def safe_load_t1(path):
    if os.path.isfile(path):
        try:
            return nib.load(path)
        except Exception:
            pass
    return None

def discover_stories(folder):
    paths = glob.glob(os.path.join(folder, "zscored_correlations_story_*.pkl"))
    print("paths =", paths)
    stories = []
    for p in paths:
        print(p)
        base = os.path.basename(p)
        story = base.split("zscored_correlations_story_")[1].split(".pkl")[0]
        if os.path.isfile(os.path.join(folder, f"null_corrs_story_{story}.pkl")):
            stories.append(story)
    return sorted(set(stories))

# ---------- Simes helpers ----------
def simes_1d(pvals):
    """
    Simes across a 1-D vector of p-values (e.g., all voxels in an ROI for one story).
    """
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    m = p.size
    if m == 0:
        return np.nan
    ps = np.sort(p)
    i  = np.arange(1, m+1, dtype=float)
    return float(np.clip(np.min((m/i)*ps), 0.0, 1.0))

def simes_across_stories(p_stack):
    """
    Simes across the first axis (stories) for each voxel column.
    p_stack: shape (n_stories, n_vox)
    returns p_combined per voxel, shape (n_vox,)
    """
    p = np.asarray(p_stack, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    m = p.shape[0]
    if m == 0:
        return np.full(p.shape[1], np.nan, dtype=float)
    ps = np.sort(p, axis=0)
    i = np.arange(1, m+1, dtype=float)[:, None]
    simes = np.min((m / i) * ps, axis=0)
    return np.clip(simes, 0.0, 1.0)

def abs_threshold_img(img, thr):
    data = np.asarray(img.get_fdata(), dtype=float)
    data[np.abs(data) < thr] = 0.0
    return nib.Nifti1Image(data, img.affine, img.header)

def save_violin(values, out_base, title=None, ylim=(-0.3, 0.3),
                dpi=600, save_pdf=True):
    values = np.asarray(values, dtype=float)
    fig = plt.figure(figsize=(3.2, 3.2))
    try:
        vp = plt.violinplot(values, showmeans=True, showextrema=False,
                            quantiles=[0.25, 0.5, 0.75])
    except TypeError:
        vp = plt.violinplot(values, showmeans=True, showextrema=False)
        for q in [0.25, 0.5, 0.75]:
            y = np.quantile(values, q)
            plt.hlines(y, 0.85, 1.15, linewidth=0.8)
    for b in vp['bodies']:
        b.set_edgecolor('black'); b.set_linewidth(0.5)
    plt.axhline(0, lw=0.5, ls="--")
    plt.ylim(*ylim)
    plt.xticks([])
    if title:
        plt.title(title)
    plt.ylabel("voxelwise r")
    plt.tight_layout()
    fig.savefig(out_base + ".png", dpi=dpi)
    if save_pdf:
        fig.savefig(out_base + ".pdf")
    plt.close(fig)

def build_roi_simes_count_map(df_master, mask_img_path, roi_glob, t1_path, out_dir):
    """
    Create a whole-brain NIfTI & PNG where each ROI is filled with the
    number of stories in which it had Simes p<.05, rendered over a T1.
    """
    import os, glob
    import numpy as np
    import nibabel as nib
    from nilearn import image, plotting

    os.makedirs(out_dir, exist_ok=True)

    # --- 1) Count Simes<.05 per ROI ---------------------------------
    df = df_master.copy()
    df["sig_simes"] = df["p_simes"] < 0.05
    roi_counts = (df.groupby("roi", as_index=False)
                    .agg(n_stories_sig=("sig_simes","sum")))
    n_stories_total = df["story"].nunique()
    roi_counts["fraction"] = roi_counts["n_stories_sig"] / float(n_stories_total)
    roi_counts = roi_counts.sort_values(["n_stories_sig","roi"], ascending=[False,True])
    roi_counts.to_csv(os.path.join(out_dir, f"roi_simes_counts{RHO}.csv"), index=False)

    # --- 2) Fill counts into a volume in mask space ------------------
    mask_img = nib.load(mask_img_path)
    mask_data = mask_img.get_fdata().astype(bool)
    count_data = np.zeros(mask_img.shape, dtype=np.float32)

    for roi_path in sorted(glob.glob(roi_glob)):
        roi_name = os.path.splitext(os.path.basename(roi_path))[0]
        count = int(roi_counts.loc[roi_counts["roi"] == roi_name, "n_stories_sig"].values[0]) \
                if (roi_counts["roi"] == roi_name).any() else 0
        try:
            roi_img = nib.load(roi_path)
            if (roi_img.shape != mask_img.shape) or (not np.allclose(roi_img.affine, mask_img.affine)):
                roi_img = image.resample_to_img(roi_img, mask_img, interpolation="nearest")
            roi_mask = roi_img.get_fdata().astype(bool)
            count_data[roi_mask] = np.maximum(count_data[roi_mask], count)
        except Exception as e:
            print(f"[roi_simes_count_map] skipping {roi_name}: {e}")

    # Hide overlay outside the brain so T1 is visible there
    outside = ~mask_data
    count_data[outside] = np.nan

    count_img = nib.Nifti1Image(count_data, mask_img.affine, mask_img.header)
    nii_path = os.path.join(out_dir, "roi_simes_count_volume.nii.gz")
    nib.save(count_img, nii_path)

    # --- 3) Pretty plotting over T1 ---------------------------------
    t1_img = nib.load(t1_path) if (t1_path and os.path.isfile(t1_path)) else None
    vmax = int(np.nanmax(count_data)) if np.isfinite(count_data).any() else 1
    vmax = max(1, vmax)

    disp = plotting.plot_stat_map(
        count_img,
        bg_img=t1_img,
        display_mode="mosaic",
        cmap="magma",                 # discrete-friendly, high contrast
        black_bg=True,                # black background (as requested)                   # keep anatomy visible (lightly dim)
        vmin=0, vmax=vmax,
       # threshold=0,                  # show all counts >= 0
        interpolation="nearest",      # crisp ROI edges
        annotate=False,
        symmetric_cbar=False,
        figure=plt.figure(figsize=(10, 8))
    )
    #bg_img = t1_img,
   # title = None, display_mode = "mosaic", cmap = "magma",
  #  black_bg = True, symmetric_cbar = False,
  #  vmin = -0.1, vmax = vmax, figure = plt.figure(figsize=(10, 8)),
    # Label the colorbar
    try:
        disp._cbar_ax.set_ylabel("Stories with Simes p < 0.05", rotation=90, va="center")
    except Exception:
        pass

    png_path = os.path.join(out_dir, "roi_simes_count_map.png")
    disp.savefig(png_path, dpi=600)
    if SAVE_PDF_FIGS:
        disp.savefig(os.path.join(out_dir, "roi_simes_count_map.pdf"))
    plt.close(disp.frame_axes.figure)

    print(f"[ROI COUNT MAP] wrote {nii_path} and {png_path}")


# ---------- Across-stories voxelwise combination & plots ----------
def build_across_stories_maps(corr_vecs, emp_ps, wb_rej, mask_img_path, t1_path, out_dir,
                              alpha=0.05):
    if not corr_vecs:
        return

    os.makedirs(out_dir, exist_ok=True)
    mask_img = nib.load(mask_img_path)
    masker = NiftiMasker(mask_img=mask_img, standardize=None).fit()

    R = np.vstack(corr_vecs)
    P = np.clip(np.vstack(emp_ps), 0.0, 1.0)
    S = R.shape[0]
    assert P.shape == R.shape

    mean_r = np.nanmean(R, axis=0)
    mean_abs_r = np.nanmean(np.abs(R), axis=0)
    p_simes = simes_across_stories(P)  # correct: across the stories axis
    rej_comb, p_fdr_comb, _, _ = multipletests(p_simes, alpha=alpha, method="fdr_bh")

    if wb_rej:
        WB = np.vstack(wb_rej).astype(int)
        sig_count = WB.sum(axis=0)
    else:
        sig_count = np.zeros_like(mean_r, dtype=int)

    mean_r_img          = image.index_img(masker.inverse_transform(mean_r.reshape(1, -1)), 0)
    mean_abs_r_img      = image.index_img(masker.inverse_transform(mean_abs_r.reshape(1, -1)), 0)
    p_simes_img         = image.index_img(masker.inverse_transform(p_simes.reshape(1, -1)), 0)
    mean_r_masked_vec   = mean_r.copy(); mean_r_masked_vec[~rej_comb] = 0.0
    mean_r_masked_img   = image.index_img(masker.inverse_transform(mean_r_masked_vec.reshape(1, -1)), 0)
    sig_count_img       = image.index_img(masker.inverse_transform(sig_count.reshape(1, -1)), 0)

    nib.save(mean_r_img,        os.path.join(out_dir, "across_mean_r.nii.gz"))
    nib.save(mean_abs_r_img,    os.path.join(out_dir, "across_mean_abs_r.nii.gz"))
    nib.save(mean_r_masked_img, os.path.join(out_dir, "across_mean_r_masked_FDR.nii.gz"))
    nib.save(p_simes_img,       os.path.join(out_dir, "across_combined_p_simes.nii.gz"))
    nib.save(sig_count_img,     os.path.join(out_dir, "across_sigcount.nii.gz"))

    t1_img = safe_load_t1(t1_path)

    # UNTHRESH & THRESH mosaics for across-stories mean r (paper-ready)
    dispA = plotting.plot_stat_map(
        mean_r_img, bg_img=t1_img,
        title=None, display_mode="mosaic", cmap="cold_hot",
        black_bg=True, symmetric_cbar=True,
        vmin=VMIN, vmax=VMAX, figure=plt.figure(figsize=(10, 8)),
    )
    figA = dispA.frame_axes.figure
    figA.savefig(os.path.join(out_dir, "across_mean_r_mosaic.png"), dpi=600)
    if SAVE_PDF_FIGS: figA.savefig(os.path.join(out_dir, "across_mean_r_mosaic.pdf"))
    plt.close(figA)

    mean_r_thr_img = abs_threshold_img(mean_r_img, THRESH)
    dispB = plotting.plot_stat_map(
        mean_r_thr_img, bg_img=t1_img,
        title=None, display_mode="mosaic", cmap="magma",
        black_bg=True, symmetric_cbar=False,
        vmin=VMIN, vmax=VMAX, figure=plt.figure(figsize=(10, 8)),
    )
    figB = dispB.frame_axes.figure
    figB.savefig(os.path.join(out_dir, "across_mean_r_mosaic_thr.png"), dpi=600)
    if SAVE_PDF_FIGS: figB.savefig(os.path.join(out_dir, "across_mean_r_mosaic_thr.pdf"))
    plt.close(figB)

    # Mean r masked by across-stories FDR
    dispC = plotting.plot_stat_map(
        mean_r_masked_img, bg_img=t1_img,
        title=None, display_mode="mosaic", cmap="magma",
        black_bg=True, symmetric_cbar=False,
        vmin=-0.1, figure=plt.figure(figsize=(10, 8)),
    )
    figC = dispC.frame_axes.figure
    figC.savefig(os.path.join(out_dir, "across_mean_r_masked_mosaic.png"), dpi=600)
    if SAVE_PDF_FIGS: figC.savefig(os.path.join(out_dir, "across_mean_r_masked_mosaic.pdf"))
    plt.close(figC)

    # Consistency map
    vmax = max(1, int(np.max(sig_count)))
    dispD = plotting.plot_stat_map(
        sig_count_img, bg_img=t1_img,
        title=None, display_mode="mosaic", cmap="magma",
        black_bg=True, symmetric_cbar=False,
        vmin=-0.1, vmax=vmax, figure=plt.figure(figsize=(10, 8)),
    )
    try:
        dispD._cbar_ax.set_ylabel(f"# stories WB-FDR < 0.05 (n={S})", rotation=90, va="center")
    except Exception:
        pass
    figD = dispD.frame_axes.figure
    figD.savefig(os.path.join(out_dir, "across_sigcount_mosaic.png"), dpi=600)
    if SAVE_PDF_FIGS: figD.savefig(os.path.join(out_dir, "across_sigcount_mosaic.pdf"))
    plt.close(figD)

# ---------------- globals for across-stories aggregation ----------------
MASTER_ROWS = []          # ROI family rows across stories
GLOBAL_CORR_VECS = []     # per-story corr_vec (for across-stories mean)
GLOBAL_EMP_P     = []     # per-story emp_p (for across-stories Simes)
GLOBAL_WB_REJ    = []     # per-story whole-brain BH rejections (for consistency map)
GLOBAL_ALL_R     = []     # all voxels from all stories pooled (for global violin)

# ---------------- per-story pipeline ----------------
def process_story(story):
    print(f"\n=== STORY: {story} ===")

    story_dir = os.path.join(HERE, in_folder)
    ZFILE   = os.path.join(story_dir, f"zscored_correlations_story_{story}.pkl")
    NFILE   = os.path.join(story_dir, f"null_corrs_story_{story}.pkl")
    OUT_DIR = os.path.join(story_dir, story)
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(ZFILE, "rb") as f:
        zscores = pickle.load(f)
    with open(NFILE, "rb") as f:
        null_flat = pickle.load(f)

    MASK_IMG  = nib.load(MASK_IMG_PATH)
    mask_bool = MASK_IMG.get_fdata().astype(bool)
    n_vox     = int(mask_bool.sum())

    # Reconstruct raw correlations (r) from z + null (global mean/std)
    null_mean = float(np.mean(null_flat))
    null_std  = float(np.std(null_flat))
    corr_vec  = np.asarray(zscores * null_std + null_mean, dtype=float).ravel()

    if corr_vec.shape[0] != n_vox:
        raise ValueError(f"{story}: corr_vec has {corr_vec.shape[0]} vox, mask has {n_vox}.")

    masker            = NiftiMasker(mask_img=MASK_IMG, standardize=None).fit()
    corr_img4         = masker.inverse_transform(corr_vec.reshape(1, -1))
    corr_template_img = image.index_img(corr_img4, 0)

    # --- PER-STORY ANATOMICAL PLOTS (optional) ---
    if DO_PLOTS_PER_STORY:
        t1_img = safe_load_t1(T1_BRAIN_NII) or safe_load_t1(T1_NII)

        # Unthresholded
        disp = plotting.plot_stat_map(
            corr_template_img, bg_img=t1_img,
            title=None, display_mode="mosaic", cmap="cold_hot",
            black_bg=True, symmetric_cbar=True,
            vmin=VMIN, vmax=VMAX, figure=plt.figure(figsize=(10, 8)),
        )
        disp.savefig(os.path.join(OUT_DIR, f"corr_map_mosaic__{story}.png"), dpi=300)
        plt.close(disp.frame_axes.figure)

        # Thresholded (|r| ≥ THRESH)
        thr_vec  = corr_vec.copy()
        thr_vec[np.abs(thr_vec) < THRESH] = 0.0
        thr_img4 = masker.inverse_transform(thr_vec.reshape(1, -1))
        thr_img  = image.index_img(thr_img4, 0)
        disp_thr = plotting.plot_stat_map(
            thr_img, bg_img=t1_img,
            title=None, display_mode="mosaic", cmap="cold_hot",
            black_bg=True, symmetric_cbar=True,
            vmin=VMIN, vmax=VMAX, figure=plt.figure(figsize=(10, 8)),
        )
        disp_thr.savefig(os.path.join(OUT_DIR, f"corr_map_mosaic_thr__{story}.png"), dpi=300)
        plt.close(disp_thr.frame_axes.figure)

    # --- Stats using null ---
    if null_flat.ndim != 1 or (null_flat.size % n_vox) != 0:
        raise ValueError(f"{story}: null_corrs shape {null_flat.shape} not divisible by #vox={n_vox}.")
    n_perms = null_flat.size // n_vox

    null_mat = null_flat.reshape(n_vox, n_perms)  # (vox, perm)
    emp_p    = ( (np.abs(null_mat) >= np.abs(corr_vec)[:, None]).sum(axis=1) + 1 ) / (n_perms + 1)

    rej_wb, p_fdr_wb, _, _ = multipletests(emp_p, alpha=0.05, method="fdr_bh")

    # Save per-voxel stats
    wb_df = pd.DataFrame({
        "masked_idx": np.arange(n_vox, dtype=int),
        "r": corr_vec,
        "emp_p": emp_p,
        "sig_FDR_wholebrain": rej_wb.astype(bool),
    })
    write_csv_with_sep_hint(wb_df, os.path.join(OUT_DIR, f"wb_voxel_stats__{story}.csv"))
    nib.save(corr_template_img, os.path.join(OUT_DIR, f"corr_volume__{story}.nii.gz"))

    if DO_PLOTS_PER_STORY and DO_WHOLE_BRAIN_VIOLIN:
        # Global violin (per story)
        fig = plt.figure(figsize=(3.2, 3.2))
        plt.violinplot(corr_vec, showmeans=True, showextrema=False)
        plt.axhline(0, lw=0.5, ls="--"); plt.ylim(VMIN, VMAX); plt.xticks([])
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"violin__WB__{story}.png"), dpi=600)
        if SAVE_PDF_FIGS: fig.savefig(os.path.join(OUT_DIR, f"violin__WB__{story}.pdf"))
        plt.close(fig)

    # ROI loop — first pass
    roi_paths = [
        p for p in glob.glob(REGION_MASKS_GLOB)
        if os.path.basename(p) != os.path.basename(MASK_IMG_PATH)
    ]
    roi_summary_rows = []
    roi_family_rows  = []

    # vector (masked order) boolean for intersections
    mask_vec_bool = apply_mask(image.new_img_like(MASK_IMG, mask_bool.astype(int)), MASK_IMG).astype(bool)

    tmp_roi_names, tmp_roi_p_simes, tmp_roi_sizes = [], [], []
    per_roi_cache = {}

    for mask_path in sorted(roi_paths):
        roi_name = os.path.splitext(os.path.basename(mask_path))[0]
        try:
            roi_img0 = nib.load(mask_path)
            roi_img  = align_to_corr_space(roi_img0, corr_template_img)
        except Exception as e:
            print(f"[ROI {roi_name}] resample failed: {e}")
            continue

        roi_mask_3d   = roi_img.get_fdata().astype(bool)
        inter_mask_3d = roi_mask_3d & mask_bool
        n_vox_roi     = int(inter_mask_3d.sum())
        if n_vox_roi == 0:
            continue

        inter_vec_bool = apply_mask(image.new_img_like(MASK_IMG, inter_mask_3d.astype(int)), MASK_IMG).astype(bool)
        inter_vec_bool = inter_vec_bool & mask_vec_bool
        masked_idx     = np.where(inter_vec_bool)[0]
        if masked_idx.size == 0:
            continue

        r_roi   = corr_vec[masked_idx]
        p_roi   = emp_p[masked_idx]

        # Within-ROI voxel FDR (descriptive)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rej_roi, p_fdr_roi, _, _ = multipletests(p_roi, alpha=0.05, method="fdr_bh")

        abs_r_roi    = np.abs(r_roi)
        mean_abs_r   = float(np.mean(abs_r_roi))
        median_abs_r = float(np.median(abs_r_roi))
        sig_mask     = rej_roi.astype(bool)
        n_sig_pos    = int(np.sum(sig_mask & (r_roi > 0)))
        n_sig_neg    = int(np.sum(sig_mask & (r_roi < 0)))

        # Per-ROI, per-story voxel table (optional)
        roi_df = pd.DataFrame({
            "masked_idx": masked_idx.astype(int),
            "r": r_roi.astype(float),
            "emp_p": p_roi.astype(float),
            "sig_FDR_withinROI": rej_roi.astype(bool),
        }).sort_values("r", ascending=False).reset_index(drop=True)
        write_csv_with_sep_hint(roi_df, os.path.join(OUT_DIR, f"roi_voxel_stats__{story}__{roi_name}.csv"))

        # Descriptive per-ROI summary
        roi_summary_rows.append(dict(
            story=story, roi=roi_name, n_vox=int(masked_idx.size),
            mean_r=float(np.mean(r_roi)), median_r=float(np.median(r_roi)),
            mean_abs_r=mean_abs_r, median_abs_r=median_abs_r,
            n_sig_FDR_withinROI=int(np.sum(rej_roi)),
            prop_sig_FDR_withinROI=float(np.mean(rej_roi)),
            n_sig_withinROI_pos=n_sig_pos,
            n_sig_withinROI_neg=n_sig_neg,
        ))

        # Cache for second pass (plots + family rows)
        per_roi_cache[roi_name] = dict(
            masked_idx=masked_idx, r_roi=r_roi, p_roi=p_roi,
            rej_roi=rej_roi, p_fdr_roi=p_fdr_roi,
            mean_abs_r=mean_abs_r, median_abs_r=median_abs_r,
            n_sig_pos=n_sig_pos, n_sig_neg=n_sig_neg
        )

        # ---- FIXED: ROI-level omnibus p via Simes (ACROSS VOXELS) ----
        p_simes = simes_1d(p_roi)   # <-- correct 1-D Simes across voxels in this ROI
        tmp_roi_names.append(roi_name)
        tmp_roi_p_simes.append(p_simes)
        tmp_roi_sizes.append(int(masked_idx.size))

    # FWE across ROIs (Holm + Bonferroni)
    if tmp_roi_names:
        p_simes_arr = np.array(tmp_roi_p_simes, dtype=float)

        # Holm–Bonferroni (controls FWE under arbitrary dependence)
        rej_holm, p_holm, _, _ = multipletests(p_simes_arr, alpha=0.05, method="holm")

        # Bonferroni (also FWE, conservative)
        rej_bonf, p_bonf, _, _ = multipletests(p_simes_arr, alpha=0.05, method="bonferroni")

        for i, (roi_name, p_simes_val, n_vox_roi) in enumerate(zip(tmp_roi_names, tmp_roi_p_simes, tmp_roi_sizes)):
            entry      = per_roi_cache[roi_name]
            r_roi      = entry["r_roi"]
            rej_roi    = entry["rej_roi"]

            if DO_PLOTS_PER_STORY and DO_PER_ROI_VIOLIN_PLOTS:
                save_violin(
                    values=r_roi,
                    out_base=os.path.join(OUT_DIR, f"violin__{story}__{roi_name}"),
                    ylim=(VMIN, VMAX),
                    dpi=600,
                    save_pdf=SAVE_PDF_FIGS
                )

            if DO_PLOTS_PER_STORY and DO_PER_ROI_ANAT_PLOTS:
                roi_vec  = np.zeros_like(corr_vec)
                roi_vals = r_roi.copy()
                roi_vals[np.abs(roi_vals) < THRESH] = 0.0
                roi_vec[entry["masked_idx"]] = roi_vals
                roi_img4   = masker.inverse_transform(roi_vec.reshape(1, -1))
                roi_img    = image.index_img(roi_img4, 0)
                t1_img = safe_load_t1(T1_BRAIN_NII) or safe_load_t1(T1_NII)
                disp_roi   = plotting.plot_stat_map(
                    roi_img, bg_img=t1_img,
                    title=None, display_mode="mosaic", cmap="cold_hot",
                    black_bg=True, symmetric_cbar=True,
                    vmin=VMIN, vmax=VMAX, figure=plt.figure(figsize=(10, 8)),
                )
                fig_roi = disp_roi.frame_axes.figure
                fig_roi.savefig(os.path.join(OUT_DIR, f"roi_corr_map__{story}__{roi_name}.png"), dpi=300)
                plt.close(fig_roi)

            roi_family_rows.append(dict(
                story=story,
                roi=roi_name,
                n_vox=n_vox_roi,
                mean_r=float(np.mean(r_roi)),
                median_r=float(np.median(r_roi)),
                mean_abs_r=float(entry["mean_abs_r"]),
                median_abs_r=float(entry["median_abs_r"]),
                any_sig_withinROI=bool(np.any(rej_roi)),
                prop_sig_withinROI=float(np.mean(rej_roi)),
                n_sig_withinROI_pos=int(entry["n_sig_pos"]),
                n_sig_withinROI_neg=int(entry["n_sig_neg"]),
                p_simes=float(p_simes_val),         # ROI-level omnibus p (Simes)
                p_holm=float(p_holm[i]),            # Holm-adjusted p (FWE)
                sig_FWE_Holm=bool(rej_holm[i]),
                p_bonferroni=float(p_bonf[i]),      # Bonferroni-adjusted p (FWE)
                sig_FWE_Bonferroni=bool(rej_bonf[i]),
            ))

    # Write summaries
    if roi_summary_rows:
        roi_summary_df = pd.DataFrame(roi_summary_rows).sort_values(["roi"]).reset_index(drop=True)
        write_csv_with_sep_hint(roi_summary_df, os.path.join(OUT_DIR, f"roi_summary__{story}.csv"))

    if roi_family_rows:
        roi_family_df = pd.DataFrame(roi_family_rows).sort_values(["roi"]).reset_index(drop=True)
        write_csv_with_sep_hint(roi_family_df, os.path.join(OUT_DIR, f"roi_family_summary__{story}.csv"))
        MASTER_ROWS.extend(roi_family_rows)
    else:
        print(f"[ROI] {story}: no ROI family rows written.")

    GLOBAL_CORR_VECS.append(corr_vec)
    GLOBAL_EMP_P.append(emp_p)
    GLOBAL_WB_REJ.append(rej_wb.astype(bool))
    GLOBAL_ALL_R.append(corr_vec)

# ---------------- main ----------------
def main():
    story_dir = os.path.join(".", in_folder)
    stories = discover_stories(story_dir)
    print(f"Discovered stories: {stories}")
    if not stories:
        raise RuntimeError("No stories found in " + story_dir)

    for s in stories:
        try:
            process_story(s)
        except Exception as e:
            print(f"[{s}] ERROR: {e}")

    out_dir = os.path.join(HERE, in_folder, "out")
    os.makedirs(out_dir, exist_ok=True)

    if MASTER_ROWS:
        df_master = pd.DataFrame(MASTER_ROWS).sort_values(["story", "roi"]).reset_index(drop=True)
        write_csv_with_sep_hint(df_master, os.path.join(out_dir, "summary_all_stories__roi_family.csv"))
        print("[MASTER] wrote out/summary_all_stories__roi_family.csv")

        if DO_PLOT_ROI_COUNTS:
            build_roi_simes_count_map(
                df_master=df_master,
                mask_img_path=MASK_IMG_PATH,
                roi_glob=REGION_MASKS_GLOB,
                t1_path=(T1_BRAIN_NII if os.path.isfile(T1_BRAIN_NII) else T1_NII),
                out_dir=out_dir
            )

    if DO_PLOT_ACROSS_STORIES and GLOBAL_CORR_VECS:
        build_across_stories_maps(
            corr_vecs=GLOBAL_CORR_VECS,
            emp_ps=GLOBAL_EMP_P,
            wb_rej=GLOBAL_WB_REJ,
            mask_img_path=MASK_IMG_PATH,
            t1_path=(T1_BRAIN_NII if os.path.isfile(T1_BRAIN_NII) else T1_NII),
            out_dir=out_dir,
            alpha=ACROSS_STORIES_ALPHA
        )

    if DO_GLOBAL_ALL_VOX_VIOLIN and GLOBAL_ALL_R:
        all_r = np.concatenate(GLOBAL_ALL_R, axis=0)
        total_vox = all_r.size
        save_violin(
            values=all_r,
            out_base=os.path.join(out_dir, "violin__GLOBAL_ALL_STORIES_ALL_VOXELS"),
            ylim=(VMIN, VMAX),
            dpi=600,
            save_pdf=SAVE_PDF_FIGS
        )
        print(f"[GLOBAL] Wrote global all-voxels violin (n={total_vox}).")

if __name__ == "__main__":
    main()
