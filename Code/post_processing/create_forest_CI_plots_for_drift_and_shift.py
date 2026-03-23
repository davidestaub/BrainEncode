#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, re, glob, pickle
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.masking import apply_mask
from nilearn.maskers import NiftiMasker
from statsmodels.stats.multitest import multipletests
from scipy import stats
from scipy.stats import norm, t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- PATHS (compute first, then plot) ----------------
SHIFT_DIR = "../../../../DATA/BrainEncode/text_metrics_results/text_eventboundary_log_prob_STACKED_EVENT_chunklen40_alphas_-6to0_30_nboots_100_use_corr_True_MEAN_HIP_R"
DRIFT_DIR = "../../../../DATA/BrainEncode/text_metrics_results/text_context_drift_DETREND_STABILIZE0_RHO02_USE_PCA0_layer32_chunklen40_alphas_-6to0_50_nboots_100_use_corr_True_MEAN_HIP_R"

DATA_OUT_DIR = os.path.join("out", "UNIQUE_shift_vs_drift")   # NIfTIs, CSVs, etc.
FIG_DIR      = os.path.join(DATA_OUT_DIR, "figures")          # forest plots, etc.

MASK_IMG_PATH = os.path.join("../../../../DATA/BrainEncode/anatomical_files", "binarized_mask.nii.gz")
ROI_GLOB      = os.path.join("../../../../DATA/BrainEncode/MASKS", "*.nii*")
T1_IMG        = os.path.join("../../../../DATA/BrainEncode/anatomical_files", "Robert_T1_brain_in_func_space_space.nii.gz")
ZEROS_IDX_PKL = "indices_of_zero_voxels_GM.pkl"  # optional

ALPHA_FDR = 0.05           # voxelwise BH–FDR
ALPHA_LABEL = 0.05         # label star threshold at ROI level
SAVE_PDF = True
PERC_LIM = 99              # Δβ mosaic scale

np.random.seed(123)

# ---------------- ROI set patterns (for figure subsets) ----------
LANG_PATTERNS = [
    ["heschl"], ["planum temporale"], ["planum polare"],
    ["superior temporal gyrus","anterior"],
    ["superior temporal gyrus","posterior"],
    ["middle temporal gyrus"], ["temporal pole"],
    ["supramarginal gyrus"], ["frontal operculum"], ["central opercular"], ["parietal operculum"],
    ["inferior frontal gyrus","triangularis"], ["inferior frontal gyrus","opercularis"],
    ["insular cortex"]
]
DMNPI_PATTERNS = [
    ["angular gyrus"], ["precuneus"],
    ["posterior cingulate"], ["cingulate gyrus","posterior"],
    ["frontal medial cortex"], ["medial frontal cortex"], ["paracingulate"],
    ["superior parietal"]
]

# ---------------- Small helpers ----------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def load_pkl(path):
    with open(path, "rb") as f: return pickle.load(f)

def zscore_time(X):
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True); sd[sd==0]=1
    return X / sd

def discover_common_stories(shift_dir, drift_dir):
    pat = "full_prediction_*.pkl"
    s = {os.path.basename(p).split("full_prediction_")[1].split(".pkl")[0]
         for p in glob.glob(os.path.join(shift_dir, pat))}
    print(s)
    d = {os.path.basename(p).split("full_prediction_")[1].split(".pkl")[0]
         for p in glob.glob(os.path.join(drift_dir, pat))}
    print(d)
    return sorted(list(s & d))

def insert_dropped_voxels(vec_kept, dropped_idx, n_vox_full):
    if dropped_idx is None: return vec_kept
    full = np.zeros(n_vox_full, dtype=vec_kept.dtype)
    keep_mask = np.ones(n_vox_full, dtype=bool); keep_mask[dropped_idx] = False
    full[keep_mask] = vec_kept
    return full

def one_sample_t(x):   # x: (S, V)
    m  = x.mean(axis=0)
    se = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    t  = np.divide(m, se, out=np.zeros_like(m), where=se>0)
    p  = 2*stats.t.sf(np.abs(t), df=x.shape[0]-1)
    return m, p

def simes_1d(pvals):
    p = np.asarray(pvals, dtype=float)
    p = p[np.isfinite(p)]
    m = p.size
    if m == 0: return np.nan
    ps = np.sort(p)
    i  = np.arange(1, m+1, dtype=float)
    return float(np.clip(np.min((m/i)*ps), 0.0, 1.0))

def paired_cohens_d_against_zero(x):
    x = np.asarray(x); s = x.std(ddof=1)
    return (x.mean()/s) if s>0 else np.nan

def t_based_mean_ci(x, alpha=0.05):
    x = np.asarray(x); S=x.size; m=x.mean(); s=x.std(ddof=1)
    h = t.ppf(1 - alpha/2, df=S-1) * s / np.sqrt(S) if S>1 and s>0 else 0.0
    return (m - h, m + h)

def bootstrap_mean_ci(x, n_boot=20000, alpha=0.05, random_state=12345):
    x = np.asarray(x); S = x.size
    rng = np.random.default_rng(random_state)
    idx = rng.integers(0, S, size=(n_boot, S))
    boot_means = x[idx].mean(axis=1)
    point = float(x.mean())
    # BCa
    eps = 1e-12
    prop = np.clip(np.mean(boot_means < point), eps, 1-eps)
    z0 = norm.ppf(prop)
    if S >= 3:
        jack = np.array([np.mean(np.delete(x, i)) for i in range(S)])
        jm = jack.mean()
        num = np.sum((jm - jack)**3)
        den = 6.0 * (np.sum((jm - jack)**2) ** 1.5)
        a = num/den if den>0 else 0.0
    else:
        a = 0.0
    z_alpha = norm.ppf([alpha/2, 1 - alpha/2])
    z_adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
    probs = norm.cdf(z_adj)
    lo, hi = np.quantile(boot_means, probs)
    return point, (float(lo), float(hi)), boot_means

# -------------- Pretty ROI labels -----------------
def _std(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _matches_any_pattern(roi_label, pattern_list):
    s = _std(roi_label)
    for tokens in pattern_list:
        if all(tok in s for tok in tokens): return True
    return False

REPLACERS = [
    (r"superior temporal gyrus anterior division", "STG (ant.)"),
    (r"superior temporal gyrus posterior division", "STG (post.)"),
    (r"middle temporal gyrus anterior division", "MTG (ant.)"),
    (r"middle temporal gyrus posterior division", "MTG (post.)"),
    (r"heschl.?s gyrus.*", "Heschl’s"),
    (r"planum temporale", "Planum temporale"),
    (r"planum polare", "Planum polare"),
    (r"supramarginal gyrus anterior division", "SMG (ant.)"),
    (r"supramarginal gyrus posterior division", "SMG (post.)"),
    (r"central opercular cortex", "Operculum (central)"),
    (r"parietal operculum cortex", "Operculum (parietal)"),
    (r"inferior frontal gyrus pars triangularis", "IFG (tri.)"),
    (r"inferior frontal gyrus pars opercularis", "IFG (op.)"),
    (r"temporal pole", "Temporal pole"),
    (r"insular cortex", "Insula"),
    (r"angular gyrus", "Angular gyrus"),
    (r"precuneus cortex|precuneus", "Precuneus"),
    (r"frontal medial cortex|medial frontal cortex", "mPFC"),
    (r"paracingulate gyrus", "Paracingulate"),
    (r"posterior cingulate gyrus|cingulate gyrus posterior division", "Posterior cingulate"),
    (r"superior parietal lobule", "SPL"),
]

def pretty_roi_name(raw):
    """
    From e.g., '45_Heschl's_Gyrus_includes_H1_and_H2_right.nii' to 'Heschl’s (R)'
    """
    s = os.path.splitext(os.path.basename(raw))[0]
    s = re.sub(r"^\d+_", "", s)        # drop numeric prefix
    s = s.replace("_", " ")
    hemi = None
    if re.search(r"\bleft\b", s, re.I):  hemi = "(L)"
    if re.search(r"\bright\b", s, re.I): hemi = "(R)"
    s = re.sub(r"\b(left|right)\b", "", s, flags=re.I).strip()

    s_std = _std(s)
    pretty = s  # default
    for pat, repl in REPLACERS:
        if re.search(pat, s_std, flags=re.I):
            pretty = repl; break

    if pretty == s:
        pretty = s.title()

    return f"{pretty} {hemi}" if hemi else pretty

# -------------- Forest plotting -------------------
def forest(df, value_col, lo_col, hi_col, label_col, pcol,
           title, xlabel, outfile_png, outfile_pdf, top_n=20, annotate_d_col=None):
    if df.empty: return
    dfx = df.copy()
    dfx["abs_effect"] = dfx[value_col].abs()
    dfx = dfx.sort_values("abs_effect", ascending=False).head(top_n)
    dfx = dfx.iloc[::-1]   # largest at bottom
    y = np.arange(len(dfx))
    x  = dfx[value_col].to_numpy()
    lo = dfx[lo_col].to_numpy()
    hi = dfx[hi_col].to_numpy()

    # Build labels
    labels = []
    for _, row in dfx.iterrows():
        lab = pretty_roi_name(row[label_col])
        if pd.notnull(row[pcol]) and row[pcol] < ALPHA_LABEL:
            lab = lab  # keep as-is; star/formatting could be added here if desired
        labels.append(lab)

    fig_h = max(4.0, 0.45*len(dfx)+1.0)
    fig, ax = plt.subplots(figsize=(8.7, fig_h))

    # Error bars
    xerr = np.vstack([x - lo, hi - x])
    ax.errorbar(x, y, xerr=xerr, fmt="o", capsize=3, linewidth=1.2)

    # symmetric x-axis around zero
    lim = float(np.nanmax(np.abs(np.r_[lo, hi])))
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlim(-lim*1.05, +lim*1.05)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)

    # Leave a right gutter for d-labels and place them OUTSIDE the axes
    # so they never overlap with CIs.
    # The rect right edge < 1.0 reserves space for the gutter.
    fig.tight_layout(rect=[0, 0, 0.86, 1])  # ~14% gutter on the right

    if annotate_d_col and (annotate_d_col in dfx.columns):
        for yi, dval in zip(y, dfx[annotate_d_col].to_numpy()):
            if pd.notnull(dval):
                ax.text(1.005, yi, f"d={dval:.2f}",
                        transform=ax.get_yaxis_transform(),  # x in axes (0..1), y in data
                        ha="left", va="center",
                        clip_on=False, fontsize=9)

    # Save
    fig.savefig(outfile_png, dpi=300, bbox_inches="tight")
    if SAVE_PDF:
        fig.savefig(outfile_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------- Main compute --------------------
def compute_and_write():
    ensure_dir(DATA_OUT_DIR); ensure_dir(FIG_DIR)

    mask_img  = nib.load(MASK_IMG_PATH)
    mask_bool = mask_img.get_fdata().astype(bool)
    n_vox_full = int(mask_bool.sum())
    masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()

    dropped_idx = None
    if os.path.isfile(ZEROS_IDX_PKL):
        dropped_idx = np.array(load_pkl(ZEROS_IDX_PKL), dtype=int)

    stories = discover_common_stories(SHIFT_DIR, DRIFT_DIR)
    if not stories: raise RuntimeError("No overlapping stories found.")

    bet_s_list, bet_d_list, dbet_list = [], [], []
    for s in stories:
        print("STORY:",s)
        PS = load_pkl(os.path.join(SHIFT_DIR, f"full_prediction_{s}.pkl")).T
        PD = load_pkl(os.path.join(DRIFT_DIR, f"full_prediction_{s}.pkl")).T
        resp_path = os.path.join("../../../../DATA/BrainEncode/response_data", f"full_response_{s}.pkl")
        if not os.path.isfile(resp_path):
            rp = os.path.join(SHIFT_DIR, f"full_response_{s}.pkl")
            if not os.path.isfile(rp): rp = os.path.join(DRIFT_DIR, f"full_response_{s}.pkl")
            resp_path = rp
        Y  = load_pkl(resp_path).T

        Y  = zscore_time(Y); PS = zscore_time(PS); PD = zscore_time(PD)
        T, V = Y.shape
        bet_s = np.empty(V); bet_d = np.empty(V)
        for v in range(V):
            X = np.column_stack([PS[:,v], PD[:,v]])
            XtX = X.T @ X; Xty = X.T @ Y[:,v]
            try:    beta = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError: beta = np.linalg.pinv(XtX) @ Xty
            bet_s[v], bet_d[v] = beta
        bet_s_list.append(bet_s); bet_d_list.append(bet_d); dbet_list.append(bet_s - bet_d)

    bet_s_arr = np.vstack(bet_s_list)
    bet_d_arr = np.vstack(bet_d_list)
    dbet_arr  = np.vstack(dbet_list)

    mean_bs, p_bs = one_sample_t(bet_s_arr)
    mean_bd, p_bd = one_sample_t(bet_d_arr)
    mean_db, p_db = one_sample_t(dbet_arr)

    # --- Save NIfTIs (as you had) ---
    def to_img(vec_kept):
        vec_full = insert_dropped_voxels(vec_kept, dropped_idx, n_vox_full)
        return image.index_img(masker.inverse_transform(vec_full.reshape(1,-1)), 0)

    nib.save(to_img(mean_bs), os.path.join(DATA_OUT_DIR,"mean_beta_shift_unique.nii.gz"))
    nib.save(to_img(mean_bd), os.path.join(DATA_OUT_DIR,"mean_beta_drift_unique.nii.gz"))
    nib.save(to_img(mean_db), os.path.join(DATA_OUT_DIR,"mean_delta_beta_unique.nii.gz"))

    rej_bs, _, _, _ = multipletests(p_bs, alpha=ALPHA_FDR, method="fdr_bh")
    rej_bd, _, _, _ = multipletests(p_bd, alpha=ALPHA_FDR, method="fdr_bh")
    rej_db, _, _, _ = multipletests(p_db, alpha=ALPHA_FDR, method="fdr_bh")

    nib.save(to_img(rej_bs.astype(float)), os.path.join(DATA_OUT_DIR,"sig_unique_shift_FDR.nii.gz"))
    nib.save(to_img(rej_bd.astype(float)), os.path.join(DATA_OUT_DIR,"sig_unique_drift_FDR.nii.gz"))
    nib.save(to_img(rej_db.astype(float)), os.path.join(DATA_OUT_DIR,"sig_delta_beta_FDR.nii.gz"))

    # Δβ mosaic
    mean_db_vec = insert_dropped_voxels(mean_db, dropped_idx, n_vox_full)
    rej_db_vec  = insert_dropped_voxels(rej_db.astype(float), dropped_idx, n_vox_full).astype(bool)
    mean_db_masked = mean_db_vec.copy(); mean_db_masked[~rej_db_vec] = 0.0
    mean_db_masked_img = image.index_img(masker.inverse_transform(mean_db_masked.reshape(1,-1)), 0)
    nib.save(mean_db_masked_img, os.path.join(DATA_OUT_DIR,"mean_delta_beta_unique_masked_FDR.nii.gz"))
    absmax = np.nanpercentile(np.abs(mean_db_vec[rej_db_vec]), PERC_LIM) if np.any(rej_db_vec) else np.nanpercentile(np.abs(mean_db_vec), PERC_LIM)
    absmax = float(max(absmax, 1e-6))
    t1 = nib.load(T1_IMG) if os.path.isfile(T1_IMG) else None
    disp = plotting.plot_stat_map(mean_db_masked_img, bg_img=t1,
                                  title=None, display_mode="mosaic",
                                  cmap="cold_hot", black_bg=True,
                                  symmetric_cbar=True, vmin=-absmax, vmax=+absmax,
                                  figure=plt.figure(figsize=(10,8)))
    disp.savefig(os.path.join(DATA_OUT_DIR,"delta_beta_mosaic_masked_FDR.png"), dpi=600)
    if SAVE_PDF: disp.savefig(os.path.join(DATA_OUT_DIR,"delta_beta_mosaic_masked_FDR.pdf"))
    plt.close(disp.frame_axes.figure)

    # --- ROI-level CIs and per-story values ---
    keep_mask = np.ones(n_vox_full, dtype=bool)
    if dropped_idx is not None: keep_mask[dropped_idx] = False
    kept_to_full_idx = np.where(keep_mask)[0]
    full_to_kept = -np.ones(n_vox_full, dtype=int)
    full_to_kept[kept_to_full_idx] = np.arange(kept_to_full_idx.size)

    p_bs_full = insert_dropped_voxels(p_bs, dropped_idx, n_vox_full)
    p_bd_full = insert_dropped_voxels(p_bd, dropped_idx, n_vox_full)
    p_db_full = insert_dropped_voxels(p_db, dropped_idx, n_vox_full)

    roi_ci_rows, per_story_rows = [], []
    for roi_path in sorted(glob.glob(ROI_GLOB)):
        if os.path.basename(roi_path) == os.path.basename(MASK_IMG_PATH): continue
        try:
            roi_img = nib.load(roi_path)
            if roi_img.shape != mask_img.shape or not np.allclose(roi_img.affine, mask_img.affine):
                roi_img = image.resample_to_img(roi_img, mask_img, interpolation="nearest")
        except Exception:
            continue
        roi_mask_full = (roi_img.get_fdata().astype(bool) & mask_bool)
        roi_vec_full  = apply_mask(image.new_img_like(mask_img, roi_mask_full.astype(np.uint8)), mask_img).astype(bool)
        roi_full_idx = np.where(roi_vec_full)[0]
        if roi_full_idx.size==0: continue
        roi_kept_pos = full_to_kept[roi_full_idx]; roi_kept_pos = roi_kept_pos[roi_kept_pos>=0]
        if roi_kept_pos.size==0: continue

        # per-story means in this ROI
        roi_shift_story = np.nanmean(bet_s_arr[:, roi_kept_pos], axis=1)
        roi_drift_story = np.nanmean(bet_d_arr[:, roi_kept_pos], axis=1)
        roi_delta_story = np.nanmean(dbet_arr[:, roi_kept_pos], axis=1)

        mean_delta, (lo_delta, hi_delta), _ = bootstrap_mean_ci(roi_delta_story)
        mean_shift, (lo_shift, hi_shift), _ = bootstrap_mean_ci(roi_shift_story)
        mean_drift, (lo_drift, hi_drift), _ = bootstrap_mean_ci(roi_drift_story)

        d_delta = paired_cohens_d_against_zero(roi_delta_story)
        d_shift = paired_cohens_d_against_zero(roi_shift_story)
        d_drift = paired_cohens_d_against_zero(roi_drift_story)

        p_simes_shift = simes_1d(p_bs_full[roi_vec_full])
        p_simes_drift = simes_1d(p_bd_full[roi_vec_full])
        p_simes_delta = simes_1d(p_db_full[roi_vec_full])

        roi_name = os.path.splitext(os.path.basename(roi_path))[0]
        roi_ci_rows.append(dict(
            roi=roi_name,
            mean_delta_beta=mean_delta, ci_lo_delta_beta=lo_delta, ci_hi_delta_beta=hi_delta,
            cohens_d_delta=d_delta,
            mean_beta_shift_unique=mean_shift, ci_lo_shift=lo_shift, ci_hi_shift=hi_shift,
            cohens_d_shift=d_shift, p_simes_unique_shift=p_simes_shift,
            mean_beta_drift_unique=mean_drift, ci_lo_drift=lo_drift, ci_hi_drift=hi_drift,
            cohens_d_drift=d_drift, p_simes_unique_drift=p_simes_drift,
            p_simes_delta_beta=p_simes_delta, n_vox=int(roi_kept_pos.size)
        ))

        for story_idx, story_name in enumerate(stories):
            per_story_rows.append(dict(
                roi=roi_name, story=story_name,
                delta_beta=roi_delta_story[story_idx],
                beta_shift_unique=roi_shift_story[story_idx],
                beta_drift_unique=roi_drift_story[story_idx]
            ))

    df_ci = pd.DataFrame(roi_ci_rows).sort_values("roi").reset_index(drop=True)
    df_ci.to_csv(os.path.join(DATA_OUT_DIR,"roi_unique_effects_with_CI.csv"), index=False)
    pd.DataFrame(per_story_rows).to_csv(os.path.join(DATA_OUT_DIR,"roi_unique_effects_per_story.csv"), index=False)
    return df_ci

# ---------------- Make plots ----------------------
def make_forest_plots(df_ci):
    # subsets by a‑priori sets + ROI‑level Simes filter
    def _filter_set(df, patterns, value_col, pcol):
        keep = df["roi"].apply(lambda r: _matches_any_pattern(r, patterns))
        sub = df.loc[keep].copy()
        sub = sub.loc[sub[pcol].notna() & (sub[pcol] < ALPHA_LABEL)].copy()
        return sub

    df_lang  = _filter_set(df_ci, LANG_PATTERNS,  "mean_beta_shift_unique", "p_simes_unique_shift")
    df_dmnpi = _filter_set(df_ci, DMNPI_PATTERNS, "mean_beta_drift_unique", "p_simes_unique_drift")

    forest(df_lang,
           value_col="mean_beta_shift_unique", lo_col="ci_lo_shift", hi_col="ci_hi_shift",
           label_col="roi", pcol="p_simes_unique_shift",
           xlabel="Unique β  (shift controlling for drift)",
           outfile_png=os.path.join(FIG_DIR,"fig_forest_beta_shift_LANG.png"),
           outfile_pdf=os.path.join(FIG_DIR,"fig_forest_beta_shift_LANG.pdf"),
           top_n=20, annotate_d_col="cohens_d_shift",
           title=None)

    forest(df_dmnpi,
           value_col="mean_beta_drift_unique", lo_col="ci_lo_drift", hi_col="ci_hi_drift",
           label_col="roi", pcol="p_simes_unique_drift",
           xlabel="Unique β (drift controlling for shift)",
           outfile_png=os.path.join(FIG_DIR,"fig_forest_beta_drift_DMNPIs.png"),
           outfile_pdf=os.path.join(FIG_DIR,"fig_forest_beta_drift_DMNPIs.pdf"),
           top_n=20, annotate_d_col="cohens_d_drift",title=None)

# ---------------- script entry --------------------
if __name__ == "__main__":
    df_ci = compute_and_write()
    make_forest_plots(df_ci)
