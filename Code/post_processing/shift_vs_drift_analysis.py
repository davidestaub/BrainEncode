#!/usr/bin/env python
import os, glob, pickle, numpy as np, pandas as pd
import nibabel as nib
from nilearn import image, plotting
from nilearn.masking import apply_mask
from nilearn.maskers import NiftiMasker
from statsmodels.stats.multitest import multipletests
from scipy import stats
from scipy.stats import t as tdist, norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import ndimage
import json

# ==================== GLOBAL CONFIG ====================
#SELECT WHICH TWO FEATURS TO COMPARE

SHIFT_DIR = "../../../../DATA/BrainEncode/text_metrics_results/text_eventboundary_log_prob_STACKED_EVENT_chunklen40_alphas_-6to0_30_nboots_100_use_corr_True_MEAN_HIP_R"

# DRIFT path template (placeholders {rho} and {layer})
DRIFT_DIR_TEMPLATE = "../../../../DATA/BrainEncode/text_metrics_results/text_context_drift_DETREND_STABILIZE0_RHO{rho}_USE_PCA0_layer{layer}_chunklen40_alphas_-6to0_50_nboots_100_use_corr_True_MEAN_HIP_R"

#PUNCT PATH
DRIFT_DIR_TEMPLATE = "../../../../DATA/BrainEncode/text_metrics_results/text_punctuation_chunklen40_alphas_-6to0_50_nboots_20_use_corr_True_MEAN_HIP_R"


# Variants to run
LAYERS    = [79]
RHO_CODES = ["02",]

# Output root
OUT_DIR = os.path.join("out", "UNIQUE_shift_vs_punct")

# Anatomy / masks
MASK_IMG_PATH = os.path.join("../../../../DATA/BrainEncode/anatomical_files", "binarized_mask.nii.gz")
ROI_GLOB      = os.path.join("../../../../DATA/BrainEncode/MASKS", "*.nii*")
T1_IMG        = os.path.join("../../../../DATA/BrainEncode/anatomical_files", "Robert_T1_brain_in_func_space_space.nii.gz")

# Optional: reinsert dropped voxels
ZEROS_IDX_PKL = "indices_of_zero_voxels_GM.pkl"

# Stats / plotting
ALPHA   = 0.05
PERC_LIM = 99
Z_THR    = 2.3
SAVE_PDF = True

# Final figure style (no labels/markers)
ROI_ANNOT_MIN_PROP   = 0.02
ROI_ANNOT_TOPN       = 12
CUTS_Q               = [0.15, 0.35, 0.65, 0.85]


# ==================== HELPERS ====================


def _voxel_sizes_mm(aff):
    # lengths of the three column vectors of the affine (mm per voxel along i,j,k)
    ax, ay, az = aff[:3, 0], aff[:3, 1], aff[:3, 2]
    return float(np.linalg.norm(ax)), float(np.linalg.norm(ay)), float(np.linalg.norm(az))

def _estimate_fwhm_and_resels_from_residuals(bet_arr, mean_vec, mask_img, masker):
    """
    Estimate smoothness (FWHM) from normalized residuals u = r / ||r|| at each voxel,
    then compute resel counts (R0..R3) using a convex-box approximation of the search region.

    Returns:
      dict(fwhm_iso_mm, resels=(R0,R1,R2,R3), R3_vol_check)
    """
    mask_bool = mask_img.get_fdata().astype(bool)
    S, V = bet_arr.shape
    # residuals across stories for the second-level one-sample model
    resid = bet_arr - mean_vec[None, :]  # (S, V)

    # residual volumes (S, X, Y, Z)
    vols = []
    for s in range(S):
        vol = image.index_img(masker.inverse_transform(resid[s].reshape(1, -1)), 0).get_fdata()
        vols.append(vol)
    vols = np.stack(vols, axis=0)

    # normalized residuals u = r / ||r|| (per voxel across stories)
    rnorm = np.sqrt(np.sum(vols**2, axis=0))
    rnorm[rnorm < 1e-12] = 1e-12
    u = vols / rnorm  # (S, X, Y, Z)

    # spatial derivatives of u in mm-units
    dx, dy, dz = _voxel_sizes_mm(mask_img.affine)
    grads_x = np.empty_like(u); grads_y = np.empty_like(u); grads_z = np.empty_like(u)
    for s in range(S):
        gx, gy, gz = np.gradient(u[s], dx, dy, dz, edge_order=1)
        grads_x[s] = gx; grads_y[s] = gy; grads_z[s] = gz

    # 3x3 Gram matrix M = (∇u)^T(∇u) at each voxel (inner product over 'stories' dimension)
    M_xx = np.sum(grads_x**2, axis=0)
    M_yy = np.sum(grads_y**2, axis=0)
    M_zz = np.sum(grads_z**2, axis=0)
    M_xy = np.sum(grads_x*grads_y, axis=0)
    M_xz = np.sum(grads_x*grads_z, axis=0)
    M_yz = np.sum(grads_y*grads_z, axis=0)

    # determinant of a symmetric 3x3
    detM = (M_xx*M_yy*M_zz + 2*M_xy*M_yz*M_xz
            - M_xx*M_yz**2 - M_yy*M_xz**2 - M_zz*M_xy**2)
    detM[~mask_bool] = np.nan
    detM = np.maximum(detM, 1e-24)

    # FWHM map (mm) from Worsley/SPM estimator; take geometric mean over mask for iso FWHM
    D = 3.0
    fwhm_map_mm = np.sqrt(4.0*np.log(2.0)) * np.power(detM, -1.0/(2.0*D))
    fwhm_iso_mm = float(np.exp(np.nanmean(np.log(fwhm_map_mm[mask_bool]))))

    # Resel counts with convex (bounding box) approximation
    ijk = np.column_stack(np.where(mask_bool))
    i0,i1 = ijk[:,0].min(), ijk[:,0].max()
    j0,j1 = ijk[:,1].min(), ijk[:,1].max()
    k0,k1 = ijk[:,2].min(), ijk[:,2].max()
    Lx_mm = (i1 - i0 + 1) * dx
    Ly_mm = (j1 - j0 + 1) * dy
    Lz_mm = (k1 - k0 + 1) * dz

    Lx = Lx_mm / fwhm_iso_mm
    Ly = Ly_mm / fwhm_iso_mm
    Lz = Lz_mm / fwhm_iso_mm

    R3 = Lx * Ly * Lz
    R2 = Lx*Ly + Lx*Lz + Ly*Lz
    R1 = Lx + Ly + Lz
    R0 = 1.0

    # volume-based cross-check for R3 (should be close)
    vox_vol = float(np.abs(np.linalg.det(mask_img.affine[:3, :3])))
    V_mm3 = vox_vol * int(mask_bool.sum())
    R3_vol = V_mm3 / (fwhm_iso_mm**3)

    return dict(fwhm_iso_mm=fwhm_iso_mm, resels=(R0, R1, R2, R3), R3_vol_check=R3_vol)

def _rft_peak_pvals_from_z(z_vec, resels):
    """
    Peak-level FWE p-values via EEC for a 3D Gaussian Z-field, two-sided.
    P(max Z >= u) ≈ Σ_{d=0}^3 R_d * ρ_d(u), where ρ_d are EC densities.
    """
    R0, R1, R2, R3 = resels
    u = np.abs(z_vec.astype(float))
    e = np.exp(-0.5 * u*u)

    c1 = np.sqrt(4.0*np.log(2.0)) / (2.0*np.pi)
    c2 = (4.0*np.log(2.0)) / ((2.0*np.pi)**1.5)
    c3 = (4.0*np.log(2.0))**1.5 / ((2.0*np.pi)**2)

    term0 = R0 * (1.0 - norm.cdf(u))
    term1 = R1 * c1 * e
    term2 = R2 * c2 * (u * e)
    term3 = R3 * c3 * ((u*u - 1.0) * e)

    eec = term0 + term1 + term2 + term3
    p = np.clip(2.0 * eec, 0.0, 1.0)  # two-sided correction
    return p



def load_pkl(path):
    with open(path, "rb") as f: return pickle.load(f)

def zscore_time(X):
    X = X - X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True); sd[sd==0]=1
    return X / sd

def discover_common_stories(shift_dir, drift_dir):
    pat = "full_prediction_*.pkl"
    sfiles = glob.glob(os.path.join(shift_dir, pat))
    dfiles = glob.glob(os.path.join(drift_dir, pat))
    def names(files):
        return {os.path.basename(p).split("full_prediction_")[1].split(".pkl")[0] for p in files}
    return sorted(list(names(sfiles) & names(dfiles)))

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

def one_sample_stats_full(x):  # x: (S, V)
    S  = x.shape[0]
    m  = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    se = np.divide(sd, np.sqrt(S), out=np.full_like(sd, np.nan), where=sd>0)
    t  = np.divide(m, se, out=np.zeros_like(m), where=np.isfinite(se) & (se>0))
    p  = 2.0 * tdist.sf(np.abs(t), df=S-1)
    d  = np.divide(m, sd, out=np.full_like(m, np.nan), where=sd>0)  # Cohen's d
    z  = np.sign(t) * norm.isf(p/2.0)  # signed z (two-sided)
    return m, t, z, d, p, sd

def nii_stem(name: str) -> str:
    b = os.path.basename(name)
    if b.endswith(".nii.gz"):
        return b[:-7]
    return os.path.splitext(b)[0]


# ==================== CORE PIPELINE (one drift variant) ====================
def run_for_drift_variant(shift_dir, drift_dir, out_root, layer, rho):
    tag = f"layer{layer}_rho{rho}"
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    def outfile(name):  # prefix filenames to make them self-descriptive
        return os.path.join(out_dir, f"{tag}__{name}")

    print(f"\n=== Running drift variant: {tag}")
    print(f"DRIFT_DIR = {drift_dir}")
    if not os.path.isdir(drift_dir):
        print(f"[WARN] Drift dir not found, skipping: {drift_dir}")
        return

    # Common imaging objects
    mask_img = nib.load(MASK_IMG_PATH)
    mask_bool = mask_img.get_fdata().astype(bool)
    n_vox_full = int(mask_bool.sum())
    masker = NiftiMasker(mask_img=mask_img, standardize=False).fit()
    t1 = nib.load(T1_IMG) if os.path.isfile(T1_IMG) else None

    dropped_idx = None
    if os.path.isfile(ZEROS_IDX_PKL):
        dropped_idx = np.array(load_pkl(ZEROS_IDX_PKL), dtype=int)

    stories = discover_common_stories(shift_dir, drift_dir)
    if not stories:
        print(f"[WARN] No overlapping stories for {tag}; skipping.")
        return

    # ========== Fit per story ==========
    bet_s_list, bet_d_list, dbet_list = [], [], []

    for s in stories:
        print(f"[story] {s}")
        PS = load_pkl(os.path.join(shift_dir, f"full_prediction_{s}.pkl")).T
        PD = load_pkl(os.path.join(drift_dir, f"full_prediction_{s}.pkl")).T
        resp_path = os.path.join("../../../../DATA/BrainEncode/response_data", f"full_response_{s}.pkl")
        Y = load_pkl(resp_path).T

        # Z-score across time
        Y  = zscore_time(Y)
        PS = zscore_time(PS)
        PD = zscore_time(PD)

        T, V = Y.shape
        bet_s = np.empty(V); bet_d = np.empty(V)
        for v in range(V):
            X = np.column_stack([PS[:, v], PD[:, v]])
            XtX = X.T @ X
            Xty = X.T @ Y[:, v]
            try:
                beta = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(XtX) @ Xty
            bet_s[v], bet_d[v] = beta

        dbet = bet_s - bet_d
        bet_s_list.append(bet_s); bet_d_list.append(bet_d); dbet_list.append(dbet)

        np.save(outfile(f"beta_shift_unique__{s}.npy"), bet_s)
        np.save(outfile(f"beta_drift_unique__{s}.npy"), bet_d)
        np.save(outfile(f"delta_beta_unique__{s}.npy"), dbet)

    # ========== Random-effects across stories ==========
    bet_s_arr = np.vstack(bet_s_list)
    bet_d_arr = np.vstack(bet_d_list)
    dbet_arr  = np.vstack(dbet_list)

    mean_bs, p_bs = one_sample_t(bet_s_arr)
    mean_bd, p_bd = one_sample_t(bet_d_arr)
    mean_db, p_db = one_sample_t(dbet_arr)

    rej_bs, p_bs_fdr, _, _ = multipletests(p_bs, alpha=ALPHA, method="fdr_bh")
    rej_bd, p_bd_fdr, _, _ = multipletests(p_bd, alpha=ALPHA, method="fdr_bh")
    rej_db, p_db_fdr, _, _ = multipletests(p_db, alpha=ALPHA, method="fdr_bh")

    # helper to write nifti from vector
    def to_img(vec_kept):
        vec_full = insert_dropped_voxels(vec_kept, dropped_idx, n_vox_full)
        return image.index_img(masker.inverse_transform(vec_full.reshape(1, -1)), 0)

    # Save means + masks
    img_mean_bs = to_img(mean_bs); nib.save(img_mean_bs, outfile("mean_beta_shift_unique.nii.gz"))
    img_mean_bd = to_img(mean_bd); nib.save(img_mean_bd, outfile("mean_beta_drift_unique.nii.gz"))
    img_mean_db = to_img(mean_db); nib.save(img_mean_db, outfile("mean_delta_beta_unique.nii.gz"))

    nib.save(to_img(rej_bs.astype(float)), outfile("sig_unique_shift_FDR.nii.gz"))
    nib.save(to_img(rej_bd.astype(float)), outfile("sig_unique_drift_FDR.nii.gz"))
    nib.save(to_img(rej_db.astype(float)), outfile("sig_delta_beta_FDR.nii.gz"))

    # Δβ masked mosaic
    mean_db_vec_full = insert_dropped_voxels(mean_db, dropped_idx, n_vox_full)
    rej_db_vec_full  = insert_dropped_voxels(rej_db.astype(float), dropped_idx, n_vox_full).astype(bool)
    mean_db_masked = mean_db_vec_full.copy()
    mean_db_masked[~rej_db_vec_full] = 0.0
    img_mean_db_masked = image.index_img(masker.inverse_transform(mean_db_masked.reshape(1, -1)), 0)
    nib.save(img_mean_db_masked, outfile("mean_delta_beta_unique_masked_FDR.nii.gz"))

    absmax = (np.nanpercentile(np.abs(mean_db_vec_full[rej_db_vec_full]), PERC_LIM)
              if np.any(rej_db_vec_full)
              else np.nanpercentile(np.abs(mean_db_vec_full), PERC_LIM))
    absmax = float(max(absmax, 1e-6))
    t1 = nib.load(T1_IMG) if os.path.isfile(T1_IMG) else None

    disp = plotting.plot_stat_map(
        img_mean_db_masked, bg_img=t1, title=None, display_mode="mosaic", cmap="cold_hot",
        black_bg=True, symmetric_cbar=True, vmin=-absmax, vmax=+absmax,
        figure=plt.figure(figsize=(10, 8))
    )
    disp.savefig(outfile("delta_beta_mosaic_masked_FDR.png"), dpi=600)
    if SAVE_PDF: disp.savefig(outfile("delta_beta_mosaic_masked_FDR.pdf"))
    plt.close(disp.frame_axes.figure)

    # ========== ROI-level vectors we reuse ==========
    mean_bs_vec = apply_mask(img_mean_bs, mask_img)
    mean_bd_vec = apply_mask(img_mean_bd, mask_img)
    mean_db_vec = apply_mask(img_mean_db, mask_img)

    # z / t / d (for plots)
    mean_bs2, t_bs, z_bs, d_bs, p_bs2, _ = one_sample_stats_full(bet_s_arr)
    mean_bd2, t_bd, z_bd, d_bd, p_bd2, _ = one_sample_stats_full(bet_d_arr)

    # Convenience plots
    def plot_pair(vec_left, vec_right, title_left, title_right, fn_left, fn_right,
                  cmap="bwr_r", threshold=None):
        both = np.r_[vec_left[np.isfinite(vec_left)], vec_right[np.isfinite(vec_right)]]
        nz = np.abs(both[(~np.isnan(both)) & (both != 0)])
        vmax = float(np.nanpercentile(nz, PERC_LIM)) if nz.size else 1e-6
        vmax = max(vmax, 1e-6)
        imgL = image.index_img(masker.inverse_transform(insert_dropped_voxels(vec_left, dropped_idx, n_vox_full).reshape(1,-1)), 0)
        imgR = image.index_img(masker.inverse_transform(insert_dropped_voxels(vec_right, dropped_idx, n_vox_full).reshape(1,-1)), 0)
        dispL = plotting.plot_stat_map(imgL, bg_img=t1, title=title_left, display_mode="mosaic",
                                       cmap=cmap, black_bg=True, symmetric_cbar=True,
                                       vmin=-vmax, vmax=+vmax, threshold=threshold,
                                       figure=plt.figure(figsize=(10,8)))
        dispL.savefig(outfile(fn_left), dpi=600); plt.close(dispL.frame_axes.figure)
        dispR = plotting.plot_stat_map(imgR, bg_img=t1, title=title_right, display_mode="mosaic",
                                       cmap=cmap, black_bg=True, symmetric_cbar=True,
                                       vmin=-vmax, vmax=+vmax, threshold=threshold,
                                       figure=plt.figure(figsize=(10,8)))
        dispR.savefig(outfile(fn_right), dpi=600); plt.close(dispR.frame_axes.figure)

    # β maps
    plot_pair(mean_bs, mean_bd,
              "Mean β (unique shift)", "Mean β (unique drift)",
              "beta_shift_unique_bwr.png", "beta_drift_unique_bwr.png", cmap="bwr_r")

    # t maps
    nib.save(to_img(t_bs), outfile("t_unique_shift.nii.gz"))
    nib.save(to_img(t_bd), outfile("t_unique_drift.nii.gz"))
    plot_pair(t_bs, t_bd,
              "t (unique shift, across stories)", "t (unique drift, across stories)",
              "t_shift_bwr.png", "t_drift_bwr.png", cmap="bwr_r")

    # z maps
    nib.save(to_img(z_bs), outfile("z_unique_shift.nii.gz"))
    nib.save(to_img(z_bd), outfile("z_unique_drift.nii.gz"))
    plot_pair(z_bs, z_bd,
              "z (unique shift)", "z (unique drift)",
              "z_shift_bwr.png", "z_drift_bwr.png", cmap="bwr_r")

    # z maps (uncorrected |z| >= Z_THR)
    z_bs_thr = z_bs.copy(); z_bs_thr[np.abs(z_bs_thr) < Z_THR] = 0.0
    z_bd_thr = z_bd.copy(); z_bd_thr[np.abs(z_bd_thr) < Z_THR] = 0.0
    nib.save(to_img(z_bs_thr), outfile(f"z_unique_shift_thr{str(Z_THR).replace('.','p')}.nii.gz"))
    nib.save(to_img(z_bd_thr), outfile(f"z_unique_drift_thr{str(Z_THR).replace('.','p')}.nii.gz"))
    plot_pair(z_bs_thr, z_bd_thr,
              f"z (shift), |z|≥{Z_THR}", f"z (drift), |z|≥{Z_THR}",
              f"z_shift_bwr_thr{str(Z_THR).replace('.','p')}.png",
              f"z_drift_bwr_thr{str(Z_THR).replace('.','p')}.png",
              cmap="bwr_r", threshold=0.0)

    # z maps (FDR-masked & |z| >= Z_THR)
    z_bs_fdrthr = z_bs.copy(); z_bs_fdrthr[~rej_bs] = 0.0; z_bs_fdrthr[np.abs(z_bs_fdrthr) < Z_THR] = 0.0
    z_bd_fdrthr = z_bd.copy(); z_bd_fdrthr[~rej_bd] = 0.0; z_bd_fdrthr[np.abs(z_bd_fdrthr) < Z_THR] = 0.0

    nib.save(to_img(z_bs_fdrthr), outfile(f"z_unique_shift_FDRmask_thr{str(Z_THR).replace('.','p')}.nii.gz"))
    nib.save(to_img(z_bd_fdrthr), outfile(f"z_unique_drift_FDRmask_thr{str(Z_THR).replace('.','p')}.nii.gz"))
    plot_pair(z_bs_fdrthr, z_bd_fdrthr,
              f"z (shift), FDR & |z|≥{Z_THR}", f"z (drift), FDR & |z|≥{Z_THR}",
              f"z_shift_bwr_FDRmasked_thr{str(Z_THR).replace('.','p')}.png",
              f"z_drift_bwr_FDRmasked_thr{str(Z_THR).replace('.','p')}.png",
              cmap="bwr_r", threshold=0.0)

    # Cohen's d
    nib.save(to_img(d_bs), outfile("cohen_d_unique_shift.nii.gz"))
    nib.save(to_img(d_bd), outfile("cohen_d_unique_drift.nii.gz"))
    plot_pair(d_bs, d_bd,
              "Cohen's d (unique shift)", "Cohen's d (unique drift)",
              "d_shift_bwr.png", "d_drift_bwr.png", cmap="bwr_r")

    # ==================== ROI infrastructure used by both FDR and z-only ====================
    # Cache ROI -> voxel vector (mask space)
    roi_vecs = {}
    for roi_path in sorted(glob.glob(ROI_GLOB)):
        if os.path.basename(roi_path) == os.path.basename(MASK_IMG_PATH):
            continue
        try:
            roi_img = nib.load(roi_path)
            if roi_img.shape != mask_img.shape or not np.allclose(roi_img.affine, mask_img.affine):
                roi_img = image.resample_to_img(roi_img, mask_img, interpolation="nearest")
        except Exception:
            continue
        roi_mask = (roi_img.get_fdata().astype(bool) & mask_bool)
        roi_vec  = apply_mask(image.new_img_like(mask_img, roi_mask.astype(np.uint8)), mask_img).astype(bool)
        roi_vecs[os.path.splitext(os.path.basename(roi_path))[0]] = roi_vec

    # convenient world-space extents for cut coords
    ijk_all = np.column_stack(np.where(mask_bool))
    xyz_all = nib.affines.apply_affine(mask_img.affine, ijk_all)
    x_cuts = np.quantile(xyz_all[:, 0], CUTS_Q).tolist()
    y_cuts = np.quantile(xyz_all[:, 1], CUTS_Q).tolist()
    z_cuts = np.quantile(xyz_all[:, 2], CUTS_Q).tolist()

    def build_prop_img(which: str, top_df: pd.DataFrame, mask_sel_vec: np.ndarray,
                       fill_whole_roi: bool = True, require_sig: bool = True):
        """Color whole ROI by ROI proportion; require_sig=True means ROI must have ≥1 suprathreshold voxel."""
        vals = np.zeros(n_vox_full, dtype=float)
        for _, row in top_df.iterrows():
            name = row["roi"]
            vec = roi_vecs.get(name, None)
            if vec is None: continue
            prop = float(row["prop_shift"] if which == "shift" else row["prop_drift"])
            if prop < ROI_ANNOT_MIN_PROP: continue
            has_sig = np.any(vec & mask_sel_vec)
            if require_sig and not has_sig: continue
            sel = vec if fill_whole_roi else (vec & mask_sel_vec)
            if not np.any(sel): continue
            vals[sel] = np.maximum(vals[sel], prop)
        return image.index_img(masker.inverse_transform(vals.reshape(1, -1)), 0)

    def draw_overlay_mosaic(which: str, top_df: pd.DataFrame, mask_sel_vec: np.ndarray,
                            base_tag: str, caption_suffix: str):
        """Anatomical overlay heatmap (no labels)."""
        if top_df.empty or not np.any(mask_sel_vec):
            print(f"[ROI overlay] No {which} ROIs after filtering; skipping.")
            return
        img = build_prop_img(which, top_df, mask_sel_vec)
        vmax = float(max(ROI_ANNOT_MIN_PROP, top_df["prop_shift"].max() if which=="shift" else top_df["prop_drift"].max()))
        cmap = plt.cm.magma.copy()
        cmap.set_under((0, 0, 0, 0))
        fig, axes = plt.subplots(3, 4, figsize=(10, 8), facecolor="black")
        thr_vis = max(0.0, ROI_ANNOT_MIN_PROP - 1e-12)

        for i, c in enumerate(z_cuts):
            plotting.plot_stat_map(img, bg_img=t1, display_mode="z", cut_coords=[float(c)],
                                   axes=axes[0, i], black_bg=True, cmap=cmap,
                                   threshold=thr_vis, vmin=ROI_ANNOT_MIN_PROP, vmax=vmax,
                                   colorbar=False)
        for i, c in enumerate(x_cuts):
            plotting.plot_stat_map(img, bg_img=t1, display_mode="x", cut_coords=[float(c)],
                                   axes=axes[1, i], black_bg=True, cmap=cmap,
                                   threshold=thr_vis, vmin=ROI_ANNOT_MIN_PROP, vmax=vmax,
                                   colorbar=False)
        for i, c in enumerate(y_cuts):
            plotting.plot_stat_map(img, bg_img=t1, display_mode="y", cut_coords=[float(c)],
                                   axes=axes[2, i], black_bg=True, cmap=cmap,
                                   threshold=thr_vis, vmin=ROI_ANNOT_MIN_PROP, vmax=vmax,
                                   colorbar=False)

        # shared colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=ROI_ANNOT_MIN_PROP, vmax=vmax))
        sm.set_array([])
        cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label(f"Proportion of ROI voxels, |z|≥{Z_THR}{caption_suffix}", rotation=90, fontsize=9, color="white")
        cb.ax.tick_params(labelsize=8, colors="white")
        for spine in cb.ax.spines.values(): spine.set_edgecolor("white")

        axes[0, 0].set_title("Axial (z)", fontsize=10, color="white", loc="left")
        axes[1, 0].set_title("Sagittal (x)", fontsize=10, color="white", loc="left")
        axes[2, 0].set_title("Coronal (y)", fontsize=10, color="white", loc="left")
        fig.suptitle(f"{tag} — ROI heatmap on anatomy — {which.capitalize()} (no labels) {caption_suffix}",
                     color="white", fontsize=12)

        fig.tight_layout(rect=[0, 0, 0.90, 1])
        fn_png = outfile(f"roi_{base_tag}_brain_annot_{which}.png")
        fig.savefig(fn_png, dpi=300, facecolor=fig.get_facecolor())
        if SAVE_PDF:
            fig.savefig(outfile(f"roi_{base_tag}_brain_annot_{which}.pdf"), facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[ROI overlay] Wrote: {fn_png}")

    def save_roi_matrix_heatmap(roi_df, top_shift, top_drift, base_tag: str, caption_suffix: str):
        """ROI × {Shift,Drift} matrix heatmap for the union of top ROIs."""
        rows = pd.Index(top_shift["roi"]).union(top_drift["roi"])
        if len(rows) == 0:
            print(f"[ROI matrix] No ROIs passed filters; skipping ({base_tag}).")
            return
        hm = (roi_df.set_index("roi")
                    .loc[rows, ["prop_shift", "prop_drift"]]
                    .copy())

        A = hm.to_numpy()
        A_masked = ma.masked_where(A < ROI_ANNOT_MIN_PROP, A)

        cmap = plt.cm.magma.copy()
        cmap.set_bad(color=(0.90, 0.90, 0.90, 1.0))

        fig_h = max(4.0, 0.35 * len(hm) + 1.2)
        fig, ax = plt.subplots(figsize=(6.0, fig_h))
        im = ax.imshow(A_masked, aspect="auto", cmap=cmap,
                       vmin=ROI_ANNOT_MIN_PROP, vmax=float(np.nanmax(A)) if np.isfinite(np.nanmax(A)) else ROI_ANNOT_MIN_PROP)

        ax.set_yticks(np.arange(len(hm)))
        ax.set_yticklabels(hm.index, fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Shift", "Drift"], fontsize=9)
        ax.set_xlabel(f"Proportion of ROI voxels with |z|≥{Z_THR}{caption_suffix}", fontsize=9)

        # annotate visible cells
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] >= ROI_ANNOT_MIN_PROP:
                    ax.text(j, i, f"{100*A[i,j]:.1f}%", ha="center", va="center", fontsize=7, color="white")

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Proportion", rotation=90, fontsize=9)

        fig.tight_layout()
        png = outfile(f"roi_{base_tag}_matrix_heatmap.png")
        fig.savefig(png, dpi=300)
        if SAVE_PDF:
            fig.savefig(outfile(f"roi_{base_tag}_matrix_heatmap.pdf"))
        plt.close(fig)
        print(f"[ROI matrix] Wrote: {png}")

    def roi_outputs_from_masks(mask_shift, mask_drift, base_tag: str, caption_suffix: str):
        """Build ROI summary/top and save matrix heatmap + anatomical overlays."""
        ROI_MIN_VOX   = 20
        ROI_TOPN      = 25

        # Summarize per ROI
        roi_records = []
        for name, vec in roi_vecs.items():
            n_roi = int(vec.sum())
            if n_roi < ROI_MIN_VOX: continue
            n_shift = int((vec & mask_shift).sum())
            n_drift = int((vec & mask_drift).sum())
            roi_records.append(dict(
                roi=name,
                n_vox=n_roi,
                n_shift=n_shift,
                n_drift=n_drift,
                prop_shift=(n_shift / n_roi),
                prop_drift=(n_drift / n_roi),
            ))

        roi_df = pd.DataFrame(roi_records).sort_values("roi").reset_index(drop=True)
        roi_df.to_csv(outfile(f"roi_{base_tag}_summary.csv"), index=False)

        top_shift = (roi_df.loc[roi_df["prop_shift"] >= ROI_ANNOT_MIN_PROP]
                            .sort_values("prop_shift", ascending=False)
                            .head(ROI_TOPN))
        top_drift = (roi_df.loc[roi_df["prop_drift"] >= ROI_ANNOT_MIN_PROP]
                            .sort_values("prop_drift", ascending=False)
                            .head(ROI_TOPN))

        top_shift.to_csv(outfile(f"roi_{base_tag}_top_shift.csv"), index=False)
        top_drift.to_csv(outfile(f"roi_{base_tag}_top_drift.csv"), index=False)

        # ROI matrix heatmap
        save_roi_matrix_heatmap(roi_df, top_shift, top_drift, base_tag, caption_suffix)

        # Anatomical overlays
        draw_overlay_mosaic("shift", top_shift, mask_shift, base_tag, caption_suffix)
        draw_overlay_mosaic("drift", top_drift, mask_drift, base_tag, caption_suffix)

    # ==================== Build masks and emit BOTH variants ====================
    # Base masks from z only
    z_shift_full = insert_dropped_voxels(z_bs, dropped_idx, n_vox_full)
    z_drift_full = insert_dropped_voxels(z_bd, dropped_idx, n_vox_full)
    mask_shift_zonly = (np.abs(z_shift_full) >= Z_THR)
    mask_drift_zonly = (np.abs(z_drift_full) >= Z_THR)

    # FDR-gated masks
    rej_shift_full = insert_dropped_voxels(rej_bs.astype(bool), dropped_idx, n_vox_full)
    rej_drift_full = insert_dropped_voxels(rej_bd.astype(bool), dropped_idx, n_vox_full)
    mask_shift_fdr = mask_shift_zonly & rej_shift_full
    mask_drift_fdr = mask_drift_zonly & rej_drift_full

    # --- FDR variant (as before) ---
    base_tag_fdr = f"zthr{str(Z_THR).replace('.','p')}_FDR"
    roi_outputs_from_masks(mask_shift_fdr, mask_drift_fdr, base_tag_fdr, " & FDR")

    # --- NEW: z-only variant (no FDR) ---
    base_tag_no  = f"zthr{str(Z_THR).replace('.','p')}_noFDR"
    roi_outputs_from_masks(mask_shift_zonly, mask_drift_zonly, base_tag_no, "")

    # Final sanity
    print("mask_shift (FDR) voxels:", int(mask_shift_fdr.sum()),
          "mask_drift (FDR) voxels:", int(mask_drift_fdr.sum()))
    print("mask_shift (z-only) voxels:", int(mask_shift_zonly.sum()),
          "mask_drift (z-only) voxels:", int(mask_drift_zonly.sum()))


    # ==================== RFT peak-FWE (voxel-level) ====================
    # Smoothness & resels estimated separately for shift/drift random-effects maps
    rft_shift = _estimate_fwhm_and_resels_from_residuals(bet_s_arr, mean_bs, mask_img, masker)
    rft_drift = _estimate_fwhm_and_resels_from_residuals(bet_d_arr, mean_bd, mask_img, masker)

    with open(outfile("rft_params_shift.json"), "w") as f:
        json.dump(rft_shift, f, indent=2)
    with open(outfile("rft_params_drift.json"), "w") as f:
        json.dump(rft_drift, f, indent=2)

    print(f"[RFT] shift FWHM≈{rft_shift['fwhm_iso_mm']:.2f} mm, resels={tuple(round(x,2) for x in rft_shift['resels'])}")
    print(f"[RFT] drift FWHM≈{rft_drift['fwhm_iso_mm']:.2f} mm, resels={tuple(round(x,2) for x in rft_drift['resels'])}")

    # voxelwise peak-level FWE p-values from |z|
    pFWE_shift = _rft_peak_pvals_from_z(z_bs, rft_shift["resels"])
    pFWE_drift = _rft_peak_pvals_from_z(z_bd, rft_drift["resels"])

    # Save pFWE maps
    nib.save(to_img(pFWE_shift), outfile("pFWE_RFT_voxelwise_unique_shift.nii.gz"))
    nib.save(to_img(pFWE_drift), outfile("pFWE_RFT_voxelwise_unique_drift.nii.gz"))

    # Build FWE masks and save as NIfTI
    rej_shift_rft = (pFWE_shift <= ALPHA).astype(float)
    rej_drift_rft = (pFWE_drift <= ALPHA).astype(float)
    nib.save(to_img(rej_shift_rft), outfile("sig_unique_shift_RFTFWE.nii.gz"))
    nib.save(to_img(rej_drift_rft), outfile("sig_unique_drift_RFTFWE.nii.gz"))

    # z maps masked by peak-FWE
    z_bs_rft = z_bs.copy(); z_bs_rft[pFWE_shift > ALPHA] = 0.0
    z_bd_rft = z_bd.copy(); z_bd_rft[pFWE_drift > ALPHA] = 0.0
    nib.save(to_img(z_bs_rft), outfile("z_unique_shift_RFTFWE_masked.nii.gz"))
    nib.save(to_img(z_bd_rft), outfile("z_unique_drift_RFTFWE_masked.nii.gz"))

    # Visual mosaics of RFT-FWE masked z (same style as your FDR/z-only)
    plot_pair(z_bs_rft, z_bd_rft,
              f"z (shift), RFT peak-FWE≤{ALPHA}", f"z (drift), RFT peak-FWE≤{ALPHA}",
              f"z_shift_bwr_RFTFWE_thr{str(ALPHA).replace('.','p')}.png",
              f"z_drift_bwr_RFTFWE_thr{str(ALPHA).replace('.','p')}.png",
              cmap="bwr_r", threshold=0.0)

    # --- ROI summaries/overlays for the RFT-FWE masks (full-vector masks) ---
    mask_shift_rft_full = insert_dropped_voxels((pFWE_shift <= ALPHA).astype(bool), dropped_idx, n_vox_full)
    mask_drift_rft_full = insert_dropped_voxels((pFWE_drift <= ALPHA).astype(bool), dropped_idx, n_vox_full)

    base_tag_rft = f"RFTFWE_alpha{str(ALPHA).replace('.','p')}"
    roi_outputs_from_masks(mask_shift_rft_full, mask_drift_rft_full, base_tag_rft, " (RFT peak-FWE)")


    print(f"[DONE {tag}] Outputs in: {out_dir}")


# ==================== DRIVER ====================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for layer in LAYERS:
        for rho in RHO_CODES:
            drift_dir = DRIFT_DIR_TEMPLATE.format(rho=rho, layer=layer)
            run_for_drift_variant(SHIFT_DIR, drift_dir, OUT_DIR, layer, rho)

if __name__ == "__main__":
    main()
