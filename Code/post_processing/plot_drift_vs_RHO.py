# -*- coding: utf-8 -*-
"""
Replicable ROI effects across integration timescales (ρ):
- Loads roi_simes_counts*.csv for ρ ∈ {0.01, 0.05, ..., 0.90}
- Assigns ROIs to "Language hubs" and "DMN+ hubs"
- Optionally reads .nii.gz masks to weight ROI means by voxel count
- Saves two journal-ready panels and a single composite figure (a+b)

Notes:
- No custom colors are set (use Matplotlib defaults).
- Each panel is made as its own figure (no subplots).
- The final composite is built by stitching the two PNGs (not a subplot).
"""

import os, re, glob, warnings, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import nibabel as nib
    HAVE_NIB = True
except Exception:
    HAVE_NIB = False
    warnings.warn("nibabel not available; ROI-size weighting will default to equal weights (1 per ROI).")

# --------------------- PATHS ---------------------
CSV_DIR    = "../../../../DATA/BrainEncode/roi_simes_counts_RHO_79"   # your counts
MASK_DIR   = "../../../../DATA/BrainEncode/MASKS"                  # your .nii.gz masks
OUTPUT_DIR = "./rho_figs"                                         # where to save figures/tables
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional curated ROI lists. If present, these override substring rules.
LANG_LIST_FP = os.path.join(OUTPUT_DIR, "language_rois.txt")   # one ROI filename per line (optional)
DMN_LIST_FP  = os.path.join(OUTPUT_DIR, "dmnplus_rois.txt")    # one ROI filename per line (optional)

# --------------------- RHO map -------------------
RHO_MAP = {
    "001": 0.01, "005": 0.05, "01": 0.10, "02": 0.20, "03": 0.30,
    "04": 0.40, "05": 0.50, "06": 0.60, "07": 0.70, "08": 0.80, "09": 0.90
}

# --------------------- Groups --------------------
# Fallback substring keys (used only if curated lists are absent)
LANGUAGE_KEYS = [
    "Heschl", "Planum_Temporale", "Planum_Polare", "Superior_Temporal_Gyrus",
    "Middle_Temporal_Gyrus", "Inferior_Frontal_Gyrus", "Frontal_Operculum",
    "Central_Opercular_Cortex", "Parietal_Operculum", "Supramarginal_Gyrus",
    "Temporal_Pole", "Insular_Cortex"
]
DMN_PLUS_KEYS = [
    "Angular_Gyrus", "Precuneous_Cortex", "Cingulate_Gyrus_posterior_division",
    "Paracingulate_Gyrus", "Frontal_Medial_Cortex", "Superior_Parietal_Lobule"
]

def _load_curated(fp):
    if os.path.exists(fp):
        items = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                t = line.strip()
                if t and not t.startswith("#"):
                    # accept with or without .nii/.nii.gz extension in file
                    t = t.replace(".nii.gz", "").replace(".nii", "")
                    items.append(t)
        return set(items)
    return None

CURATED_LANG = _load_curated(LANG_LIST_FP)   # set or None
CURATED_DMN  = _load_curated(DMN_LIST_FP)    # set or None

def label_group(roi_basename_noext: str) -> str:
    """Return group label for a given ROI base name (no extension)."""
    if CURATED_LANG is not None or CURATED_DMN is not None:
        if CURATED_LANG is not None and roi_basename_noext in CURATED_LANG:
            return "Language hubs"
        if CURATED_DMN is not None and roi_basename_noext in CURATED_DMN:
            return "DMN+ hubs"
        return "Other"

    # Fallback substring matching:
    for k in LANGUAGE_KEYS:
        if k in roi_basename_noext:
            return "Language hubs"
    for k in DMN_PLUS_KEYS:
        if k in roi_basename_noext:
            return "DMN+ hubs"
    return "Other"

def parse_rho_from_filename(path: str):
    base = os.path.basename(path)
    m = re.search(r"counts(\d+)\.csv$", base)
    if m:
        return RHO_MAP.get(m.group(1), None)
    return None

def load_all_counts(csv_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(csv_dir, "roi_simes_counts*.csv")))
    dfs = []
    for p in paths:
        rho = parse_rho_from_filename(p)
        if rho is None:
            continue
        df = pd.read_csv(p)

        # normalize 'roi' to a base name WITHOUT extension for robust matching
        base_noext = df["roi"].str.replace(".nii.gz", "", regex=False)\
                               .str.replace(".nii", "", regex=False)
        df["roi_base"] = base_noext
        df["group"] = df["roi_base"].apply(label_group)
        df["rho"] = rho
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError("No matching CSV files found. Check CSV_DIR and filenames.")
    return pd.concat(dfs, ignore_index=True)

def count_mask_voxels(mask_path: str) -> int:
    if not HAVE_NIB or not os.path.exists(mask_path):
        return 0
    try:
        data = nib.load(mask_path).get_fdata()
        return int((data != 0).sum())
    except Exception as e:
        warnings.warn(f"Failed to load {mask_path}: {e}")
        return 0

def get_roi_sizes(df: pd.DataFrame, masks_dir: str) -> pd.Series:
    sizes = {}
    missing = []
    for roi_base in sorted(df["roi_base"].unique()):
        # Accept either .nii.gz or .nii
        candidate_gz = os.path.join(masks_dir, roi_base + ".nii.gz")
        candidate_ni = os.path.join(masks_dir, roi_base + ".nii")
        candidate = candidate_gz if os.path.exists(candidate_gz) else candidate_ni
        nvox = count_mask_voxels(candidate)
        if nvox <= 0:
            missing.append(roi_base)
            sizes[roi_base] = 1  # fallback equal weight
        else:
            sizes[roi_base] = nvox
    if missing:
        warnings.warn(f"{len(missing)} ROI masks not found in MASK_DIR; using equal weights for those (e.g., {missing[0]}).")
    return pd.Series(sizes, name="n_voxels")

# --------- Aesthetics: Nature-ish rcParams ----------
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "savefig.dpi": 600,
})

def style_axes(ax):
    # thin, minimalist spines
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(0.5)
    ax.tick_params(width=0.5)

def save_panel_sum(sum_counts: pd.DataFrame, outpath: str):
    # Single-column width ~ 89 mm = 3.5 in
    fig = plt.figure(figsize=(3.5, 2.6))
    ax  = fig.add_subplot(111)
    x   = np.arange(len(sum_counts.index)); width = 0.35

    dmn = sum_counts.get("DMN+ hubs", pd.Series(0, index=sum_counts.index)).values
    lng = sum_counts.get("Language hubs", pd.Series(0, index=sum_counts.index)).values

    ax.bar(x - width/2, dmn, width, label="DMN-PI",color='darkblue')
    ax.bar(x + width/2, lng, width, label="LANG (peri-Sylvian)",color='lightblue')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.2f}" for r in sum_counts.index])
    ax.set_xlabel("Leak parameter $\\rho$")
    ax.set_ylabel("Sum of Simes story counts")
    style_axes(ax)
    ax.legend(frameon=False, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_panel_weighted_mean(weighted_mean: pd.DataFrame, outpath: str):
    fig = plt.figure(figsize=(3.5, 2.6))
    ax  = fig.add_subplot(111)
    x   = np.arange(len(weighted_mean.index))

    dmn = weighted_mean.get("DMN+ hubs", pd.Series(np.nan, index=weighted_mean.index)).values
    lng = weighted_mean.get("Language hubs", pd.Series(np.nan, index=weighted_mean.index)).values

    ax.plot(x, dmn, marker="o", label="DMN-PI",c='darkblue')
    ax.plot(x, lng, marker="o", label="LANG (peri-Sylvian)",c='lightblue')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.2f}" for r in weighted_mean.index])
    ax.set_xlabel("Leak parameter $\\rho$")
    ax.set_ylabel("Mean stories per ROI (size-weighted)")
    style_axes(ax)
    ax.legend(frameon=False, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def stitch_side_by_side(png_left: str, png_right: str, out_png: str, title=None):
    """Create a composite image by concatenating two PNGs side-by-side with small gutter and panel labels."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        # Pillow not available; fallback: read as arrays and concatenate
        import matplotlib.image as mpimg
        import imageio.v2 as imageio

        A = mpimg.imread(png_left)
        B = mpimg.imread(png_right)

        # Match heights
        h = max(A.shape[0], B.shape[0])
        def pad_to_h(img, h):
            if img.shape[0] == h: return img
            pad = h - img.shape[0]
            top = pad // 2
            bottom = pad - top
            return np.pad(img, ((top,bottom),(0,0),(0,0)), mode="constant", constant_values=1.0)
        A = pad_to_h(A, h)
        B = pad_to_h(B, h)

        gutter = np.ones((h, int(0.05*h), 4), dtype=A.dtype)
        composite = np.concatenate([A, gutter, B], axis=1)
        imageio.imwrite(out_png, composite)
        return

    # With Pillow (preferred)
    A = Image.open(png_left).convert("RGB")
    B = Image.open(png_right).convert("RGB")

    h = max(A.height, B.height)
    # pad to same height
    def pad_h(img, target_h):
        if img.height == target_h:
            return img
        top = (target_h - img.height) // 2
        new = Image.new("RGB", (img.width, target_h), (255,255,255))
        new.paste(img, (0, top))
        return new

    A = pad_h(A, h)
    B = pad_h(B, h)

    gutter_px = int(0.08 * h)
    W = A.width + gutter_px + B.width
    H = h + int(0.18 * h)  # room for title/panel labels
    canvas = Image.new("RGB", (W, H), (255,255,255))

    # Title
    draw = ImageDraw.Draw(canvas)
    try:
        # Use default PIL font if system fonts unavailable
        font = ImageFont.load_default()
    except Exception:
        font = None

    if title:
        draw.text((10, 8), title, fill=(0,0,0), font=font)

    # Paste panels
    y0 = int(0.14 * H)
    canvas.paste(A, (0, y0))
    canvas.paste(B, (A.width + gutter_px, y0))

    # Panel labels
    draw.text((6, y0 + 4), "a", fill=(0,0,0), font=font)
    draw.text((A.width + gutter_px + 6, y0 + 4), "b", fill=(0,0,0), font=font)

    canvas.save(out_png)

def main():
    # Load data
    full = load_all_counts(CSV_DIR)

    # ROI sizes (voxel counts) keyed by ROI base name
    roi_sizes = get_roi_sizes(full, MASK_DIR)
    full = full.merge(roi_sizes.rename("n_voxels"),
                      left_on="roi_base", right_index=True, how="left")
    full["n_voxels"] = full["n_voxels"].fillna(1)

    # -------- Aggregations --------
    # 1) Raw sums across ROIs
    sum_counts = (
        full.groupby(["rho", "group"])["n_stories_sig"].sum()
            .reset_index().pivot(index="rho", columns="group", values="n_stories_sig")
            .sort_index().fillna(0)
    )

    # 2) Mean per ROI (size-weighted)
    def _weighted_mean(sub):
        w = sub["n_voxels"].values.astype(float)
        x = sub["n_stories_sig"].values.astype(float)
        return float(np.average(x, weights=w)) if w.sum() > 0 else np.nan

    weighted_mean = (
        full.groupby(["rho", "group"]).apply(_weighted_mean)
            .reset_index(name="weighted_mean")
            .pivot(index="rho", columns="group", values="weighted_mean")
            .sort_index()
    )

    # Save tidy summary for supplement
    summary = pd.concat(
        {
            "sum_counts": sum_counts,
            "weighted_mean_per_roi": weighted_mean
        },
        axis=1
    )
    summary_fp = os.path.join(OUTPUT_DIR, "rho_simes_counts_summary.csv")
    summary.to_csv(summary_fp, float_format="%.3f")

    # -------- Make panels --------
    out_sum = os.path.join(OUTPUT_DIR, "rho_simes_counts_sum.png")
    out_mean = os.path.join(OUTPUT_DIR, "rho_simes_counts_mean_weighted.png")
    save_panel_sum(sum_counts, out_sum)
    save_panel_weighted_mean(weighted_mean, out_mean)

    # -------- Composite (a+b) ----
    out_comp = os.path.join(OUTPUT_DIR, "rho_simes_counts_COMPOSITE.png")
    stitch_side_by_side(out_sum, out_mean, out_comp,
                        title="Replicable ROI effects across integration timescales")

    # Persist ROI membership used
    groups_txt = []
    for group_name in ["LANG (peri-Sylvian)", "DMN-PI"]:
        members = sorted(full.loc[full["group"] == group_name, "roi_base"].unique())
        groups_txt.append(f"# {group_name} ({len(members)} ROIs)\n" + "\n".join(members) + "\n")
    with open(os.path.join(OUTPUT_DIR, "roi_groups_used.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(groups_txt))

    print("Saved:")
    print("  ", out_sum)
    print("  ", out_mean)
    print("  ", out_comp)
    print("  ", summary_fp)
    print("  ", os.path.join(OUTPUT_DIR, "roi_groups_used.txt"))

if __name__ == "__main__":
    main()
