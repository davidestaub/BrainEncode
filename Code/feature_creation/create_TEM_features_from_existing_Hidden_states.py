#!/usr/bin/env python3
"""
make_extra_features.py   (patched 2025‑05‑13)

Creates seven TR‑aligned metrics from layer‑32 hidden states
and saves them as (N_TR, 1) arrays next to the existing .npz files.

Existing metrics
  • Velocity
  • Context novelty
  • Diff energy
  • Similarity‑entropy

New (TEM‑inspired)
  • Context drift magnitude
  • Reinstatement strength
  • Context pattern‑separation index
"""

# ---------------------------------------------------------------------
# standard libs
# ---------------------------------------------------------------------
import numpy as np
from pathlib import Path
from itertools import product

# core SciPy / scikit‑learn
from scipy.ndimage import uniform_filter1d
from scipy.signal import detrend
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------
# constants that never vary in the sweep
# ---------------------------------------------------------------------
ROOT      = Path("data/ours/extracted_text_features")
LAYER     = 79
LOOKBACK1 = 256
LOOKBACK2 = 512
EPS       = 1e-9

# ---------- helpers --------------------------------------------------
def load_npz(folder: Path, pattern: str) -> np.ndarray:
    file = next(folder.glob(pattern))
    return np.load(file)["features"]          # (N_TR, D)

def save_feature(folder: Path, fname: str, vec: np.ndarray):
    folder.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(folder / fname, features=vec[:, None].astype(np.float32))

# ---------------------------------------------------------------------
# PARAM GRID  – tweak to narrow / widen the sweep
# ---------------------------------------------------------------------
PARAM_GRID = {
    "ROLL_K":           [4],              # novelty window (TRs)
    "ENT_K":            [5],                 # similarity‑entropy window
    "BETA":             [10],            # soft‑max temperature
    "RHO":              [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],          # leaky integrator decay
    "USE_PCA":          [False],       # raw vs. reduced
    "PCA_DIM":          [20],            # components when PCA=True
    "ROLLING_WINDOW":   [100],          # pattern‑sep horizon
    "WIN_SKIP":         [10],                # skip recent TRs
    "CLIP_NEG_COS":     [True],              # bound similarity ∈[0,1]
    "DETREND_STABILIZE":[False],       # applies to DRIFT only
}

# ---------------------------------------------------------------------
# which knobs affect which feature?
# ---------------------------------------------------------------------
DEPENDENCIES = {
    "velocity":          {"USE_PCA", "PCA_DIM"},
    "novelty":           {"ROLL_K", "USE_PCA", "PCA_DIM"},
    "diff_energy":       {"USE_PCA", "PCA_DIM"},
    "similarity_entropy":{"ENT_K", "BETA", "USE_PCA", "PCA_DIM"},
    "drift":             {"RHO", "USE_PCA", "PCA_DIM", "DETREND_STABILIZE"},
    "reinstatement":     {"RHO", "USE_PCA", "PCA_DIM"},
    "separation":        {"RHO", "USE_PCA", "PCA_DIM",
                          "ROLLING_WINDOW", "WIN_SKIP", "CLIP_NEG_COS"},
}

# ---------------------------------------------------------------------
# tag() – builds suffix strings from *relevant* params only
# ---------------------------------------------------------------------
def tag(params: dict, feature: str) -> str:
    keys = DEPENDENCIES[feature]
    if not keys:
        return ""
    parts = []
    for k in sorted(keys):
        # PCA_DIM is irrelevant when USE_PCA=False
        if k == "PCA_DIM" and not params["USE_PCA"]:
            continue
        v = params[k]
        if isinstance(v, bool):
            parts.append(f"{k}{int(v)}")
        elif isinstance(v, float):
            parts.append(f"{k}{str(v).replace('.','')}")
        else:
            parts.append(f"{k}{v}")
    return "_" + "_".join(parts)

# =====================================================================
# MAIN SWEEP LOOP
# =====================================================================
for P in product(*PARAM_GRID.values()):
    params = dict(zip(PARAM_GRID.keys(), P))
    print("\n>> building features with:", params)

    # unpack for convenience
    (ROLL_K, ENT_K, BETA, RHO, USE_PCA, PCA_DIM,
     ROLLING_WINDOW, WIN_SKIP, CLIP_NEG_COS, DETREND_STABILIZE) = (
        params["ROLL_K"],  params["ENT_K"],  params["BETA"],  params["RHO"],
        params["USE_PCA"], params["PCA_DIM"],
        params["ROLLING_WINDOW"], params["WIN_SKIP"], params["CLIP_NEG_COS"],
        params["DETREND_STABILIZE"]
    )

    # ------------------------------------------------------------------
    # iterate over stories
    # ------------------------------------------------------------------
    for story_dir in sorted(ROOT.iterdir()):
        if not story_dir.is_dir():
            continue
        print(f"[{story_dir.name}]")

        # ---- load original hidden states (full dimension) ------------
        H_full = load_npz(
            story_dir,
            f"final_outputs_hidden_states_layer{LAYER}"
            f"_context_{LOOKBACK1}_{LOOKBACK2}.npz"
        )                                   # (N, D0)
        # ---- optional PCA --------------------------------------------
        if USE_PCA and H_full.shape[1] > PCA_DIM:
            H = PCA(n_components=PCA_DIM, random_state=0).fit_transform(H_full)
        else:
            H = H_full
        N = H.shape[0]

        # ---- difference vector after any PCA -------------------------
        D = np.diff(H, axis=0, prepend=H[:1])
        # --------------------------------------------------------------
        # build leaky context g_t (or raw H)
        # --------------------------------------------------------------
        if RHO < 1.0:
            g = np.empty_like(H)
            g[0] = H[0]
            for t in range(1, N):
                g[t] = RHO * g[t-1] + (1.0 - RHO) * H[t]
        else:
            g = H
        g_norm = np.linalg.norm(g, axis=1) + EPS

        # ==============================================================
        # 1) VELOCITY
        # ==============================================================
        featname = "velocity"
        suffix   = tag(params, featname)
        fname    = story_dir / (
            f"final_outputs_{featname}{suffix}_layer{LAYER}"
            f"_context_{LOOKBACK1}_{LOOKBACK2}.npz")
        if not fname.exists():
            vel = np.linalg.norm(D, axis=1).astype(np.float32)
            save_feature(story_dir, fname.name, vel)

        # ==============================================================
        # 2) CONTEXT NOVELTY
        # ==============================================================
        featname = "novelty"
        suffix   = tag(params, featname)
        fname    = story_dir / (
            f"final_outputs_context_{featname}{suffix}_layer{LAYER}"
            f"_context_{LOOKBACK1}_{LOOKBACK2}.npz")
        if not fname.exists():
            mu  = uniform_filter1d(H, size=ROLL_K, axis=0, origin=-ROLL_K//2)
            nov = 1.0 - np.sum(H*mu,1) / (np.linalg.norm(H,1)*np.linalg.norm(mu,1)+EPS)
            nov[:ROLL_K] = 0.0
            save_feature(story_dir, fname.name, nov.astype(np.float32))

        # ==============================================================
        # 3) DIFF ENERGY
        # ==============================================================
        featname = "diff_energy"
        suffix   = tag(params, featname)
        fname    = story_dir / (
            f"final_outputs_{featname}{suffix}_layer{LAYER}"
            f"_context_{LOOKBACK1}_{LOOKBACK2}.npz")
        if not fname.exists():
            ene = np.mean(np.abs(D), axis=1).astype(np.float32)
            save_feature(story_dir, fname.name, ene)

        # ==============================================================
        # 4) SIMILARITY ENTROPY
        # ==============================================================
        featname = "similarity_entropy"
        suffix   = tag(params, featname)
        fname    = story_dir / (
            f"final_outputs_{featname}{suffix}_layer{LAYER}"
            f"_context_{LOOKBACK1}_{LOOKBACK2}.npz")
        if not fname.exists():
            sim = cosine_similarity(H, H).astype(np.float32)
            ent = np.zeros(N, dtype=np.float32)
            for t in range(ENT_K, N):
                s = sim[t, t-ENT_K:t] * BETA
                p = np.exp(s - s.max());  p /= p.sum() + EPS
                ent[t] = -(p*np.log(p+EPS)).sum()
            save_feature(story_dir, fname.name, ent)

        # ==============================================================
        # 5) DRIFT  (optional detrend + stabilise)
        # ==============================================================
        featname = "drift"
        suffix   = tag(params, featname)
        fname    = story_dir / (
            f"final_outputs_context_{featname}{suffix}_layer{LAYER}"
            f"_context_{LOOKBACK1}_{LOOKBACK2}.npz")
        if not fname.exists():
            drift = np.linalg.norm(np.diff(g, axis=0, prepend=g[:1]), axis=1)
            if DETREND_STABILIZE:
                drift = detrend(drift, type='linear')
                drift = np.maximum(drift, 0) + 1e-6
                drift = np.log(drift)
                drift = (drift - drift.mean()) / (drift.std(ddof=0) + EPS)
            save_feature(story_dir, fname.name, drift.astype(np.float32))

        # ==============================================================
        # 6) REINSTATEMENT STRENGTH
        # ==============================================================
        featname = "reinstatement"
        suffix   = tag(params, featname)
        fname    = story_dir / (
            f"final_outputs_{featname}{suffix}_layer{LAYER}"
            f"_context_{LOOKBACK1}_{LOOKBACK2}.npz")
        if not fname.exists():
            dot = np.sum(g * H, axis=1)
            norms = g_norm * np.linalg.norm(H, axis=1) + EPS
            reinst = (dot / norms).astype(np.float32)
            save_feature(story_dir, fname.name, reinst)

        # ==============================================================
        # 7) PATTERN‑SEPARATION
        # ==============================================================
        featname = "separation"
        suffix   = tag(params, featname)
        fname    = story_dir / (
            f"final_outputs_context_{featname}{suffix}_layer{LAYER}"
            f"_context_{LOOKBACK1}_{LOOKBACK2}.npz")
        if not fname.exists():
            sep = np.zeros(N, dtype=np.float32)
            for t in range(1, N):
                start = max(0, t - ROLLING_WINDOW)
                stop  = t - WIN_SKIP
                if stop <= start:
                    continue
                sims = (g[t] @ g[start:stop].T) / (g_norm[t] * g_norm[start:stop])
                if CLIP_NEG_COS:
                    sims = np.clip(sims, 0.0, 1.0)
                sep[t] = 1.0 - np.max(sims)
            #WINSOR clipping because otherwise huge spike in the beginning
            # winsorise using the 2nd–98th percentiles *excluding* the first element
            lo, hi = np.quantile(sep[1:], [0.02, 0.98])  # returns two scalars
            sep = np.clip(sep, lo, hi)
            save_feature(story_dir, fname.name, sep)

print("\nAll combinations finished — features written only when needed.")
