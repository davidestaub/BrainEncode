import pandas as pd



path = 'pos_summary_all_metrics.csv'

# Detect and skip a leading "sep=" line if present
with open(path, 'r', encoding='utf-8') as f:
    first = f.readline()
skip = 1 if first.strip().lower().startswith('sep=') else 0

df = pd.read_csv(path, sep=None, engine='python', skiprows=skip)

# If metric/story/roi became the index, bring them back as columns (optional)
if isinstance(df.index, pd.MultiIndex) or df.index.name is not None:
    df = df.reset_index()

# Strip any stray spaces from column names
df.columns = df.columns.astype(str).str.strip()

print(df.columns.tolist())        # sanity check
print(df.head())

# ------------------------------------------------------------------
# 1) Load the data
# ------------------------------------------------------------------
#df = pd.read_csv("pos_summary_all_metrics.csv",delimiter=";")
#print(df)
#print(df['max_corr'])

# ------------------------------------------------------------------
# 2) ROIs where *hidden_states* metric is strong (max_corr > 0.2)
# ------------------------------------------------------------------
#hidden_mask = df["metric"].str.contains("hidden_states", regex=False,na=False)
strong_mask = df['max_corr'] > 0.15

hidden_hits   = df.loc[strong_mask, ["roi", "max_corr"]]
hidden_rois   = set(hidden_hits["roi"])

print("ROIs with hidden_states metric > 0.2:")
print(sorted(hidden_rois))

# ------------------------------------------------------------------
# 3) ROIs + METRICS where any *other* metric is strong (> 0.2)
# ------------------------------------------------------------------
other_hits = df.loc[strong_mask, ["roi", "metric", "max_corr"]]

print("\nNon‑hidden_states metrics > 0.2 (roi, metric, max_corr):")
print(other_hits.sort_values(["roi", "max_corr"], ascending=[True, False]))

# ------------------------------------------------------------------
# 4) ROIs that are strong for a non‑hidden_states metric
#    **but NOT** strong for any hidden_states metric
# ------------------------------------------------------------------
exclusive_rois   = set(other_hits["roi"]) - hidden_rois
exclusive_hits   = other_hits[other_hits["roi"].isin(exclusive_rois)]

print("\nROIs with a strong non‑hidden metric but NO strong hidden_states metric:")
print(sorted(exclusive_rois))

# If you also want to see which specific non‑hidden metric(s) drove each ROI:
print("\nDetails for those exclusive ROIs:")
print(exclusive_hits.sort_values(["roi", "max_corr"], ascending=[True, False]))


