#%%writefile evaluation/script9_shap_analysis.py
"""
script9_shap_analysis.py
════════════════════════
SHAP delta-attribution analysis for XGBoost source vs adapted models.

Produces:
  - Bootstrap 95% CIs for per-feature SHAP importance (N=1000 resamples)
  - Delta-Attribution (ΔΦ) between adapted and source XGBoost models
  - shap_ci_results.json
  - shap_delta_results.json

Requires:
  - train/test parquet files from script1
  - XGBoost models saved by script5 (xgb_models dict)
  
Note: XGBoost models must be loaded from script5 outputs.
If you saved them via joblib in script5, load them here.
"""

import numpy as np
import polars as pl
import shap
import json
import joblib
from pathlib import Path


from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import normalize

# ══════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════
BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")
s2        = Path("/kaggle/input/datasets/fatematamanna/ptfiles")
SAVE_PATH = Path("/kaggle/working")

LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
N_BOOTSTRAP = 1000



# ══════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════
print("Loading data...")

import torch
ckpt = torch.load(s2 / "two_stream_models (3).pt",
                  map_location="cpu", weights_only=False)
train_stats = ckpt["train_stats"]

def load_split(name):
    df = pl.read_parquet(BASE_PATH / f"{name}_final_enriched.parquet")
    if "gender" in df.columns:
        df = df.with_columns(
            (pl.col("gender") == "M").cast(pl.Float32).alias("gender_M"))
    for eth in ["WHITE","BLACK","HISPANIC","ASIAN"]:
        if "ethnicity" in df.columns:
            df = df.with_columns(
                (pl.col("ethnicity") == eth).cast(pl.Float32).alias(f"eth_{eth}"))
    return df



def flatten_for_xgb(df, seq_features, treat_features, label_cols, seq_len=6):
    seq_cols = [c for c in seq_features
                if c in df.columns and not c.endswith("_mask")]
    stays = df.sort(["stay_id","hrs_from_admit"])
    stay_ids = (stays.select("stay_id").unique()
                .sort("stay_id")["stay_id"].to_list())
    rows, labels_list = [], []
    for sid in stay_ids:
        s = stays.filter(pl.col("stay_id") == sid)
        seq_vals = s.select(seq_cols).to_numpy().astype(np.float32)
        if seq_vals.shape[0] < seq_len:
            seq_vals = np.vstack([
                seq_vals,
                np.zeros((seq_len - seq_vals.shape[0], seq_vals.shape[1]),
                         dtype=np.float32)])
        else:
            seq_vals = seq_vals[:seq_len]
        last_val = seq_vals[-1]
        mean_val = seq_vals.mean(axis=0)
        std_val  = seq_vals.std(axis=0)
        min_val  = seq_vals.min(axis=0)
        max_val  = seq_vals.max(axis=0)
        treat_val = np.array(
            s.select([c for c in treat_features if c in df.columns]).row(0),
            dtype=np.float32)
        rows.append(np.concatenate([
            last_val, mean_val, std_val, min_val, max_val, treat_val]))
        labels_list.append(
            np.array(s.select(label_cols).row(0), dtype=np.float32))
    X = np.stack(rows)
    Y = np.stack(labels_list)
    feat_names = (
        [f"last_{c}" for c in seq_cols] +
        [f"mean_{c}" for c in seq_cols] +
        [f"std_{c}"  for c in seq_cols] +
        [f"min_{c}"  for c in seq_cols] +
        [f"max_{c}"  for c in seq_cols] +
        [c for c in treat_features if c in df.columns]
    )
    return X, Y, stay_ids, feat_names

test_df  = normalize(load_split("test"),  train_stats)
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

with open(s2 / "eval_post_stays.json") as f:
    import json
    eval_post_stays = json.load(f)

eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))

X_post, Y_post, sids_post, feat_names = flatten_for_xgb(
    eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
X_post[~np.isfinite(X_post)] = 0.0

print(f"Post-drift eval: {X_post.shape[0]} stays, {X_post.shape[1]} features")

# ══════════════════════════════════════
# LOAD XGBOOST MODELS
# ══════════════════════════════════════
# Load models saved by script5
# If you saved with joblib in script5 add those save/load lines here
# If models are only in memory from script5, rerun script5 first
# and save models at the end:
#   for lbl in LABEL_COLS:
#       joblib.dump(xgb_models[lbl],         SAVE_PATH / f"xgb_source_{lbl}.pkl")
#       joblib.dump(xgb_adapted_models[lbl], SAVE_PATH / f"xgb_adapted_{lbl}.pkl")

xgb_models         = {}
xgb_adapted_models = {}
for lbl in LABEL_COLS:
    xgb_models[lbl]         = joblib.load(SAVE_PATH / f"xgb_source_{lbl}.pkl")
    xgb_adapted_models[lbl] = joblib.load(SAVE_PATH / f"xgb_adapted_{lbl}.pkl")
print("✅ XGBoost models loaded")

# ══════════════════════════════════════
# PART 1: BOOTSTRAP CI FOR SHAP IMPORTANCE
# ══════════════════════════════════════
print("\nPart 1: Bootstrapping SHAP CIs (N=1000)...")
rng = np.random.default_rng(42)
shap_ci_results = {}

for label_name in LABEL_COLS:
    print(f"  {label_name}...")
    explainer = shap.TreeExplainer(xgb_adapted_models[label_name])
    shap_vals = explainer.shap_values(X_post)

    n_samples  = len(X_post)
    boot_means = np.zeros((N_BOOTSTRAP, X_post.shape[1]))

    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_samples, size=n_samples)
        boot_means[b] = np.abs(shap_vals[idx]).mean(axis=0)

    ci_lower  = np.percentile(boot_means, 2.5,  axis=0)
    ci_upper  = np.percentile(boot_means, 97.5, axis=0)
    mean_shap = np.abs(shap_vals).mean(axis=0)

    shap_ci_results[label_name] = {
        "features" : feat_names,
        "mean"     : mean_shap.tolist(),
        "ci_lower" : ci_lower.tolist(),
        "ci_upper" : ci_upper.tolist(),
    }

    # Print top 5
    top_idx = np.argsort(mean_shap)[::-1][:5]
    print(f"    Top 5 features (adapted):")
    for i in top_idx:
        print(f"      {feat_names[i]}: "
              f"{mean_shap[i]:.4f} "
              f"[{ci_lower[i]:.4f}, {ci_upper[i]:.4f}]")

with open(SAVE_PATH / "shap_ci_results.json", "w") as f:
    json.dump(shap_ci_results, f, indent=2)
print("✅ Saved shap_ci_results.json")

# ══════════════════════════════════════
# PART 2: DELTA-ATTRIBUTION (ΔΦ)
# ══════════════════════════════════════
print("\nPart 2: Delta-Attribution (ΔΦ = adapted - source)...")
delta_results = {}

for label_name in LABEL_COLS:
    print(f"  {label_name}...")
    explainer_src = shap.TreeExplainer(xgb_models[label_name])
    explainer_adp = shap.TreeExplainer(xgb_adapted_models[label_name])

    shap_src = explainer_src.shap_values(X_post)
    shap_adp = explainer_adp.shap_values(X_post)

    mean_src = np.abs(shap_src).mean(axis=0)
    mean_adp = np.abs(shap_adp).mean(axis=0)
    delta    = mean_adp - mean_src

    # Bootstrap CI for delta
    boot_deltas = np.zeros((N_BOOTSTRAP, X_post.shape[1]))
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(X_post), size=len(X_post))
        boot_deltas[b] = (np.abs(shap_adp[idx]).mean(axis=0) -
                          np.abs(shap_src[idx]).mean(axis=0))

    delta_ci_lower = np.percentile(boot_deltas, 2.5,  axis=0)
    delta_ci_upper = np.percentile(boot_deltas, 97.5, axis=0)

    delta_results[label_name] = {
        "features"      : feat_names,
        "mean_src"      : mean_src.tolist(),
        "mean_adp"      : mean_adp.tolist(),
        "delta"         : delta.tolist(),
        "delta_ci_lower": delta_ci_lower.tolist(),
        "delta_ci_upper": delta_ci_upper.tolist(),
    }

    # Print top deltas
    top_pos = np.argsort(delta)[::-1][:5]
    top_neg = np.argsort(delta)[:5]
    print(f"    Top 5 increasing importance:")
    for i in top_pos:
        print(f"      {feat_names[i]}: "
              f"ΔΦ={delta[i]:.4f} "
              f"[{delta_ci_lower[i]:.4f}, {delta_ci_upper[i]:.4f}]")
    print(f"    Top 5 decreasing importance (amnesia candidates):")
    for i in top_neg:
        print(f"      {feat_names[i]}: "
              f"ΔΦ={delta[i]:.4f} "
              f"[{delta_ci_lower[i]:.4f}, {delta_ci_upper[i]:.4f}]")

with open(SAVE_PATH / "shap_delta_results.json", "w") as f:
    json.dump(delta_results, f, indent=2)
print("✅ Saved shap_delta_results.json")

print("\n✅ script9_shap_analysis.py complete")
print(f"   Outputs: shap_ci_results.json, shap_delta_results.json")