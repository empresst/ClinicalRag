"""
script15_septic_shock_24_cases.py
══════════════════════════════════
Identifies the 24 septic shock cases caught by Run B and missed by
XGBoost-adapted. Extracts clinical features and SHAP attributions
to demonstrate systematic pattern.

Requires:
  - train/test parquet files from script1
  - two_stream_models.pt from script2
  - eval_post_stays.json from script3
  - xgb_predictions_corrected.npz from script6
  - post_drift_predictions_with_runD.npz from script6
  - xgb_adapted_label_septic_shock.pkl from script5
"""

import numpy as np
import polars as pl
import shap
import json
import joblib
import torch
import torch.nn as nn
from pathlib import Path

# ══════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════
BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")
s2        = Path("/kaggle/input/datasets/fatematamanna/ptfiles")
SAVE_PATH = Path("/kaggle/working")

LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
SEQ_LEN    = 6

SEQ_FEATURES = [
    "heart_rate","sbp_noninvasive","dbp_noninvasive","sbp_invasive","dbp_invasive",
    "map_invasive","temperature_c","spo2","resp_rate",
    "creatinine","wbc","platelets","lactate","bun","bilirubin_total","glucose",
    "hematocrit","potassium","sodium","troponin_t","ph_venous","pco2_venous",
    "base_excess","rbc","chloride","calcium",
    "urine_output","urine_output_ml_kg_hr","weight",
    "heart_rate_time_delta","map_invasive_time_delta","lactate_time_delta",
    "creatinine_baseline","creatinine_delta","creatinine_ratio",
    "lactate_baseline","lactate_delta","lactate_ratio",
    "bun_baseline","bun_delta","bun_ratio",
    "glucose_baseline","glucose_delta","glucose_ratio",
    "bilirubin_total_baseline","bilirubin_total_delta","bilirubin_total_ratio",
    "resp_rate_rollmean_3","resp_rate_rollstd_3","spo2_rollmean_6","spo2_rollstd_4",
    "heart_rate_mask","sbp_noninvasive_mask","dbp_noninvasive_mask",
    "sbp_invasive_mask","dbp_invasive_mask","map_invasive_mask",
    "temperature_c_mask","spo2_mask","resp_rate_mask",
    "creatinine_mask","wbc_mask","platelets_mask","lactate_mask","bun_mask",
    "bilirubin_total_mask","glucose_mask","hematocrit_mask","potassium_mask",
    "sodium_mask","troponin_t_mask","ph_venous_mask","pco2_venous_mask",
    "base_excess_mask","rbc_mask","chloride_mask","calcium_mask",
    "urine_output_mask","urine_output_ml_kg_hr_mask","weight_mask",
]

TREATMENT_FEATURES = [
    "total_crystalloid_ml","has_norepinephrine_obs","has_phenylephrine_obs",
    "has_dopamine_obs","has_vasopressin_obs","time_to_first_vaso_hrs",
    "early_steroid","early_antibiotic","n_distinct_meds",
    "steroid_ordered","time_to_first_abx_order_hrs",
    "max_fio2_obs","mean_fio2_obs","max_peep_obs","max_tidal_volume_obs",
    "high_fio2_flag","on_peep_flag",
    "has_propofol_midaz_obs","total_sedation_dose_obs",
    "age","gender_M","eth_WHITE","eth_BLACK","eth_HISPANIC","eth_ASIAN",
]

# ══════════════════════════════════════
# LOAD PREDICTIONS
# ══════════════════════════════════════
print("Loading prediction arrays...")
pred_data = np.load(SAVE_PATH / "post_drift_predictions_with_runD.npz",
                    allow_pickle=True)
stay_ids  = pred_data["stay_ids"].tolist()
labels    = pred_data["labels"]
probs_b   = pred_data["probs_run_b"]

xgb_data          = np.load(SAVE_PATH / "xgb_predictions_corrected.npz")
probs_xgb_adapted = xgb_data["probs_xgb_adapted"]
xgb_sids          = xgb_data["stay_ids"].tolist()

# Verify ordering
assert xgb_sids == stay_ids, "Ordering mismatch between prediction arrays"
print(f"✅ {len(stay_ids)} stays loaded and order verified")

# ══════════════════════════════════════
# IDENTIFY 24 CASES
# ══════════════════════════════════════
shock_idx = 2  # label_septic_shock is index 2

caught_by_b = []
for i, sid in enumerate(stay_ids):
    true_label = labels[i, shock_idx]
    pb = probs_b[i, shock_idx]
    px = probs_xgb_adapted[i, shock_idx]
    if true_label == 1 and pb >= 0.50 and px < 0.10:
        caught_by_b.append({
            "stay_id"         : sid,
            "prob_run_b"      : float(pb),
            "prob_xgb_adapted": float(px),
            "true_label"      : int(true_label)
        })

print(f"\nFound {len(caught_by_b)} cases where Run B catches and XGBoost misses")
for c in caught_by_b:
    print(f"  stay={c['stay_id']} | "
          f"RunB={c['prob_run_b']:.3f} | "
          f"XGB={c['prob_xgb_adapted']:.4f}")

# ══════════════════════════════════════
# LOAD CLINICAL FEATURES
# ══════════════════════════════════════
print("\nLoading clinical features...")

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

def normalize(df, stats):
    exprs = []
    for c, (mu, sd) in stats.items():
        if c in df.columns:
            if c.endswith("_mask") or c in {
                "gender_M","early_steroid","early_antibiotic",
                "steroid_ordered","high_fio2_flag","on_peep_flag"
            } or c.startswith(("has_","eth_")):
                exprs.append(pl.col(c).cast(pl.Float32))
            else:
                exprs.append(
                    ((pl.col(c).cast(pl.Float64)-mu)/sd
                     ).cast(pl.Float32).alias(c))
    return df.with_columns(exprs)

def flatten_for_xgb(df, seq_features, treat_features,
                    label_cols, seq_len=6):
    seq_cols = [c for c in seq_features
                if c in df.columns and not c.endswith("_mask")]
    stays    = df.sort(["stay_id","hrs_from_admit"])
    sids     = (stays.select("stay_id").unique()
                .sort("stay_id")["stay_id"].to_list())
    rows, labels_list = [], []
    for sid in sids:
        s        = stays.filter(pl.col("stay_id") == sid)
        seq_vals = s.select(seq_cols).to_numpy().astype(np.float32)
        if seq_vals.shape[0] < seq_len:
            seq_vals = np.vstack([
                seq_vals,
                np.zeros((seq_len-seq_vals.shape[0], seq_vals.shape[1]),
                         dtype=np.float32)])
        else:
            seq_vals = seq_vals[:seq_len]
        last_val  = seq_vals[-1]
        mean_val  = seq_vals.mean(axis=0)
        std_val   = seq_vals.std(axis=0)
        min_val   = seq_vals.min(axis=0)
        max_val   = seq_vals.max(axis=0)
        treat_val = np.array(
            s.select([c for c in treat_features
                      if c in df.columns]).row(0), dtype=np.float32)
        rows.append(np.concatenate([
            last_val, mean_val, std_val, min_val, max_val, treat_val]))
        labels_list.append(
            np.array(s.select(label_cols).row(0), dtype=np.float32))
    X         = np.stack(rows)
    Y         = np.stack(labels_list)
    feat_names = (
        [f"last_{c}" for c in seq_cols] +
        [f"mean_{c}" for c in seq_cols] +
        [f"std_{c}"  for c in seq_cols] +
        [f"min_{c}"  for c in seq_cols] +
        [f"max_{c}"  for c in seq_cols] +
        [c for c in treat_features if c in df.columns]
    )
    return X, Y, sids, feat_names

test_df   = normalize(load_split("test"), train_stats)
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

caught_ids  = [c["stay_id"] for c in caught_by_b]
caught_df   = test_post.filter(pl.col("stay_id").is_in(caught_ids))

# One row per stay for clinical summary
caught_h0 = caught_df.filter(pl.col("hrs_from_admit") == 0).sort("stay_id")

# ══════════════════════════════════════
# CLINICAL FEATURE SUMMARY
# ══════════════════════════════════════
key_features = [
    "stay_id", "age", "gender",
    "lactate", "lactate_baseline", "lactate_delta",
    "map_invasive", "heart_rate", "resp_rate",
    "creatinine", "wbc",
    "has_norepinephrine_obs", "has_vasopressin_obs",
    "total_crystalloid_ml", "time_to_first_vaso_hrs",
    "early_antibiotic"
]
available = [c for c in key_features if c in caught_h0.columns]

print("\n=== CLINICAL FEATURES OF CAUGHT PATIENTS ===")
print(caught_h0.select(available))

print("\n=== SUMMARY STATISTICS ACROSS 24 PATIENTS ===")
for feat in ["lactate", "map_invasive", "heart_rate",
             "total_crystalloid_ml", "time_to_first_vaso_hrs"]:
    if feat in caught_h0.columns:
        vals = caught_h0[feat].drop_nulls()
        print(f"  {feat}: mean={vals.mean():.3f} std={vals.std():.3f}")

if "has_norepinephrine_obs" in caught_h0.columns:
    pct = caught_h0["has_norepinephrine_obs"].mean() * 100
    print(f"  has_norepinephrine: {pct:.1f}%")

if "early_antibiotic" in caught_h0.columns:
    pct = caught_h0["early_antibiotic"].mean() * 100
    print(f"  early_antibiotic:   {pct:.1f}%")

# ══════════════════════════════════════
# SHAP ON XGBoost FOR THESE 24 PATIENTS
# ══════════════════════════════════════
print("\nRunning SHAP on XGBoost for the 24 caught patients...")

X_caught, Y_caught, sids_caught, feat_names = flatten_for_xgb(
    caught_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
X_caught[~np.isfinite(X_caught)] = 0.0

model_xgb_shock = joblib.load(
    SAVE_PATH / "xgb_adapted_label_septic_shock.pkl")
explainer   = shap.TreeExplainer(model_xgb_shock)
shap_values = explainer.shap_values(X_caught)

mean_shap = np.abs(shap_values).mean(axis=0)
shap_df_data = sorted(
    zip(feat_names, mean_shap),
    key=lambda x: x[1], reverse=True)

print("\nTop 10 features XGBoost attends to for these 24 patients:")
for feat, val in shap_df_data[:10]:
    print(f"  {feat}: {val:.4f}")

# Check lactate and MAP specifically
print("\nLactate features:")
for feat, val in shap_df_data:
    if "lactate" in feat:
        print(f"  {feat}: {val:.4f}")

print("\nMAP features:")
for feat, val in shap_df_data:
    if "map" in feat:
        print(f"  {feat}: {val:.4f}")

# ══════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════
results = {
    "n_cases"   : len(caught_by_b),
    "cases"     : caught_by_b,
    "shap_top10": [{"feature": f, "mean_abs_shap": float(v)}
                   for f, v in shap_df_data[:10]]
}
with open(SAVE_PATH / "septic_shock_24_cases.json", "w") as f:
    json.dump(results, f, indent=2)
print("\n✅ Saved septic_shock_24_cases.json")
print("✅ script15_septic_shock_24_cases.py complete")