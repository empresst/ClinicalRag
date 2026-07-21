%%writefile evaluation/script_mc4_delta_attribution.py
"""
script_mc4_delta_attribution.py
════════════════════════════════════════════════════════════════════
MC4 Reviewer Response — Population-level Δ-Attribution table.

Computes mean |SHAP| per feature across ALL 5,749 post-drift
held-out patients for XGBoost-source and XGBoost-adapted, then:
  ΔΦ_j = mean|SHAP_adapted_j| - mean|SHAP_source_j|

Features are separated into physiological vs treatment streams
using TREATMENT_FEATURES. Bootstrap CIs (N=1000, percentile)
are computed on each ΔΦ_j by resampling patients.

Top-10 features by |ΔΦ| are reported per stream per label,
producing the table needed to support the biological amnesia claim
at the population level rather than single-patient level.

Run AFTER script5 (needs xgb_source_*.pkl, xgb_adapted_*.pkl,
xgb_predictions_corrected.npz, eval_split.json, parquet data).

Outputs:
  delta_attribution_table.json   → full per-feature ΔΦ + CIs
  delta_attribution_summary.txt  → paper-ready table for §3.6
"""

import json
import warnings
import joblib
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

# ── same constants as script5 ─────────────────────────────────────────────────
SEED           = 42
SEQ_LEN        = 6
N_BOOTSTRAP    = 1000
BOOTSTRAP_SEED = 42
TOP_K          = 10   # top features to report per stream per label

LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH  = Path("/kaggle/working")

np.random.seed(SEED)

# ── TREATMENT_FEATURES — must exactly match script5 / utils/constants.py ──────
# These are the 12 treatment features used in the two-stream architecture.
# All other XGBoost flattened features are physiological.
TREATMENT_FEATURES = [
    "total_crystalloid_ml",
    "age",
    "n_distinct_meds",
    "time_to_first_abx_order_hrs",
    "early_antibiotic",
    "early_steroid",
    "steroid_ordered",
    "has_insulin_infusion_obs",
    "has_blood_products_obs",
    "has_rrt_obs",
    "gender_M",
    "total_prbc_ml",
]

# ── LOAD EVAL SPLIT ───────────────────────────────────────────────────────────
print("Loading eval split...")
with open(SAVE_PATH / "eval_split.json") as f:
    split = json.load(f)

eval_post_stays = list(map(int, split["eval_post_stays"]))
pre_cp_stays    = list(map(int, split["pre_cp_stays"]))
post_cp_stays   = list(map(int, split["post_cp_stays"]))
drift_tag       = split.get("drift_tag", "2020 - 2022")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading parquet data...")
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES as TF_IMPORTED
from utils.data_utils import load_enriched_split, normalize

# Use imported TREATMENT_FEATURES from utils to guarantee match with script5
TREATMENT_FEATURES = TF_IMPORTED

ckpt_path = SAVE_PATH / "two_stream_models.pt"
import torch
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
train_stats = ckpt["train_stats"]

test_df  = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
test_df  = normalize(test_df, train_stats)

test_post    = test_df.filter(pl.col("stay_id").is_in(post_cp_stays))
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))

print(f"  Post-drift held-out: {eval_post_df['stay_id'].n_unique()} stays")

# ── FLATTEN — identical to script5's flatten_for_xgb ─────────────────────────
def flatten_for_xgb(df, seq_features, treat_features, label_cols, seq_len=6):
    seq_cols   = [c for c in seq_features   if c in df.columns and not c.endswith("_mask")]
    treat_cols = [c for c in treat_features if c in df.columns]

    stays    = df.sort(["stay_id", "hrs_from_admit"])
    stay_ids = stays.select("stay_id").unique().sort("stay_id")["stay_id"].to_list()

    flat_features, flat_labels, flat_stay_ids = [], [], []

    for sid in stay_ids:
        s        = stays.filter(pl.col("stay_id") == sid)
        seq_vals = s.select(seq_cols).to_numpy().astype(np.float32)
        if seq_vals.shape[0] < seq_len:
            pad      = np.zeros((seq_len - seq_vals.shape[0], seq_vals.shape[1]), dtype=np.float32)
            seq_vals = np.vstack([seq_vals, pad])
        else:
            seq_vals = seq_vals[:seq_len]

        last_val  = seq_vals[-1]
        mean_val  = np.nanmean(seq_vals, axis=0)
        std_val   = np.nanstd(seq_vals,  axis=0)
        min_val   = np.nanmin(seq_vals,  axis=0)
        max_val   = np.nanmax(seq_vals,  axis=0)
        treat_val = np.array(s.select(treat_cols).row(0), dtype=np.float32)

        flat_vec = np.concatenate([last_val, mean_val, std_val, min_val, max_val, treat_val])
        flat_features.append(flat_vec)
        flat_labels.append(np.array(s.select(label_cols).row(0), dtype=np.float32))
        flat_stay_ids.append(sid)

    X = np.stack(flat_features)
    Y = np.stack(flat_labels)

    names = []
    for prefix in ["last", "mean", "std", "min", "max"]:
        names += [f"{prefix}_{c}" for c in seq_cols]
    names += treat_cols

    return X, Y, flat_stay_ids, names


print("Flattening features...")
X_post, Y_post, post_stay_ids, feat_names = flatten_for_xgb(
    eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
)
X_post[~np.isfinite(X_post)] = 0.0
print(f"  X_post shape: {X_post.shape}  ({len(feat_names)} features)")

# ── BUILD STREAM MASK ─────────────────────────────────────────────────────────
# A flattened feature is treatment if its base column name is in TREATMENT_FEATURES
treat_set  = set(TREATMENT_FEATURES)
is_treatment = np.zeros(len(feat_names), dtype=bool)
for i, name in enumerate(feat_names):
    # strip the stat prefix: "last_X", "mean_X" → base = everything after first "_"
    base = "_".join(name.split("_")[1:]) if "_" in name else name
    if base in treat_set or name in treat_set:
        is_treatment[i] = True

is_physio = ~is_treatment
print(f"  Physiological features: {is_physio.sum()}")
print(f"  Treatment features:     {is_treatment.sum()}")

# ── LOAD SAVED XGB MODELS ────────────────────────────────────────────────────
print("\nLoading XGBoost models...")
import shap as shap_lib

xgb_src_models = {}
xgb_adp_models = {}
for lbl in LABEL_COLS:
    xgb_src_models[lbl] = joblib.load(SAVE_PATH / f"xgb_source_{lbl}.pkl")
    xgb_adp_models[lbl] = joblib.load(SAVE_PATH / f"xgb_adapted_{lbl}.pkl")
print("  Models loaded.")

# ── POPULATION-LEVEL SHAP — batched for memory efficiency ────────────────────
BATCH_SIZE_SHAP = 200   # process patients in batches to avoid OOM

def compute_population_shap(model, X, batch_size=BATCH_SIZE_SHAP):
    """
    Returns mean_abs_shap: shape (n_features,)
    Computed as mean of |SHAP| across all patients.
    Also returns shap_matrix: shape (n_patients, n_features) for bootstrapping.
    """
    explainer   = shap_lib.TreeExplainer(model)
    n_patients  = X.shape[0]
    n_features  = X.shape[1]
    shap_matrix = np.zeros((n_patients, n_features), dtype=np.float32)

    for start in range(0, n_patients, batch_size):
        end         = min(start + batch_size, n_patients)
        batch_shap  = explainer.shap_values(X[start:end])
        shap_matrix[start:end] = batch_shap
        if (start // batch_size) % 5 == 0:
            print(f"    processed {end}/{n_patients} patients", flush=True)

    mean_abs_shap = np.mean(np.abs(shap_matrix), axis=0)
    return mean_abs_shap, shap_matrix


# ── BOOTSTRAP CI ON ΔΦ ───────────────────────────────────────────────────────
def bootstrap_delta_ci(abs_shap_src, abs_shap_adp, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    abs_shap_src, abs_shap_adp: shape (n_patients, n_features)
    Returns delta_point, ci_low, ci_high: each shape (n_features,)
    """
    rng     = np.random.RandomState(seed)
    n       = abs_shap_src.shape[0]
    n_feat  = abs_shap_src.shape[1]

    # Point estimate
    delta_point = np.mean(abs_shap_adp, axis=0) - np.mean(abs_shap_src, axis=0)

    # Bootstrap
    boot_deltas = np.zeros((n_boot, n_feat), dtype=np.float32)
    for b in range(n_boot):
        idx                = rng.choice(n, n, replace=True)
        boot_deltas[b, :] = (np.mean(np.abs(abs_shap_adp[idx]), axis=0)
                             - np.mean(np.abs(abs_shap_src[idx]), axis=0))

    ci_low  = np.percentile(boot_deltas, 2.5,  axis=0)
    ci_high = np.percentile(boot_deltas, 97.5, axis=0)

    return delta_point, ci_low, ci_high


# ── MAIN LOOP — one per label ─────────────────────────────────────────────────
results = {}

for lbl_idx, lbl in enumerate(LABEL_COLS):
    short = lbl.replace("label_", "")
    print(f"\n{'='*70}")
    print(f"LABEL: {short.upper()}")
    print(f"{'='*70}")

    src_model = xgb_src_models[lbl]
    adp_model = xgb_adp_models[lbl]

    print("  Computing population SHAP — source model...")
    mean_abs_src, shap_mat_src = compute_population_shap(src_model, X_post)

    print("  Computing population SHAP — adapted model...")
    mean_abs_adp, shap_mat_adp = compute_population_shap(adp_model, X_post)

    print(f"  Bootstrapping ΔΦ CIs (N={N_BOOTSTRAP})...")
    delta_pt, ci_lo, ci_hi = bootstrap_delta_ci(
        np.abs(shap_mat_src), np.abs(shap_mat_adp)
    )

    # ── Separate into physio and treatment streams ───────────────────────
    label_results = {"physio": [], "treatment": []}

    for i, name in enumerate(feat_names):
        stream = "treatment" if is_treatment[i] else "physio"
        label_results[stream].append({
            "feature"    : name,
            "mean_abs_src": float(mean_abs_src[i]),
            "mean_abs_adp": float(mean_abs_adp[i]),
            "delta_phi"  : float(delta_pt[i]),
            "ci_low"     : float(ci_lo[i]),
            "ci_high"    : float(ci_hi[i]),
        })

    # Sort each stream by |ΔΦ| descending
    for stream in ["physio", "treatment"]:
        label_results[stream].sort(key=lambda x: abs(x["delta_phi"]), reverse=True)

    results[lbl] = label_results

    # ── Console print — top-K per stream ────────────────────────────────
    for stream in ["physio", "treatment"]:
        top = label_results[stream][:TOP_K]
        print(f"\n  Top-{TOP_K} {stream} features by |ΔΦ| — {short}:")
        print(f"  {'Feature':<40} {'Φ_src':>8} {'Φ_adp':>8} {'ΔΦ':>8}  95% CI")
        print(f"  {'-'*80}")
        for r in top:
            sig = "*" if (r["ci_low"] > 0 or r["ci_high"] < 0) else " "
            print(f"  {r['feature']:<40} {r['mean_abs_src']:>8.4f} "
                  f"{r['mean_abs_adp']:>8.4f} {r['delta_phi']:>+8.4f}"
                  f"  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{sig}")

# ── SAVE JSON ─────────────────────────────────────────────────────────────────
out_json = SAVE_PATH / "delta_attribution_table.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Saved → {out_json}")

# ── PAPER-READY SUMMARY TABLE ─────────────────────────────────────────────────
# Produces the exact text for Table in §3.6 / supplementary
summary_lines = []
summary_lines.append("=" * 90)
summary_lines.append("Δ-ATTRIBUTION TABLE — XGBoost (Source → Adapted), Population Level")
summary_lines.append(f"N = {X_post.shape[0]} post-drift held-out patients | Bootstrap N={N_BOOTSTRAP}")
summary_lines.append("ΔΦ = mean|SHAP_adapted| − mean|SHAP_source| across all patients")
summary_lines.append("* = 95% CI excludes zero (statistically significant shift)")
summary_lines.append("=" * 90)

for lbl_idx, lbl in enumerate(LABEL_COLS):
    short = lbl.replace("label_", "").replace("_", " ").title()
    summary_lines.append(f"\n── {short} ──────────────────────────────────────────────────")

    for stream in ["physio", "treatment"]:
        stream_label = "PHYSIOLOGICAL STREAM" if stream == "physio" else "TREATMENT STREAM"
        summary_lines.append(f"\n  {stream_label}")
        summary_lines.append(f"  {'Feature':<40} {'Φ_src':>8} {'Φ_adp':>8} {'ΔΦ':>8}  95% CI")
        summary_lines.append(f"  {'-'*78}")

        top = results[lbl][stream][:TOP_K]
        for r in top:
            sig = " *" if (r["ci_low"] > 0 or r["ci_high"] < 0) else "  "
            summary_lines.append(
                f"  {r['feature']:<40} {r['mean_abs_src']:>8.4f} "
                f"{r['mean_abs_adp']:>8.4f} {r['delta_phi']:>+8.4f}"
                f"  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{sig}"
            )

        # stream-level summary
        all_deltas = [r["delta_phi"] for r in results[lbl][stream]]
        n_sig_pos  = sum(1 for r in results[lbl][stream]
                        if r["ci_low"] > 0)
        n_sig_neg  = sum(1 for r in results[lbl][stream]
                        if r["ci_high"] < 0)
        summary_lines.append(
            f"\n  Summary: {len(all_deltas)} features | "
            f"sig. increased: {n_sig_pos} | sig. decreased: {n_sig_neg} | "
            f"mean |ΔΦ|: {np.mean(np.abs(all_deltas)):.4f}"
        )

summary_lines.append("\n" + "=" * 90)

# Key contrast: Run B physio ΔΦ = 0 by construction
summary_lines.append("\nRun B PHYSIOLOGICAL ΔΦ (by architectural construction):")
summary_lines.append("  LSTM weights frozen → mean_rel_Δ = 0.0000 for all physio features")
summary_lines.append("  Any non-zero IG shift in Run B originates from fusion head, not encoder")
summary_lines.append("=" * 90)

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

out_txt = SAVE_PATH / "delta_attribution_summary.txt"
with open(out_txt, "w") as f:
    f.write(summary_text)
print(f"\n✅ Saved → {out_txt}")
print("\n✅ script_mc4_delta_attribution.py complete")