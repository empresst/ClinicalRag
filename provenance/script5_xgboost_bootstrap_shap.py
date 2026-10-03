%%writefile evaluation/script5_xgboost_bootstrap_shap.py
"""
script5_xgboost_bootstrap_shap.py
═══════════════════════
Three additions for journal-grade submission:

  ADDITION 1: Bootstrap Confidence Intervals on AUROC/AUPRC deltas
    1000 bootstrap resamples with BCa-corrected 95% CI.
    Tests whether Run B's improvement over Run A is statistically significant.

  ADDITION 2: XGBoost Baseline (flat feature model)
    Traditional ML baseline using the same features (flattened time-series +
    treatment features). Shows the two-stream architecture adds value beyond
    good feature engineering. Also tested with/without adaptation.

  ADDITION 3: Expanded RAG Corpus (200+ passages)
    PubMed-style expanded corpus: ~30 relevant ICU/sepsis/ventilation guidelines
    + ~170 noise passages across 12 unrelated domains. More realistic haystack.

Run AFTER script2 (needs two_stream_models.pt + parquet files).

Dependencies:
  pip install xgboost sentence-transformers
"""

import json
import warnings
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, SingleStreamDataset, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import PhysiologyStream, TreatmentStream, FusionHead, SingleStreamModel, TwoStreamModel, SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM, BATCH_SIZE
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH = Path("/kaggle/working")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42

# ── LOAD EVAL SPLIT (must happen before any reference to split/eval_post_stays) ──
with open(SAVE_PATH / "eval_split.json") as f:
    split = json.load(f)


eval_post_stays      = list(map(int, split["eval_post_stays"]))
adapt_train_stays    = list(map(int, split["adapt_train_stays"]))
adapt_val_stays      = list(map(int, split["adapt_val_stays"]))
pre_cp_stays         = list(map(int, split["pre_cp_stays"]))
post_cp_stays        = list(map(int, split["post_cp_stays"]))
buf_pre_stays        = list(map(int, split["buf_pre_stays"]))
eval_post_subjects   = list(map(int, split["eval_post_subjects"]))
eval_post_subj_set   = set(eval_post_subjects)
drift_tag            = split.get("drift_tag", None)
BUFFER_SIZE          = 500

# ── MODEL LOADING ─────────────────────────────────────────────────────────────
print("\nLoading saved models...")
ckpt = torch.load(SAVE_PATH / "two_stream_models.pt", map_location=device, weights_only=False)
seq_dim, treat_dim, n_targets = ckpt["seq_dim"], ckpt["treat_dim"], ckpt["n_targets"]

model_A      = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_B      = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
source_model = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)

model_A.load_state_dict(ckpt["run_a"])
model_B.load_state_dict(ckpt["run_b"])
source_model.load_state_dict(ckpt["source"])

model_A.eval(); model_B.eval(); source_model.eval()   # all three here, once
print(f"Models loaded: seq={seq_dim}, treat={treat_dim}, targets={n_targets}")

# ── DATA LOADING ──────────────────────────────────────────────────────────────
print("Loading data...")
train_df = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
val_df   = load_enriched_split(BASE_PATH, "val",   SEQ_FEATURES, TREATMENT_FEATURES)
test_df  = load_enriched_split(BASE_PATH, "test",  SEQ_FEATURES, TREATMENT_FEATURES)

train_stats = ckpt["train_stats"]

print("Normalizing...")
train_df = normalize(train_df, train_stats)
val_df   = normalize(val_df,   train_stats)
test_df  = normalize(test_df,  train_stats)


# ── SPLIT RECONSTRUCTION ──────────────────────────────────────────────────────

test_pre     = test_df.filter(pl.col("stay_id").is_in(pre_cp_stays))
test_post    = test_df.filter(pl.col("stay_id").is_in(post_cp_stays))
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))

print(f"Models loaded: seq={seq_dim}, treat={treat_dim}, targets={n_targets}")

# ══════════════════════════════════════════════════════════════════════════════
# ADDITION 1: BOOTSTRAP CONFIDENCE INTERVALS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ADDITION 1: Bootstrap Confidence Intervals (N=1000)")
print("="*70)

@torch.no_grad()
def get_predictions(model, df):
    """Get predictions for all patients in df."""
    ds = ICUDataset(df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_logits, all_labels = [], []
    for x_seq, x_treat, y in loader:
        x_seq, x_treat = x_seq.to(device), x_treat.to(device)
        logits = model(x_seq, x_treat)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return probs, labels


print(f"Test-Pre: {test_pre['stay_id'].n_unique()} stays")
print(f"Test-Post held-out: {len(eval_post_stays)} stays")

def bootstrap_auroc_delta(probs_a, probs_b, labels, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """
    Bootstrap the AUROC difference (B - A) and return:
      - point estimate
      - 95% CI (BCa-corrected)
      - p-value (proportion of resamples where delta <= 0)
    """
    rng = np.random.RandomState(seed)
    n = labels.shape[0]
    n_labels = labels.shape[1]

    results = {}
    for i, lbl in enumerate(LABEL_COLS):
        y = labels[:, i]
        pa = probs_a[:, i]
        pb = probs_b[:, i]

        # Skip if no variance in labels
        if y.sum() == 0 or y.sum() == n:
            results[lbl] = {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
                            "auroc_a": float("nan"), "auroc_b": float("nan"), "n_pos": int(y.sum())}
            continue

        auroc_a = roc_auc_score(y, pa)
        auroc_b = roc_auc_score(y, pb)
        delta_obs = auroc_b - auroc_a

        # Bootstrap
        deltas = np.zeros(n_boot)
        aurocs_a_boot = np.zeros(n_boot)
        aurocs_b_boot = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            y_b = y[idx]
            # Ensure both classes present in resample
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                deltas[b] = delta_obs  # degenerate sample, use observed
                continue
            try:
                a_b = roc_auc_score(y_b, pa[idx])
                b_b = roc_auc_score(y_b, pb[idx])
                deltas[b] = b_b - a_b
                aurocs_a_boot[b] = a_b
                aurocs_b_boot[b] = b_b
            except Exception:
                deltas[b] = delta_obs

        # BCa correction
        # Bias correction factor
        z0 = np.percentile(deltas, [2.5, 97.5])  # fallback
        prop_below = np.mean(deltas < delta_obs)
        from scipy.stats import norm as scipy_norm
        if 0 < prop_below < 1:
            z0_hat = scipy_norm.ppf(prop_below)
        else:
            z0_hat = 0.0

        # Acceleration factor (jackknife)
        jackknife_deltas = np.zeros(n)
        for j in range(min(n, 2000)):  # cap jackknife at 2000 for speed
            mask = np.ones(n, dtype=bool)
            mask[j] = False
            y_j, pa_j, pb_j = y[mask], pa[mask], pb[mask]
            if y_j.sum() == 0 or y_j.sum() == len(y_j):
                jackknife_deltas[j] = delta_obs
                continue
            try:
                jackknife_deltas[j] = roc_auc_score(y_j, pb_j) - roc_auc_score(y_j, pa_j)
            except Exception:
                jackknife_deltas[j] = delta_obs

        jk_mean = jackknife_deltas[:min(n, 2000)].mean()
        jk_diff = jk_mean - jackknife_deltas[:min(n, 2000)]
        a_hat = np.sum(jk_diff**3) / (6 * (np.sum(jk_diff**2))**1.5 + 1e-10)

        # BCa percentiles
        alpha = 0.05
        z_alpha = scipy_norm.ppf([alpha/2, 1 - alpha/2])
        adj_percentiles = []
        for z_a in z_alpha:
            num = z0_hat + z_a
            denom = 1 - a_hat * num
            if abs(denom) < 1e-10:
                adj_percentiles.append(z_a)
            else:
                adj_percentiles.append(scipy_norm.cdf(z0_hat + num / denom) * 100)

        ci_low = np.percentile(deltas, max(0.1, adj_percentiles[0]))
        ci_high = np.percentile(deltas, min(99.9, adj_percentiles[1]))

        # p-value: proportion of bootstrap deltas <= 0
        p_value = np.mean(deltas <= 0)

        # Also get CIs for individual AUROCs
        ci_a = (np.percentile(aurocs_a_boot, 2.5), np.percentile(aurocs_a_boot, 97.5))
        ci_b = (np.percentile(aurocs_b_boot, 2.5), np.percentile(aurocs_b_boot, 97.5))

        results[lbl] = {
            "auroc_a": float(auroc_a), "auroc_a_ci": [float(ci_a[0]), float(ci_a[1])],
            "auroc_b": float(auroc_b), "auroc_b_ci": [float(ci_b[0]), float(ci_b[1])],
            "delta": float(delta_obs),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "p_value": float(p_value),
            "n_pos": int(y.sum()), "n_total": int(n),
        }

    return results

def bootstrap_auprc_delta(probs_a, probs_b, labels, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Same as above but for AUPRC."""
    rng = np.random.RandomState(seed + 1)
    n = labels.shape[0]
    results = {}
    for i, lbl in enumerate(LABEL_COLS):
        y = labels[:, i]
        pa, pb = probs_a[:, i], probs_b[:, i]
        if y.sum() == 0 or y.sum() == n:
            results[lbl] = {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
            continue
        auprc_a = average_precision_score(y, pa)
        auprc_b = average_precision_score(y, pb)
        delta_obs = auprc_b - auprc_a
        deltas = np.zeros(n_boot)
        for b in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            y_b = y[idx]
            if y_b.sum() == 0 or y_b.sum() == len(y_b):
                deltas[b] = delta_obs
                continue
            try:
                deltas[b] = average_precision_score(y_b, pb[idx]) - average_precision_score(y_b, pa[idx])
            except:
                deltas[b] = delta_obs
        ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
        p_value = np.mean(deltas <= 0)
        results[lbl] = {
            "auprc_a": float(auprc_a), "auprc_b": float(auprc_b),
            "delta": float(delta_obs),
            "ci_low": float(ci_low), "ci_high": float(ci_high),
            "p_value": float(p_value),
            "n_pos": int(y.sum()),
        }
    return results

# Get predictions on all evaluation splits
print("\nGetting predictions...")

# Pre-drift (Run A == Run B, use source weights for both)


probs_a_pre, labels_pre = get_predictions(model_A, test_pre)
probs_b_pre, _          = get_predictions(source_model, test_pre)  # source == model_A pre-drift

# Post-drift held-out
probs_a_post, labels_post = get_predictions(model_A, eval_post_df)
probs_b_post, _           = get_predictions(model_B, eval_post_df)

# Val
probs_a_val, labels_val = get_predictions(model_A, val_df)
probs_b_val, _          = get_predictions(source_model, val_df)

print(f"  Pre-drift:  {labels_pre.shape[0]} patients")
print(f"  Post-drift: {labels_post.shape[0]} patients")
print(f"  Val:        {labels_val.shape[0]} patients")

# Bootstrap on post-drift (the key comparison)
print("\nBootstrapping AUROC deltas on post-drift (N=1000)...")
auroc_boot = bootstrap_auroc_delta(probs_a_post, probs_b_post, labels_post)
print("\nBootstrapping AUPRC deltas on post-drift (N=1000)...")
auprc_boot = bootstrap_auprc_delta(probs_a_post, probs_b_post, labels_post)

# Also bootstrap on pre-drift (should show ~0 delta)
print("Bootstrapping AUROC deltas on pre-drift (sanity check)...")
auroc_boot_pre = bootstrap_auroc_delta(probs_a_pre, probs_b_pre, labels_pre)

# Print results
print("\n" + "="*70)
print("BOOTSTRAP RESULTS: Post-Drift (Run A vs Run B)")
print("="*70)
print(f"{'Label':<22} {'AUROC_A':>8} {'AUROC_B':>8} {'Δ':>8} {'95% CI':>18} {'p':>8} {'Sig?':>5}")
print("-"*80)
for lbl in LABEL_COLS:
    r = auroc_boot[lbl]
    sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else "ns"
    ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
    print(f"{lbl:<22} {r['auroc_a']:.4f}   {r['auroc_b']:.4f}   {r['delta']:+.4f}   {ci_str:>18} {r['p_value']:.4f}  {sig}")

print(f"\n{'Label':<22} {'AUPRC_A':>8} {'AUPRC_B':>8} {'Δ':>8} {'95% CI':>18} {'p':>8}")
print("-"*80)
for lbl in LABEL_COLS:
    r = auprc_boot[lbl]
    ci_str = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
    print(f"{lbl:<22} {r['auprc_a']:.4f}   {r['auprc_b']:.4f}   {r['delta']:+.4f}   {ci_str:>18} {r['p_value']:.4f}")

print("\nPre-drift sanity check (deltas should be ~0):")
for lbl in LABEL_COLS:
    r = auroc_boot_pre[lbl]
    print(f"  {lbl}: Δ={r['delta']:+.6f}")

# Save bootstrap results
bootstrap_results = {
    "n_bootstrap": N_BOOTSTRAP,
    "post_drift_auroc": auroc_boot,
    "post_drift_auprc": auprc_boot,
    "pre_drift_auroc_sanity": auroc_boot_pre,
}
with open(SAVE_PATH / "bootstrap_ci_results.json", "w") as f:
    json.dump(bootstrap_results, f, indent=2)
print(f"\nSaved → {SAVE_PATH / 'bootstrap_ci_results.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# ADDITION 2: XGBOOST BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ADDITION 2: XGBoost Baseline")
print("="*70)

import xgboost as xgb

def flatten_for_xgb(df, seq_features, treat_features, label_cols, seq_len=6):
    """
    Flatten time-series into static features for XGBoost:
    - Last value of each seq feature
    - Mean of each seq feature over obs window
    - Std of each seq feature
    - Min/Max of each seq feature
    - All treatment features as-is
    """
    seq_cols = [c for c in seq_features if c in df.columns and not c.endswith("_mask")]
    treat_cols = [c for c in treat_features if c in df.columns]

    stays = df.sort(["stay_id", "hrs_from_admit"])
    stay_ids = stays.select("stay_id").unique().sort("stay_id")["stay_id"].to_list()

    flat_features = []
    flat_labels = []
    flat_stay_ids = []

    for sid in stay_ids:
        s = stays.filter(pl.col("stay_id") == sid)
        seq_vals = s.select(seq_cols).to_numpy().astype(np.float32)
        if seq_vals.shape[0] < seq_len:
            pad = np.zeros((seq_len - seq_vals.shape[0], seq_vals.shape[1]), dtype=np.float32)
            seq_vals = np.vstack([seq_vals, pad])
        else:
            seq_vals = seq_vals[:seq_len]

        # Flatten: last, mean, std, min, max → 5 * n_seq_features
        last_val = seq_vals[-1]
        mean_val = np.nanmean(seq_vals, axis=0)
        std_val  = np.nanstd(seq_vals, axis=0)
        min_val  = np.nanmin(seq_vals, axis=0)
        max_val  = np.nanmax(seq_vals, axis=0)

        treat_val = np.array(s.select(treat_cols).row(0), dtype=np.float32)
        flat_vec = np.concatenate([last_val, mean_val, std_val, min_val, max_val, treat_val])
        flat_features.append(flat_vec)
        flat_labels.append(np.array(s.select(label_cols).row(0), dtype=np.float32))
        flat_stay_ids.append(sid)

    X = np.stack(flat_features)
    Y = np.stack(flat_labels)

    # Feature names for interpretability
    names = []
    for prefix in ["last", "mean", "std", "min", "max"]:
        names += [f"{prefix}_{c}" for c in seq_cols]
    names += treat_cols

    return X, Y, flat_stay_ids, names

print("Flattening features for XGBoost...")
X_train, Y_train, _, xgb_feat_names = flatten_for_xgb(train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
X_val,   Y_val,   _, _ = flatten_for_xgb(val_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
X_pre,   Y_pre,   _, _ = flatten_for_xgb(test_pre, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
X_post,  Y_post,  _, _ = flatten_for_xgb(eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
print(f"  Train: {X_train.shape}, Val: {X_val.shape}")
print(f"  Pre: {X_pre.shape}, Post: {X_post.shape}")
print(f"  Features per patient: {X_train.shape[1]} ({len(xgb_feat_names)} named)")

# Replace NaN/Inf
for arr in [X_train, X_val, X_pre, X_post]:
    arr[~np.isfinite(arr)] = 0.0

# Train one XGBoost per label
xgb_models = {}
xgb_adapted_models = {} 
xgb_results = {"val": {}, "test_pre": {}, "test_post": {}}

for i, lbl in enumerate(LABEL_COLS):
    print(f"\n  Training XGBoost for {lbl}...")
    y_tr, y_va = Y_train[:, i], Y_val[:, i]
    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    scale = n_neg / max(n_pos, 1)

    model_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(scale, 20.0),
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=SEED,
        use_label_encoder=False,
        verbosity=0,
    )
    model_xgb.fit(X_train, y_tr, eval_set=[(X_val, y_va)], verbose=False)
    xgb_models[lbl] = model_xgb

    # Evaluate on all splits
    for split_name, X_s, Y_s in [("val", X_val, Y_val), ("test_pre", X_pre, Y_pre), ("test_post", X_post, Y_post)]:
        y_true = Y_s[:, i]
        y_prob = model_xgb.predict_proba(X_s)[:, 1]
        n_p = int(y_true.sum())
        if n_p > 0 and n_p < len(y_true):
            auroc = roc_auc_score(y_true, y_prob)
            auprc = average_precision_score(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
        else:
            auroc = auprc = brier = float("nan")
        xgb_results[split_name][lbl] = {
            "auroc": float(auroc), "auprc": float(auprc), "brier": float(brier), "n_pos": n_p
        }

    print(f"    Val:  AUROC={xgb_results['val'][lbl]['auroc']:.4f}  AUPRC={xgb_results['val'][lbl]['auprc']:.4f}")
    print(f"    Pre:  AUROC={xgb_results['test_pre'][lbl]['auroc']:.4f}  AUPRC={xgb_results['test_pre'][lbl]['auprc']:.4f}")
    print(f"    Post: AUROC={xgb_results['test_post'][lbl]['auroc']:.4f}  AUPRC={xgb_results['test_post'][lbl]['auprc']:.4f}")

# XGBoost with post-drift adaptation (retrain on same adapt data)
print("\n  Training XGBoost-Adapted (retrained on post-drift adapt data)...")

# REPLACE WITH — use script2's already-computed buffer
adapt_train_df_xgb = test_post.filter(pl.col("stay_id").is_in(adapt_train_stays))
buf_pre_df         = test_pre.filter(pl.col("stay_id").is_in(buf_pre_stays))
combined_adapt_df  = pl.concat([buf_pre_df, adapt_train_df_xgb])

X_adapt, Y_adapt, _, _ = flatten_for_xgb(combined_adapt_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
X_adapt_val_arr, Y_adapt_val_arr, _, _ = flatten_for_xgb(
    test_post.filter(pl.col("stay_id").is_in(adapt_val_stays)),
    SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
X_adapt[~np.isfinite(X_adapt)] = 0.0
X_adapt_val_arr[~np.isfinite(X_adapt_val_arr)] = 0.0

xgb_adapted_results = {"test_post": {}}
for i, lbl in enumerate(LABEL_COLS):
    y_tr = Y_adapt[:, i]
    y_va = Y_adapt_val_arr[:, i]
    n_pos = int(y_tr.sum())
    scale = (len(y_tr) - n_pos) / max(n_pos, 1)

    model_xgb_ad = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(scale, 15.0),
        eval_metric="logloss", early_stopping_rounds=20,
        random_state=SEED, use_label_encoder=False, verbosity=0,
    )
    model_xgb_ad.fit(X_adapt, y_tr, eval_set=[(X_adapt_val_arr, y_va)], verbose=False)
    xgb_adapted_models[lbl] = model_xgb_ad  
    y_true = Y_post[:, i]
    y_prob = model_xgb_ad.predict_proba(X_post)[:, 1]
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        auroc = auprc = float("nan")
    xgb_adapted_results["test_post"][lbl] = {"auroc": float(auroc), "auprc": float(auprc), "n_pos": int(y_true.sum())}
    print(f"    {lbl} adapted: AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

# ── COMPREHENSIVE COMPARISON TABLE ────────────────────────────────────────────
print("\n" + "="*70)
print("FULL MODEL COMPARISON TABLE")
print("="*70)

def get_two_stream_metrics(probs, labels):
    """Get per-label metrics from probs/labels arrays."""
    results = {}
    for i, lbl in enumerate(LABEL_COLS):
        y, p = labels[:, i], probs[:, i]
        if y.sum() > 0 and y.sum() < len(y):
            results[lbl] = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p)}
        else:
            results[lbl] = {"auroc": float("nan"), "auprc": float("nan")}
    return results

ts_a_val  = get_two_stream_metrics(probs_a_val, labels_val)
ts_a_pre  = get_two_stream_metrics(probs_a_pre, labels_pre)
ts_a_post = get_two_stream_metrics(probs_a_post, labels_post)
ts_b_post = get_two_stream_metrics(probs_b_post, labels_post)


# Print table
for split_name, models in [
    (f"Val (source era)", [
        ("XGBoost", xgb_results["val"]),
        ("Two-Stream (Run A)", ts_a_val),
    ]),
    (f"Test-Pre (pre {drift_tag})", [
        ("XGBoost", xgb_results["test_pre"]),
        ("Two-Stream (Run A)", ts_a_pre),
    ]),
    (f"Test-Post ({drift_tag}+)", [
        ("XGBoost (source)", xgb_results["test_post"]),
        ("XGBoost (adapted)", xgb_adapted_results["test_post"]),
        ("Two-Stream Run A (static)", ts_a_post),
        ("Two-Stream Run B (adapted)", ts_b_post),
    ]),
]:
    print(f"\n{split_name}:")
    print(f"  {'Model':<30} {'Vaso AUROC':>10} {'Intub AUROC':>12} {'Shock AUROC':>12} | {'Vaso AUPRC':>10} {'Intub AUPRC':>12} {'Shock AUPRC':>12}")
    print("  " + "-"*100)
    for model_name, metrics in models:
        vals = []
        for lbl in LABEL_COLS:
            m = metrics.get(lbl, {})
            vals.append(f"{m.get('auroc', float('nan')):.4f}")
            vals.append(f"{m.get('auprc', float('nan')):.4f}")
        print(f"  {model_name:<30} {vals[0]:>10} {vals[2]:>12} {vals[4]:>12} | {vals[1]:>10} {vals[3]:>12} {vals[5]:>12}")

# Save comparison
comparison = {
    "xgboost_source": xgb_results,
    "xgboost_adapted": xgb_adapted_results,
    "two_stream_run_a": {"val": ts_a_val, "test_pre": ts_a_pre, "test_post": ts_a_post},
    "two_stream_run_b": {"test_post": ts_b_post},
    "bootstrap_auroc": auroc_boot,
    "bootstrap_auprc": auprc_boot,
}
with open(SAVE_PATH / "full_comparison_results.json", "w") as f:
    json.dump(comparison, f, indent=2)
print(f"\nSaved → {SAVE_PATH / 'full_comparison_results.json'}")


# Get PyTorch ordering — this is the reference ordering

eval_ds = ICUDataset(eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
pytorch_stay_order = eval_ds.stay_ids  # sorted by stay_id ascending
print(f"PyTorch ordering: {len(pytorch_stay_order)} stays")
print(f"First 5: {pytorch_stay_order[:5]}")

# Flatten features IN THE SAME ORDER as PyTorch
# Sort eval_post_df by stay_id to match PyTorch ICUDataset ordering
eval_post_sorted = eval_post_df.filter(
    pl.col("hrs_from_admit") == 0
).sort("stay_id")

# Verify order matches PyTorch
sorted_sids = eval_post_sorted["stay_id"].to_list()
assert sorted_sids == pytorch_stay_order, "Ordering still mismatched"
print("✅ Ordering matches PyTorch")

# Now flatten for XGBoost IN THIS ORDER
X_eval, Y_eval, sids_eval, feat_names = flatten_for_xgb(
    eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
)

# Verify sids_eval matches pytorch_stay_order
print(f"XGBoost flatten ordering first 5: {sids_eval[:5]}")
assert sids_eval == pytorch_stay_order, \
    f"Mismatch at: {next(i for i,(a,b) in enumerate(zip(sids_eval, pytorch_stay_order)) if a!=b)}"
print("✅ XGBoost flatten order verified against PyTorch")

X_eval[~np.isfinite(X_eval)] = 0.0

# Generate predictions using saved XGBoost models
probs_source_new  = np.zeros((len(pytorch_stay_order), 3))
probs_adapted_new = np.zeros((len(pytorch_stay_order), 3))

for i, lbl in enumerate(LABEL_COLS):
    probs_source_new[:, i]  = xgb_models[lbl].predict_proba(X_eval)[:, 1]
    probs_adapted_new[:, i] = xgb_adapted_models[lbl].predict_proba(X_eval)[:, 1]

# Verify AUROC matches Table 4
from sklearn.metrics import roc_auc_score
print("\nAUROC verification (should match Table 4):")
for i, lbl in enumerate(LABEL_COLS):
    y = Y_eval[:, i]
    if 0 < y.sum() < len(y):
        auroc_src = roc_auc_score(y, probs_source_new[:, i])
        auroc_adp = roc_auc_score(y, probs_adapted_new[:, i])
        print(f"  {lbl}: source={auroc_src:.4f} adapted={auroc_adp:.4f}")

# Save corrected predictions
np.savez(
    SAVE_PATH / "xgb_predictions_corrected.npz",
    probs_xgb_source  = probs_source_new,
    probs_xgb_adapted = probs_adapted_new,
    stay_ids          = np.array(pytorch_stay_order),
    labels            = Y_eval,
)
print("\n✅ Saved xgb_predictions_corrected.npz")

import joblib
for lbl in LABEL_COLS:
    joblib.dump(xgb_models[lbl],
                SAVE_PATH / f"xgb_source_{lbl}.pkl")
    joblib.dump(xgb_adapted_models[lbl],
                SAVE_PATH / f"xgb_adapted_{lbl}.pkl")
print("✅ XGBoost models saved for script9")

# ══════════════════════════════════════════════════════════════════════════════
# BRIDGE: Per-patient SHAP for fig_biological_amnesia
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("BRIDGE: Computing per-patient SHAP for fig_biological_amnesia")
print("="*70)

import shap, pickle

# ── Load the IG side that script2 exported ────────────────────────────────────
ig_path = SAVE_PATH / "fig_amnesia_ig_data.pkl"
if not ig_path.exists():
    print(f"⚠ {ig_path} not found — run script2 first. Skipping bridge.")
else:
    with open(ig_path, "rb") as f:
        ig_data = pickle.load(f)

    # The fig file needs one consistent patient across IG and SHAP panels.
    # Script2 picked a patient by stay_id; find that row in the XGBoost ordering.
    fig_stay_id = ig_data["stay_id"]

    npz = np.load(SAVE_PATH / "xgb_predictions_corrected.npz", allow_pickle=True)
    stay_ids_arr = npz["stay_ids"].tolist()

    if fig_stay_id not in stay_ids_arr:
        print(f"⚠ stay_id {fig_stay_id} not in XGBoost eval set — "
              f"picking closest available patient")
        # Fall back to highest-probability positive from script2's label_idx
        LABEL_IDX = ig_data["label_idx"]
        labels_npz = npz["labels"]
        probs_npz  = npz["probs_xgb_source"]
        pos_mask   = labels_npz[:, LABEL_IDX] == 1
        if pos_mask.sum() == 0:
            xi_best = 0
        else:
            xi_best = int(np.where(pos_mask)[0][
                np.argmax(probs_npz[pos_mask, LABEL_IDX])])
        fig_stay_id = stay_ids_arr[xi_best]
        print(f"  Fell back to stay_id={fig_stay_id} (index {xi_best})")
    else:
        xi_best = stay_ids_arr.index(fig_stay_id)
        print(f"  Matched stay_id={fig_stay_id} at XGBoost index {xi_best}")

    # ── Build the feature row for this patient ────────────────────────────
    # Need to re-flatten just this patient in the same feature space script5 used
    LABEL_IDX  = ig_data["label_idx"]
    patient_df = eval_post_df.filter(pl.col("stay_id") == fig_stay_id)
    X_patient, Y_patient, _, _ = flatten_for_xgb(
        patient_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
    X_patient[~np.isfinite(X_patient)] = 0.0

    # ── SHAP values from source and adapted XGBoost ───────────────────────
    # Use the vasopressor model (LABEL_IDX=0) — same as IG panels
    src_xgb_model = xgb_models[LABEL_COLS[LABEL_IDX]]
    adp_xgb_model = xgb_adapted_models[LABEL_COLS[LABEL_IDX]]

    explainer_xgb_src = shap.TreeExplainer(src_xgb_model)
    explainer_xgb_adp = shap.TreeExplainer(adp_xgb_model)

    # shap_values shape: (1, n_features)
    shap_src_patient = explainer_xgb_src.shap_values(X_patient)
    shap_adp_patient = explainer_xgb_adp.shap_values(X_patient)

    # ── XGBoost predicted probabilities for this patient ─────────────────
    p_xgb_src = float(src_xgb_model.predict_proba(X_patient)[0, 1])
    p_xgb_adp = float(adp_xgb_model.predict_proba(X_patient)[0, 1])
    print(f"  p_xgb_src={p_xgb_src:.3f}  p_xgb_adp={p_xgb_adp:.3f}")

    # ── Build xgb_feat_names (same as flatten_for_xgb produces) ──────────
    # Already computed above as `xgb_feat_names` from the flatten call
    # Verify length matches
    assert len(xgb_feat_names) == shap_src_patient.shape[1], \
        f"Feature name count mismatch: {len(xgb_feat_names)} vs {shap_src_patient.shape[1]}"

    # ── all_xgb_deltas ────────────────────────────────────────────────────
    xgb_p95 = float(np.percentile(np.abs(shap_src_patient[0]), 95))
    xgb_p95 = max(xgb_p95, 1e-6)

    all_xgb_deltas = {}
    for i, n in enumerate(xgb_feat_names):
        src_v = float(shap_src_patient[0][i])
        adp_v = float(shap_adp_patient[0][i])
        d     = abs(adp_v - src_v) / xgb_p95
        all_xgb_deltas[n] = {"src": src_v, "adp": adp_v, "delta": d}

    # ── XGBoost normalization arrays for bottom panel ─────────────────────
    treat_col_set = set(ig_data["treat_cols"])
    seq_col_set   = set(ig_data["seq_cols"])

    phys_deltas_xgb, treat_deltas_xgb = [], []
    for n, v in all_xgb_deltas.items():
        # strip prefix from flatten: "last_X", "mean_X" etc.
        base = "_".join(n.split("_")[1:]) if "_" in n else n
        if base in treat_col_set or n in treat_col_set:
            treat_deltas_xgb.append(v["delta"])
        else:
            phys_deltas_xgb.append(v["delta"])

    xgb_phys_norm  = np.array(phys_deltas_xgb)  if phys_deltas_xgb  else np.array([0.0])
    xgb_treat_norm = np.array(treat_deltas_xgb) if treat_deltas_xgb else np.array([0.0])

    # amnesia feature: largest SHAP delta
    amnesia_feat = max(all_xgb_deltas, key=lambda n: all_xgb_deltas[n]["delta"])
    delta        = all_xgb_deltas[amnesia_feat]["delta"]
    print(f"  Amnesia feature: {amnesia_feat}  Δ={delta:.4f}")

    # ── Merge with IG data and save the complete fig pkl ──────────────────
    fig_data = {
        **ig_data,                        # all IG variables from script2
        "all_xgb_deltas":   all_xgb_deltas,
        "shap_src_all":     shap_src_patient,
        "xgb_feat_names":   xgb_feat_names,
        "xi_best":          0,             # always 0: single-patient array
        "p_xgb_src":        p_xgb_src,
        "p_xgb_adp":        p_xgb_adp,
        "xgb_phys_norm":    xgb_phys_norm,
        "xgb_treat_norm":   xgb_treat_norm,
        "xgb_p95":          xgb_p95,
        "amnesia_feat":     amnesia_feat,
        "delta":            delta,
        # override stay_id/true_label with the verified matched patient
        "stay_id":          int(fig_stay_id),
        "true_label":       int(Y_patient[0, LABEL_IDX]),
    }

    with open(SAVE_PATH / "fig_amnesia_data.pkl", "wb") as f:
        pickle.dump(fig_data, f)
    print(f"✅ Saved fig_amnesia_data.pkl — ready for fig_biological_amnesia_v3_patch.py")