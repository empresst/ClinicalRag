"""
script4_enhancements.py
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

import json, warnings, copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 42, 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH  = Path("/kaggle/working")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
TOP_K_DOCS = 5
N_CASES_PER_CONDITION = 5

torch.manual_seed(SEED); np.random.seed(SEED)

# ── FEATURE DEFINITIONS (same as script2/3) ───────────────────────────────────
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
    "total_crystalloid_ml", "has_norepinephrine_obs", "has_phenylephrine_obs",
    "has_dopamine_obs", "has_vasopressin_obs", "time_to_first_vaso_hrs",
    "early_steroid", "early_antibiotic", "n_distinct_meds",
    "steroid_ordered", "time_to_first_abx_order_hrs",
    "max_fio2_obs", "mean_fio2_obs", "max_peep_obs", "max_tidal_volume_obs",
    "high_fio2_flag", "on_peep_flag",
    "has_propofol_midaz_obs", "total_sedation_dose_obs",
    "age","gender_M","eth_WHITE","eth_BLACK","eth_HISPANIC","eth_ASIAN",
]

BINARY_COLS = {c for c in TREATMENT_FEATURES
               if c.startswith("has_") or c.startswith("eth_") or c == "gender_M"
               or c in ("early_steroid","early_antibiotic","steroid_ordered",
                        "high_fio2_flag","on_peep_flag")}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (must match script2)
# ══════════════════════════════════════════════════════════════════════════════
class PhysiologyStream(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.norm(h[-1])

class TreatmentStream(nn.Module):
    def __init__(self, input_dim, output_dim=32, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim), nn.ReLU(), nn.LayerNorm(output_dim))
    def forward(self, x): return self.mlp(x)

class FusionHead(nn.Module):
    def __init__(self, physio_dim=64, treat_dim=32, n_targets=4, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(physio_dim + treat_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.75),
            nn.Linear(32, n_targets))
    def forward(self, p, t): return self.head(torch.cat([p, t], dim=1))

class TwoStreamModel(nn.Module):
    def __init__(self, seq_input_dim, treat_input_dim, n_targets=4):
        super().__init__()
        self.physio = PhysiologyStream(seq_input_dim, HIDDEN_DIM, LSTM_LAYERS, DROPOUT)
        self.treat  = TreatmentStream(treat_input_dim, TREAT_DIM, DROPOUT)
        self.fusion = FusionHead(HIDDEN_DIM, TREAT_DIM, n_targets, DROPOUT)

    def forward(self, x_seq, x_treat):
        return self.fusion(self.physio(x_seq), self.treat(x_treat))

    def freeze_physio(self):
        for p in self.physio.parameters(): p.requires_grad = False

    def freeze_all(self):
        for p in self.parameters(): p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze every parameter — used by Run C."""
        for p in self.parameters(): p.requires_grad = True

    def unfreeze_adaptive(self):
        """Freeze physio, unfreeze treat+fusion — used by Run B."""
        self.freeze_all()
        for p in self.treat.parameters():  p.requires_grad = True
        for p in self.fusion.parameters(): p.requires_grad = True

# ── DATA LOADING ───────────────────────────────────────────────────────────────
def load_split(name):
    df = pl.read_parquet(BASE_PATH / f"{name}_final_enriched.parquet")
    if "gender" in df.columns:
        df = df.with_columns((pl.col("gender") == "M").cast(pl.Float32).alias("gender_M"))
    for eth in ["WHITE","BLACK","HISPANIC","ASIAN"]:
        if "ethnicity" in df.columns:
            df = df.with_columns((pl.col("ethnicity") == eth).cast(pl.Float32).alias(f"eth_{eth}"))
    for c in SEQ_FEATURES + TREATMENT_FEATURES:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(c))
    return df

print("Loading saved models...")
ckpt = torch.load(SAVE_PATH / "two_stream_models.pt", map_location=device, weights_only=False)
seq_dim, treat_dim, n_targets = ckpt["seq_dim"], ckpt["treat_dim"], ckpt["n_targets"]
train_stats = ckpt["train_stats"]

model_A = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_A.load_state_dict(ckpt["run_a"]); model_A.eval()
model_B = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_B.load_state_dict(ckpt["run_b"]); model_B.eval()
print(f"Models loaded: seq={seq_dim}, treat={treat_dim}, targets={n_targets}")

def normalize(df, stats):
    exprs = []
    for c, (mu, sd) in stats.items():
        if c in df.columns:
            if c.endswith("_mask") or c in BINARY_COLS:
                exprs.append(pl.col(c).cast(pl.Float32))
            else:
                exprs.append(((pl.col(c).cast(pl.Float64) - mu) / sd).cast(pl.Float32).alias(c))
    return df.with_columns(exprs)

print("Loading data...")
train_df = normalize(load_split("train"), train_stats)
val_df   = normalize(load_split("val"),   train_stats)
test_df  = normalize(load_split("test"),  train_stats)

class ICUDataset(Dataset):
    def __init__(self, df, seq_features, treat_features, label_cols, seq_len=6):
        self.seq_len, self.label_cols = seq_len, label_cols
        self.seq_cols   = [c for c in seq_features  if c in df.columns]
        self.treat_cols = [c for c in treat_features if c in df.columns]
        stays = df.sort(["stay_id","hrs_from_admit"])
        self.stay_ids = stays.select("stay_id").unique().sort("stay_id")["stay_id"].to_list()
        self.seq_data, self.treat_data, self.labels = [], [], []
        for sid in self.stay_ids:
            s = stays.filter(pl.col("stay_id") == sid)
            seq = s.select(self.seq_cols).to_numpy().astype(np.float32)
            if seq.shape[0] < seq_len:
                seq = np.vstack([seq, np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)])
            else:
                seq = seq[:seq_len]
            self.seq_data.append(seq)
            self.treat_data.append(np.array(s.select(self.treat_cols).row(0), dtype=np.float32))
            self.labels.append(np.array(s.select(label_cols).row(0), dtype=np.float32))
        self.seq_data   = np.stack(self.seq_data)
        self.treat_data = np.stack(self.treat_data)
        self.labels     = np.stack(self.labels)
    def __len__(self): return len(self.stay_ids)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.seq_data[idx]),
                torch.from_numpy(self.treat_data[idx]),
                torch.from_numpy(self.labels[idx]))

# Separate pre/post drift test sets
test_pre  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

# Reproduce the held-out split from script2
post_stays = test_post.filter(pl.col("hrs_from_admit") == 0).sort("intime")["stay_id"].to_list()
n_total_post = len(post_stays)
n_adapt_train = int(n_total_post * 0.30)
n_adapt_val   = int(n_total_post * 0.10)
eval_post_stays = post_stays[n_adapt_train + n_adapt_val:]
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))

print(f"Test-Pre: {test_pre['stay_id'].n_unique()} stays")
print(f"Test-Post held-out: {len(eval_post_stays)} stays")


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
source_model = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
source_model.load_state_dict(ckpt["source"]); source_model.eval()

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
xgb_adapted_models = {} #look
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
adapt_train_stays = post_stays[:n_adapt_train]
adapt_train_df_xgb = test_post.filter(pl.col("stay_id").is_in(adapt_train_stays))

# Include pre-drift buffer
pre_stays = test_pre.select("stay_id").unique().sort("stay_id")["stay_id"].to_list()
buf_pre_stays = pre_stays[-500:] if len(pre_stays) > 500 else pre_stays
buf_pre_df = test_pre.filter(pl.col("stay_id").is_in(buf_pre_stays))
combined_adapt_df = pl.concat([buf_pre_df, adapt_train_df_xgb])

X_adapt, Y_adapt, _, _ = flatten_for_xgb(combined_adapt_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS)
X_adapt_val_arr, Y_adapt_val_arr, _, _ = flatten_for_xgb(
    test_post.filter(pl.col("stay_id").is_in(post_stays[n_adapt_train:n_adapt_train + n_adapt_val])),
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
    xgb_adapted_models[lbl] = model_xgb_ad  #look
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
    ("Val (2014-16)", [
        ("XGBoost", xgb_results["val"]),
        ("Two-Stream (Run A)", ts_a_val),
    ]),
    ("Test-Pre (2017-19)", [
        ("XGBoost", xgb_results["test_pre"]),
        ("Two-Stream (Run A)", ts_a_pre),
    ]),
    ("Test-Post (2020-22)", [
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
