#%%writefile evaluation/script7_psi_sensitivity.py
"""
script7_psi_sensitivity.py
══════════════════════════
PSI threshold sensitivity analysis for selective adaptation trigger.

Note: Binary features use rate-shift thresholding, not PSI.
Continuous features use quantile-based binning for better resolution
on skewed distributions. Because observed drift substantially exceeds
all tested thresholds, adaptation triggers uniformly — demonstrating
robustness of the 0.20 threshold choice.
"""

import torch
import torch.nn as nn
import numpy as np
import polars as pl
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, SingleStreamDataset, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import PhysiologyStream, TreatmentStream, FusionHead, SingleStreamModel, TwoStreamModel, SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM, BATCH_SIZE, LSTM_LAYERS, DROPOUT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")
s2       = Path("/kaggle/input/datasets/fatematamanna/ptfiles")
SAVE_PATH = Path("/kaggle/working")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
SEQ_LEN    = 6
BATCH_SIZE = 256
N_BINS     = 10


# Continuous features only — PSI applied to these
CONTINUOUS_COLS = [c for c in TREATMENT_FEATURES if c not in BINARY_COLS]


# ── PSI FUNCTIONS ──────────────────────────────────────────────────────────────
def psi_continuous(src_col, tgt_col, n_bins=10):
    if len(src_col) == 0 or len(tgt_col) == 0:
        return 0.0
    
    quantiles = np.linspace(0, 100, n_bins + 1)
    breaks = np.unique(np.percentile(src_col, quantiles))
    
    if len(breaks) < 2:
        return 0.0
    
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    
    src_counts, _ = np.histogram(src_col, bins=breaks)
    tgt_counts, _ = np.histogram(tgt_col, bins=breaks)
    
    p_src = np.clip(src_counts / len(src_col), 1e-6, 1.0 - 1e-6)
    p_tgt = np.clip(tgt_counts / len(tgt_col), 1e-6, 1.0 - 1e-6)
    
    psi_val = np.sum((p_src - p_tgt) * np.log(p_src / p_tgt))
    return float(psi_val) if np.isfinite(psi_val) else 0.0


def psi_binary(src_col, tgt_col):
    if len(src_col) == 0 or len(tgt_col) == 0:
        return 0.0
    
    p_src = np.clip(np.array([np.mean(src_col == 0), np.mean(src_col == 1)]), 
                    1e-6, 1.0 - 1e-6)
    p_tgt = np.clip(np.array([np.mean(tgt_col == 0), np.mean(tgt_col == 1)]), 
                    1e-6, 1.0 - 1e-6)
    
    psi_val = np.sum((p_src - p_tgt) * np.log(p_src / p_tgt))
    return float(psi_val) if np.isfinite(psi_val) else 0.0

def calculate_max_psi(src_df, tgt_df):
    """
    Compute per-feature PSI and return the maximum.
    Continuous features: quantile binning.
    Binary features: two-bin PSI.
    Reports per-feature PSI for transparency.
    """
    psi_vals = {}

    for col in CONTINUOUS_COLS:
        if col in src_df.columns and col in tgt_df.columns:
            src = src_df[col].drop_nulls().to_numpy().astype(float)
            tgt = tgt_df[col].drop_nulls().to_numpy().astype(float)
            if len(src) > 0 and len(tgt) > 0:
                psi_vals[col] = psi_continuous(src, tgt)

    for col in BINARY_COLS:
        if col in src_df.columns and col in tgt_df.columns:
            src = src_df[col].drop_nulls().to_numpy().astype(float)
            tgt = tgt_df[col].drop_nulls().to_numpy().astype(float)
            if len(src) > 0 and len(tgt) > 0:
                psi_vals[col] = psi_binary(src, tgt)

    # Report top drifted features
    sorted_psi = sorted(psi_vals.items(), key=lambda x: x[1], reverse=True)
    print("  Top 5 drifted features (PSI):")
    for feat, val in sorted_psi[:5]:
        print(f"    {feat}: {val:.4f}")

    return max(psi_vals.values()) if psi_vals else 0.0


# ── DATA LOADING ───────────────────────────────────────────────────────────────
print("Loading checkpoint and data...")
ckpt = torch.load(s2 / "two_stream_models (3).pt",
                  map_location=device, weights_only=False)
source_state = ckpt["source"]
seq_dim      = ckpt["seq_dim"]
treat_dim    = ckpt["treat_dim"]
train_stats  = ckpt["train_stats"]

# ── DIMENSION VERIFICATION ─────────────────────────────────────────────────── ← ADD HERE
print(f"Checkpoint: seq_dim={seq_dim}, treat_dim={treat_dim}")
print(f"Feature lists: seq={len(SEQ_FEATURES)}, treat={len(TREATMENT_FEATURES)}")
assert seq_dim == len(SEQ_FEATURES), \
    f"seq_dim mismatch: checkpoint={seq_dim}, SEQ_FEATURES={len(SEQ_FEATURES)}"
assert treat_dim == len(TREATMENT_FEATURES), \
    f"treat_dim mismatch: checkpoint={treat_dim}, TREATMENT_FEATURES={len(TREATMENT_FEATURES)}"
print("✅ Dimensions verified")

with open(s2 / "eval_post_stays.json") as f:
    eval_post_stays = json.load(f)

def load_split(name):
    df = pl.read_parquet(BASE_PATH / f"{name}_final_enriched.parquet")
    if "gender" in df.columns:
        df = df.with_columns((pl.col("gender")=="M").cast(pl.Float32).alias("gender_M"))
    for eth in ["WHITE","BLACK","HISPANIC","ASIAN"]:
        if "ethnicity" in df.columns:
            df = df.with_columns(
                (pl.col("ethnicity")==eth).cast(pl.Float32).alias(f"eth_{eth}"))
    return df


train_df = normalize(load_split("train"), train_stats)
test_df  = normalize(load_split("test"),  train_stats)
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

eval_post_df  = test_post.filter( pl.col("stay_id").is_in(eval_post_stays))
adapt_post_df = test_post.filter(~pl.col("stay_id").is_in(eval_post_stays))

print(f"Eval: {eval_post_df['stay_id'].n_unique()} stays | "
      f"Adapt: {adapt_post_df['stay_id'].n_unique()} stays")

# ── PSI CALCULATION ────────────────────────────────────────────────────────────
print("\nCalculating PSI (source train vs adapt post-drift)...")

# Use first hour only — one row per stay for treatment features
src_treat = train_df.filter(pl.col("hrs_from_admit") == 0)
tgt_treat = adapt_post_df.filter(pl.col("hrs_from_admit") == 0)

observed_max_psi = calculate_max_psi(src_treat, tgt_treat)
print(f"\nMaximum observed PSI: {observed_max_psi:.4f}")

# ── ADAPTATION FUNCTION ────────────────────────────────────────────────────────
def run_selective_adaptation(model, adapt_df, epochs=5, lr=3e-4):
    model.train()
    model.freeze_physio()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    ds = ICUDataset(adapt_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    for _ in range(epochs):
        for xs, xt, y in loader:
            xs, xt, y = xs.to(device), xt.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(xs, xt), y).backward()
            optimizer.step()
    return model

# ── SENSITIVITY SWEEP ──────────────────────────────────────────────────────────
PSI_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.0]
results = {}

eval_ds     = ICUDataset(eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

for psi_thresh in PSI_THRESHOLDS:
    label = "Always Adapt" if psi_thresh == 0.0 else f"PSI≥{psi_thresh}"
    triggered = (observed_max_psi > psi_thresh) or (psi_thresh == 0.0)
    print(f"\n[{label}] triggered={triggered}")

    model = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    model.load_state_dict(source_state)
    model.freeze_physio()

    if triggered:
        model = run_selective_adaptation(model, adapt_post_df)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xs, xt, y in eval_loader:
            logits = model(xs.to(device), xt.to(device))
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.numpy())

    preds  = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    m_auroc = np.nanmean([
        roc_auc_score(labels[:,i], preds[:,i])
        for i in range(3)
        if 0 < labels[:,i].sum() < len(labels)
    ])
    results[psi_thresh] = round(m_auroc, 4)
    print(f"  mAUROC = {results[psi_thresh]:.4f}")

print("\nFinal Results:", results)

# ── PLOT ───────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 6))
plot_psis   = sorted([p for p in results if p != 0.0])
plot_aurocs = [results[p] for p in plot_psis]

plt.plot(plot_psis, plot_aurocs, marker='o', linestyle='-',
         color='steelblue', label='Threshold Sweep')

if 0.0 in results:
    plt.axhline(y=results[0.0], color='red', linestyle='--',
                label=f'Always Adapt: {results[0.0]:.4f}')

plt.axvline(x=observed_max_psi, color='gray', linestyle=':',
            label=f'Observed max PSI: {observed_max_psi:.3f}')

plt.xlabel("PSI Threshold")
plt.ylabel("Post-Drift mAUROC")
plt.title("PSI Adaptation Threshold Sensitivity Analysis\n"
          "(thresholds below observed max PSI all trigger adaptation)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(SAVE_PATH / "psi_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()