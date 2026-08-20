%%writefile models/script2_two_stream_model.py
"""
script2_two_stream_model_v7.py
═════════════════════════════════
Three-run clinical AI — Run A vs Run B vs Run C.

CHANGES vs v6:
  CHG 1:  AUROC_DROP_THR removed. AUROC_FLOOR_DROP renamed AUROC_DRIFT_THR and
          used as the single threshold in both pairwise and cumulative paths.
  CHG 2:  ensemble_perf_drop_vs_baseline simplified — majority_fired and the
          std-gate removed. Returns (mean_drop, std_drop, per_lbl_mean_drops,
          n_sig_ensemble). std_drop kept for diagnostic logging only.
  CHG 3:  BUG FIX — pairwise loop crashed: std_drop and majority_fired were
          referenced before assignment because ensemble_perf_drop_vs_baseline
          was never called there (only perf_drop_vs_baseline was). Fixed by
          calling ensemble_perf_drop_vs_baseline in the pairwise loop and
          removing the single-model perf_drop_vs_baseline call from that path.
  CHG 4:  _single_model_drop_vs_baseline removed (dead code after CHG 2/3).
  CHG 5:  all_source_val_mets_per_seed alias removed (redundant).
  CHG 6:  Drift onset uses a clean OR-gate: dist_exceeded OR perf_exceeded.
          majority_fired removed from the condition entirely.
  CHG 7:  Cumulative tracker std-gate removed: fires on mean_drop > AUROC_DRIFT_THR
          alone (std-gate made detector blind when seed variance was high).
  CHG 8:  torch.cuda.empty_cache() added after del _m_tmp in cumulative loop
          to prevent GPU OOM with 5 seeds × multiple test groups.
  CHG 9:  Val split assertion guarded against empty val_groups.
  CHG 10: Source model reconstructed once and shared between IG block and
          explainer block (was reconstructed twice with identical weights).
  CHG 11: 15-line commented-out cumulative drift override block deleted.
  CHG 12: majority_fired and std_auroc_drop scrubbed from pair_results dict,
          all print statements, and eval_split.json.
  CHG 13: ICUDataset stay_ids usage guarded with a fallback.
  CHG 14: perf_drop_vs_baseline kept for reference but no longer called in the
          pairwise loop (ensemble version used instead for consistency).

RETAINED FROM v6 / v5:
  FIX 1: Subject-level adaptation split.
  FIX 2: Honest pairwise baselines (val for all pairs).
  FIX 3: Label-window leakage audit + clip for treatment timing features.
  FIX 4: Single AUROC threshold (now AUROC_DRIFT_THR).
  FIX 5: Median-seed model saved for B and C.
  FIX 6: Hardcoded thresholds documented.
  FIX 7: ref_edges pooled across ALL training groups.
  FIX 8: Pre-drift buffer sorted by intime, subject-level leakage guard.
  FIX 9: Multi-source-seed ensemble for drift detection.

Run A: Fully frozen source model (median-seed source).
Run B: Two-stream, LSTM frozen, treatment+fusion trainable.
Run C: Two-stream, NOTHING frozen — all layers trainable.
"""
import json, warnings, copy
from pathlib import Path
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from utils.data_utils import ICUDataset, SingleStreamDataset
from models.architectures import TwoStreamModel, PhysiologyStream, TreatmentStream, FusionHead, TwoStreamModel

import datetime
from utils.drift_explainability import DriftExplainer

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 42, 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LR_INIT, LR_ADAPT = 1e-3, 3e-4
EPOCHS, ADAPT_EPOCHS = 50, 40
PATIENCE, ADAPT_PATIENCE = 8, 8
BUFFER_SIZE = 500

# FIX 9: Multi-source-seed ensemble.
# Set SOURCE_SEEDS=[42] to reproduce single-source behaviour.
# Use 3 for fast iteration, 5 for final results.
SOURCE_SEEDS = [42, 123, 7, 2024, 99]

# ── DRIFT-DETECTION THRESHOLDS ────────────────────────────────────────────────
# All operator-chosen; calibrate on synthetic drift if deploying for real.
PSI_THRESH        = 0.20  # Conventional PSI alert threshold (Siddiqi 2006).
JUMP_FACTOR       = 2.0   # Pair must exceed 2× the training-era baseline.
MIN_FEAT_DRIFTED  = 2     # ≥2 features must show distribution drift.
BINARY_DELTA_THR  = 0.05  # 5-percentage-point change in binary prevalence.
# CHG 1: single AUROC threshold used in both pairwise and cumulative paths.
# (Previously two constants: AUROC_DROP_THR=0.035 and AUROC_FLOOR_DROP=0.020.
#  Using the more conservative 0.020 to avoid missing early degradation.)
AUROC_DRIFT_THR   = 0.020 # Per-label mean AUROC drop that counts as significant.
MIN_PERF_DROPS    = 1     # ≥1 label must drop > AUROC_DRIFT_THR.

LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH  = Path("/kaggle/working")
TRAIN_YEARS = ["2008 - 2010", "2011 - 2013"]

# FIX 3: features where value > SEQ_LEN means an event happened AFTER the
# model's input window, leaking information from the label window.
LEAKAGE_TIMING_FEATS = ["time_to_first_abx_order_hrs"]

torch.manual_seed(SEED); np.random.seed(SEED)

def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, seed=0, alpha=0.05):
    """Returns (point_estimate, lo, hi) percentile-bootstrap CI."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_prob)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        vals.append(metric_fn(yt, yp))
    if not vals:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return float(point), lo, hi

# ── DATA LOADING ───────────────────────────────────────────────────────────────
print("Loading data...")
train_df = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
val_df   = load_enriched_split(BASE_PATH, "val",   SEQ_FEATURES, TREATMENT_FEATURES)
test_df  = load_enriched_split(BASE_PATH, "test",  SEQ_FEATURES, TREATMENT_FEATURES)

# CHG 9: guard against empty val_groups silently passing the subset check.
val_groups = set(val_df["anchor_year_group"].unique().to_list())
assert val_groups and val_groups <= set(TRAIN_YEARS), (
    f"Val split contains non-training-era groups: {val_groups - set(TRAIN_YEARS)}\n"
    f"Val must contain only {TRAIN_YEARS} to prevent future leakage into source model."
)
print(f"✅ Val temporal integrity confirmed: {val_groups}")

_actual_groups = set(train_df["anchor_year_group"].unique().to_list())
_missing = [y for y in TRAIN_YEARS if y not in _actual_groups]
if _missing:
    raise ValueError(
        f"TRAIN_YEARS contains groups not in data: {_missing}\n"
        f"Available: {sorted(_actual_groups)}\n"
        f"Check for en-dash vs hyphen differences.")
print(f"✅ TRAIN_YEARS validated against data")

# ── FIX 3: LABEL-WINDOW LEAKAGE AUDIT + CLIP (on raw, pre-normalisation) ──────
print("\n--- Label-leakage audit on treatment timing features ---")
for feat in LEAKAGE_TIMING_FEATS:
    if feat not in train_df.columns:
        continue
    for df_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        n_leak  = df.filter(pl.col(feat) > SEQ_LEN).height
        n_total = df.height
        if n_leak > 0:
            pct = 100 * n_leak / max(n_total, 1)
            print(f"  ⚠ {df_name}: {n_leak}/{n_total} ({pct:.1f}%) rows have "
                  f"{feat} > {SEQ_LEN}h → clipping to sentinel ({SEQ_LEN + 1})")
# Clip leakage. Values > SEQ_LEN mean event happened after the input window;
# replace with a single sentinel so the model can learn "no event within window".
SENTINEL_NO_EARLY_EVENT = float(SEQ_LEN + 1)
for feat in LEAKAGE_TIMING_FEATS:
    if feat in train_df.columns:
        train_df = train_df.with_columns(
            pl.when(pl.col(feat) > SEQ_LEN).then(SENTINEL_NO_EARLY_EVENT)
              .otherwise(pl.col(feat)).alias(feat))
        val_df = val_df.with_columns(
            pl.when(pl.col(feat) > SEQ_LEN).then(SENTINEL_NO_EARLY_EVENT)
              .otherwise(pl.col(feat)).alias(feat))
        test_df = test_df.with_columns(
            pl.when(pl.col(feat) > SEQ_LEN).then(SENTINEL_NO_EARLY_EVENT)
              .otherwise(pl.col(feat)).alias(feat))

print("Calculating statistics and normalizing...")
all_norm_cols = list(set(SEQ_FEATURES + TREATMENT_FEATURES))
train_stats = calculate_train_stats(train_df, all_norm_cols)

train_raw = train_df.clone()
test_raw  = test_df.clone()

train_df = normalize(train_df, train_stats)
val_df   = normalize(val_df,   train_stats)
test_df  = normalize(test_df,  train_stats)
print("✅ Data Loaded and Normalized!")

# Block intervention flags — direct label proxies, must never enter the model
_drop = ["vasopressor_flag", "ventilation_flag"]
train_df = train_df.drop([c for c in _drop if c in train_df.columns])
val_df   = val_df.drop([c for c in _drop if c in val_df.columns])
test_df  = test_df.drop([c for c in _drop if c in test_df.columns])

# ── DATASET ────────────────────────────────────────────────────────────────────
print("Building datasets...")
train_ds = ICUDataset(train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
val_ds   = ICUDataset(val_df,   SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
test_ds  = ICUDataset(test_df,  SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Seq features: {len(train_ds.seq_cols)} | Treat features: {len(train_ds.treat_cols)}")

_short = (train_df.group_by("stay_id")
                  .agg(pl.col("hrs_from_admit").count().alias("n_hours"))
                  .filter(pl.col("n_hours") < SEQ_LEN))
if _short.height > 0:
    print(f"  ⚠ {_short.height} stays shorter than SEQ_LEN={SEQ_LEN} "
          f"— will be zero-padded")
else:
    print(f"  ✅ All stays have ≥ {SEQ_LEN} hours")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

seq_dim   = len(train_ds.seq_cols)
treat_dim = len(train_ds.treat_cols)
print(f"\nModel: seq_dim={seq_dim}, treat_dim={treat_dim}, targets={len(LABEL_COLS)}")

# ── LOSS ───────────────────────────────────────────────────────────────────────
pos_weights = compute_pos_weights(train_ds, max_weight=20.0)
print(f"Pos weights: {pos_weights.cpu().numpy().round(2)}")
criterion = FocalBCEWithLogitsLoss(pos_weight=pos_weights, gamma=1.5, label_smoothing=0.02)

# ── TRAINING FUNCTIONS ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, crit):
    model.train()
    total_loss, n = 0, 0
    for x_seq, x_treat, y in loader:
        x_seq, x_treat, y = x_seq.to(device), x_treat.to(device), y.to(device)
        loss = crit(model(x_seq, x_treat), y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0); n += y.size(0)
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader, crit):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0, 0
    for x_seq, x_treat, y in loader:
        x_seq, x_treat, y = x_seq.to(device), x_treat.to(device), y.to(device)
        logits = model(x_seq, x_treat)
        total_loss += crit(logits, y).item() * y.size(0); n += y.size(0)
        all_logits.append(logits.cpu()); all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    metrics = {"loss": total_loss / max(n, 1)}
    for i, lbl in enumerate(LABEL_COLS):
        y_true, y_prob = labels[:, i], probs[:, i]
        n_pos, n_neg = int(y_true.sum()), int(len(y_true) - y_true.sum())
        if n_pos > 0 and n_neg > 0:
            metrics[f"{lbl}_auroc"] = roc_auc_score(y_true, y_prob)
            metrics[f"{lbl}_auprc"] = average_precision_score(y_true, y_prob)
            metrics[f"{lbl}_brier"] = brier_score_loss(y_true, y_prob)
        else:
            metrics[f"{lbl}_auroc"] = float("nan")
            metrics[f"{lbl}_auprc"] = float("nan")
            metrics[f"{lbl}_brier"] = float("nan")
        metrics[f"{lbl}_n_pos"] = n_pos
    return metrics, probs, labels

def eval_on_split(model, df, crit):
    ds     = ICUDataset(df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return evaluate(model, loader, crit)

def train_model(model, train_loader, val_loader, crit, lr, epochs, patience, tag=""):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, crit)
        v_met, _, _ = evaluate(model, val_loader, crit)
        scheduler.step(v_met["loss"])
        if v_met["loss"] < best_loss:
            best_loss, best_state, wait = v_met["loss"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if ep % 5 == 0 or wait == 0:
            aurocs = [v_met.get(f"{l}_auroc", 0) for l in LABEL_COLS]
            print(f"  {tag}Ep {ep:2d} | tL={t_loss:.4f} vL={v_met['loss']:.4f} "
                  f"mAUROC={np.nanmean(aurocs):.4f} {'*' if wait==0 else ''}")
        if wait >= patience:
            print(f"  {tag}Early stop at epoch {ep}"); break
    if best_state: model.load_state_dict(best_state)
    return model

# ── PHASE 1: TRAIN SOURCE ENSEMBLE ────────────────────────────────────────────
# FIX 9: Train N sources (one per SOURCE_SEED) instead of a single source.
# The median-val-AUROC source is used as `source_state` for all downstream
# operations (model_A init, val_met baseline, B/C init, saving).
print("\n" + "="*60)
print(f"PHASE 1: Training source ensemble ({len(SOURCE_SEEDS)} seeds) on 2008-2013")
print("="*60)

all_source_states   = []   # list of state_dicts, one per SOURCE_SEED
all_source_val_mets = []   # corresponding val metrics

for src_seed in SOURCE_SEEDS:
    print(f"\n--- Source seed {src_seed} ---")
    torch.manual_seed(src_seed); np.random.seed(src_seed)
    m_src = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    m_src = train_model(m_src, train_loader, val_loader, criterion,
                        LR_INIT, EPOCHS, PATIENCE, f"SRC-s{src_seed} ")
    v_met, _, _ = evaluate(m_src, val_loader, criterion)
    all_source_states.append(copy.deepcopy(m_src.state_dict()))
    all_source_val_mets.append(v_met)
    mean_val_auroc = np.nanmean([v_met.get(f"{l}_auroc", np.nan) for l in LABEL_COLS])
    print(f"  Seed {src_seed} val mAUROC = {mean_val_auroc:.4f}")

# Pick MEDIAN-val-AUROC source as the canonical source for downstream use
_src_mean_aurocs = np.array([
    np.nanmean([m.get(f"{l}_auroc", np.nan) for l in LABEL_COLS])
    for m in all_source_val_mets
])
_median_src_idx = int(np.argsort(_src_mean_aurocs)[len(_src_mean_aurocs) // 2])
source_state = all_source_states[_median_src_idx]
val_met      = all_source_val_mets[_median_src_idx]
_median_src_seed = SOURCE_SEEDS[_median_src_idx]

print(f"\nSource ensemble val mAUROC: "
      f"{_src_mean_aurocs.mean():.4f} ± {_src_mean_aurocs.std(ddof=1):.4f}  "
      f"(range [{_src_mean_aurocs.min():.4f}, {_src_mean_aurocs.max():.4f}])")
print(f"Canonical source (median): seed {_median_src_seed} "
      f"(mAUROC={_src_mean_aurocs[_median_src_idx]:.4f})")

print(f"\nCanonical source on VAL:")
for lbl in LABEL_COLS:
    print(f"  {lbl}: AUROC={val_met.get(f'{lbl}_auroc',0):.4f} "
          f"AUPRC={val_met.get(f'{lbl}_auprc',0):.4f}"
          f" (n_pos={val_met.get(f'{lbl}_n_pos',0)})")

# ── INITIALISE ALL THREE RUNS ─────────────────────────────────────────────────
# All three runs initialise from the canonical (median-seed) source state.
model_A = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_A.load_state_dict(source_state); model_A.freeze_all()

model_B = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_B.load_state_dict(source_state); model_B.freeze_physio()

model_C = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_C.load_state_dict(source_state)

print("\nRun A: fully frozen (canonical source)")
print("Run B: LSTM frozen, treat+fusion trainable")
print("Run C: all layers trainable (full fine-tune on post-drift data)")

# ── DRIFT DETECTION ───────────────────────────────────────────────────────────
EXCLUDE_FROM_DRIFT = {"n_distinct_meds"}  # documentation-style feature; not clinically meaningful
TREAT_ONLY = [c for c in TREATMENT_FEATURES if c not in BINARY_COLS and c != "age" and c not in EXCLUDE_FROM_DRIFT]

train_treat_ref = {}
for c in TREAT_ONLY:
    if c in train_df.columns:
        train_treat_ref[c] = train_df.filter(pl.col("hrs_from_admit") == 0)[c].to_numpy().astype(float)

BINARY_MONITOR = [c for c in TREATMENT_FEATURES if c in BINARY_COLS]
train_binary_ref = {}
for c in BINARY_MONITOR:
    if c in train_df.columns:
        train_binary_ref[c] = train_df.filter(pl.col("hrs_from_admit") == 0)[c].to_numpy().astype(float).mean()

print(f"\nDrift monitoring: {len(train_treat_ref)} continuous + {len(train_binary_ref)} binary features")

# ── PHASE 2a: MULTI-SIGNAL DRIFT DETECTION ────────────────────────────────────
print("\n" + "="*60)
print("PHASE 2a: Multi-signal drift detection (PSI + Ensemble Performance)")
print("="*60)

from scipy.stats import ks_2samp

def compute_psi(ref, cur, edges=None, bins=10):
    if len(cur) < 10: return 0.0
    if edges is None:
        edges = np.percentile(ref, np.linspace(0, 100, bins + 1))
        edges[0] -= 1e-6; edges[-1] += 1e-6
    edges = np.unique(edges)
    if len(edges) < 3: return 0.0
    ref_h, _ = np.histogram(ref, bins=edges)
    cur_h, _ = np.histogram(cur, bins=edges)
    n_bins = len(edges) - 1
    ref_p = (ref_h + 1) / (ref_h.sum() + n_bins)
    cur_p = (cur_h + 1) / (cur_h.sum() + n_bins)
    return float(np.sum((ref_p - cur_p) * np.log(ref_p / cur_p)))

def pairwise_distribution_drift(prev_df, curr_df, treat_only_cols,
                                 binary_cols, ref_edges=None):
    """PSI + KS on continuous; rate-delta on binary. Uses t0-only rows."""
    psis, drifted_cont = [], []
    for c in treat_only_cols:
        if c not in prev_df.columns or c not in curr_df.columns: continue
        ref = prev_df[c].to_numpy().astype(float)
        cur = curr_df[c].to_numpy().astype(float)
        if len(ref) < 20 or len(cur) < 20: continue
        edges = (ref_edges or {}).get(c)
        psi_val = compute_psi(ref, cur, edges=edges)
        _, ks_p = ks_2samp(ref, cur)
        psis.append(psi_val)
        if psi_val > PSI_THRESH or ks_p < 0.01:
            drifted_cont.append((c, psi_val, ks_p))

    bdeltas, drifted_bin = [], []
    for c in binary_cols:
        if c not in prev_df.columns or c not in curr_df.columns: continue
        ref_rate = prev_df[c].to_numpy().astype(float).mean()
        cur_rate = curr_df[c].to_numpy().astype(float).mean()
        delta     = abs(cur_rate - ref_rate)
        rel_delta = delta / (ref_rate + 1e-9)
        bdeltas.append(delta)
        if ref_rate >= 0.02:
            flagged = delta > BINARY_DELTA_THR or rel_delta > 0.20
        else:
            flagged = delta > BINARY_DELTA_THR
        if flagged:
            drifted_bin.append((c, delta))

    mean_psi  = float(np.mean(psis))    if psis    else 0.0
    mean_bin  = float(np.mean(bdeltas)) if bdeltas else 0.0
    score     = mean_psi + 0.5 * mean_bin
    n_drifted = len(drifted_cont) + len(drifted_bin)
    return score, n_drifted, drifted_cont, drifted_bin

# ── CHG 2: Simplified ensemble performance-drop helper ────────────────────────
# majority_fired and std-gate removed. std_drop kept for diagnostic logging only.
# Returns (mean_drop, std_drop, per_lbl_mean_drops, n_sig_ensemble).
def ensemble_perf_drop_vs_baseline(curr_norm_df, crit, baseline_met_per_seed):
    """
    Compute mean AUROC drop across ALL source seeds vs their own per-seed
    val baseline (baseline_met_per_seed aligned with all_source_states).

    Fires when mean_drop > AUROC_DRIFT_THR on >= MIN_PERF_DROPS labels.
    std_drop is returned for diagnostic logging only — not used in the trigger.

    Returns (mean_drop, std_drop, per_lbl_mean_drops, n_sig_ensemble).
    """
    N = len(all_source_states)
    per_seed_drops = []   # list of per_lbl_drops dicts

    for idx, (state, base_met) in enumerate(zip(all_source_states, baseline_met_per_seed)):
        _m = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
        _m.load_state_dict(state)
        _m.freeze_all()
        curr_met, _, _ = eval_on_split(_m, curr_norm_df, crit)
        drops = {}
        for lbl in LABEL_COLS:
            prev_a = base_met.get(f"{lbl}_auroc", np.nan)
            curr_a = curr_met.get(f"{lbl}_auroc", np.nan)
            if np.isnan(prev_a) or np.isnan(curr_a):
                drops[lbl] = np.nan
            else:
                drops[lbl] = prev_a - curr_a
        per_seed_drops.append(drops)
        del _m

    # Per-label mean and std across seeds
    per_lbl_mean_drops = {}
    per_lbl_std_drops  = {}
    for lbl in LABEL_COLS:
        vals = np.array([d.get(lbl, np.nan) for d in per_seed_drops])
        vals = vals[~np.isnan(vals)]
        per_lbl_mean_drops[lbl] = float(np.mean(vals)) if len(vals) else np.nan
        per_lbl_std_drops[lbl]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    all_mean_vals = [v for v in per_lbl_mean_drops.values() if not np.isnan(v)]
    mean_drop = float(np.mean(all_mean_vals)) if all_mean_vals else 0.0
    all_std_vals  = [v for v in per_lbl_std_drops.values()  if not np.isnan(v)]
    std_drop  = float(np.mean(all_std_vals))  if all_std_vals  else 0.0

    # CHG 2: fire on mean_drop > AUROC_DRIFT_THR alone — no std-gate, no majority vote.
    n_sig_ensemble = sum(
        1 for lbl in LABEL_COLS
        if (not np.isnan(per_lbl_mean_drops.get(lbl, np.nan))
            and per_lbl_mean_drops[lbl] > AUROC_DRIFT_THR)
    )

    return mean_drop, std_drop, per_lbl_mean_drops, n_sig_ensemble

# ── perf_drop_vs_baseline kept for reference (used in Phase 3 / eval helpers) ─
def perf_drop_vs_baseline(model_a, curr_norm_df, crit, baseline_met):
    """Compute AUROC drop on curr group vs an externally-supplied baseline metric dict."""
    curr_met, _, _ = eval_on_split(model_a, curr_norm_df, crit)
    drops, n_sig = {}, 0
    for lbl in LABEL_COLS:
        prev_a = baseline_met.get(f"{lbl}_auroc", np.nan)
        curr_a = curr_met.get(f"{lbl}_auroc", np.nan)
        if np.isnan(prev_a) or np.isnan(curr_a):
            drops[lbl] = np.nan
            continue
        drop = prev_a - curr_a
        drops[lbl] = drop
        if drop > AUROC_DRIFT_THR:
            n_sig += 1
    mean_drop = float(np.nanmean(list(drops.values())))
    return mean_drop, drops, n_sig

# ── raw data for distribution drift (PSI needs unnormalized values) ────────
raw_train_all = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
raw_test_all  = load_enriched_split(BASE_PATH, "test",  SEQ_FEATURES, TREATMENT_FEATURES)

raw_all_t0 = pl.concat([
    raw_train_all.filter(pl.col("hrs_from_admit") == 0),
    raw_test_all.filter(pl.col("hrs_from_admit") == 0),
])

norm_all = pl.concat([train_df, val_df, test_df])

ALL_GROUPS_ORDERED = sorted(raw_all_t0["anchor_year_group"].unique().to_list())
TRAIN_GROUP_SET    = set(TRAIN_YEARS)
TEST_GROUP_SET     = set(ALL_GROUPS_ORDERED) - TRAIN_GROUP_SET

print(f"Groups: {ALL_GROUPS_ORDERED}")

group_t0   = {g: raw_all_t0.filter(pl.col("anchor_year_group") == g)
              for g in ALL_GROUPS_ORDERED}
group_norm = {g: norm_all.filter(pl.col("anchor_year_group") == g)
              for g in ALL_GROUPS_ORDERED}

# FIX 7: pool reference bin edges across ALL training groups (not just oldest).
ref_edges = {}
train_groups_for_edges = [g for g in ALL_GROUPS_ORDERED if g in TRAIN_GROUP_SET]
for c in TREAT_ONLY:
    refs = []
    for g in train_groups_for_edges:
        if c in group_t0[g].columns:
            arr = group_t0[g][c].to_numpy().astype(float)
            if len(arr) >= 5:
                refs.append(arr)
    if refs:
        pooled = np.concatenate(refs)
        if len(pooled) >= 10:
            e = np.percentile(pooled, np.linspace(0, 100, 11))
            e[0] -= 1e-6; e[-1] += 1e-6
            ref_edges[c] = e
print(f"Pooled ref_edges from {len(train_groups_for_edges)} training groups for {len(ref_edges)} features")

# ── sequential pairwise evaluation ────────────────────────────────────────────
# CHG 3: ensemble_perf_drop_vs_baseline is now called in the pairwise loop
#        (previously only perf_drop_vs_baseline was called, leaving std_drop
#        and majority_fired undefined at runtime — crash bug).
# CHG 5: all_source_val_mets_per_seed alias removed; use all_source_val_mets directly.
pair_results = {}

for i in range(1, len(ALL_GROUPS_ORDERED)):
    prev_g = ALL_GROUPS_ORDERED[i - 1]
    curr_g = ALL_GROUPS_ORDERED[i]

    if group_t0[prev_g].height < 30 or group_t0[curr_g].height < 30:
        print(f"  Skipping {prev_g} → {curr_g}: too few stays"); continue

    # 1. distribution drift (on raw t0)
    dist_score, n_drifted, d_cont, d_bin = pairwise_distribution_drift(
        group_t0[prev_g], group_t0[curr_g], TREAT_ONLY, BINARY_MONITOR,
        ref_edges=ref_edges
    )

    # 2. performance drift — ensemble vs val baseline (FIX 2 + CHG 3)
    # Using ensemble for all pairs (train→train and train→test) so the
    # performance leg is consistent and std_drop is always available for logs.
    mean_drop, std_drop, per_lbl_drops, n_perf_sig = ensemble_perf_drop_vs_baseline(
        group_norm[curr_g], criterion, all_source_val_mets
    )

    pair_results[(prev_g, curr_g)] = {
        "dist_score":      dist_score,
        "n_drifted":       n_drifted,
        "drifted_cont":    d_cont,
        "drifted_bin":     d_bin,
        "mean_auroc_drop": mean_drop,
        "std_auroc_drop":  std_drop,   # diagnostic only
        "per_lbl_drops":   per_lbl_drops,
        "n_perf_sig":      n_perf_sig,
        "baseline_tag":    "val (held-out)",
    }

    prev_is_train = prev_g in TRAIN_GROUP_SET
    curr_is_train = curr_g in TRAIN_GROUP_SET
    tag = ("TRAIN→TRAIN" if (prev_is_train and curr_is_train)
           else "TRAIN→TEST" if prev_is_train
           else "TEST→TEST")
    print(f"  [{tag}] {prev_g} → {curr_g}: "
          f"dist={dist_score:.4f} (n_feat={n_drifted}) | "
          f"ΔAUROC={mean_drop:+.4f} ± {std_drop:.4f} "
          f"(n_sig={n_perf_sig})  "
          f"[baseline: val (held-out)]")

# ── calibrate distribution baseline from training-era pairs ───────────────
train_groups_ord = [g for g in ALL_GROUPS_ORDERED if g in TRAIN_GROUP_SET]
train_pairs = [
    (train_groups_ord[i-1], train_groups_ord[i])
    for i in range(1, len(train_groups_ord))
    if (train_groups_ord[i-1], train_groups_ord[i]) in pair_results
]

if train_pairs:
    baseline_dist  = float(np.mean([pair_results[p]["dist_score"]    for p in train_pairs]))
    dist_threshold = baseline_dist * JUMP_FACTOR
    print(f"\nBaseline dist={baseline_dist:.4f} → threshold={dist_threshold:.4f}")
    print(f"Per-label AUROC trigger: mean_drop > {AUROC_DRIFT_THR} on >= {MIN_PERF_DROPS} label(s) "
          f"(ensemble of {len(SOURCE_SEEDS)} source seeds)")
else:
    baseline_dist  = 0.0
    dist_threshold = PSI_THRESH
    print(f"\nNo train pairs to calibrate dist baseline — falling back to PSI_THRESH={PSI_THRESH}")

# ── drift onset: OR-gate — fires on distribution drift OR performance drop ─────
# CHG 6: clean OR-gate. majority_fired removed entirely.
# Either leg alone is sufficient to trigger adaptation.
test_groups_ord = [g for g in ALL_GROUPS_ORDERED if g in TEST_GROUP_SET]
test_pairs = [
    (ALL_GROUPS_ORDERED[ALL_GROUPS_ORDERED.index(g) - 1], g)
    for g in test_groups_ord
    if ALL_GROUPS_ORDERED.index(g) > 0
]

drift_group = None
for prev_g, curr_g in test_pairs:
    if (prev_g, curr_g) not in pair_results: continue
    res = pair_results[(prev_g, curr_g)]

    # Data drift leg: distribution shift vs pooled training reference
    dist_exceeded  = (res["dist_score"] > dist_threshold
                      and res["n_drifted"] >= MIN_FEAT_DRIFTED)

    # Performance drift leg: ensemble mean AUROC drop vs val baseline
    perf_exceeded  = res["n_perf_sig"] >= MIN_PERF_DROPS

    # OR-gate: either signal alone is sufficient
    trigger = dist_exceeded or perf_exceeded

    print(f"\n  Test pair {prev_g} → {curr_g}: "
          f"dist_ok={dist_exceeded} perf_ok={perf_exceeded} "
          f"(n_sig={res['n_perf_sig']}) "
          f"→ {'⚠ DRIFT' if trigger else '✓ stable'}")
    for lbl, drop in res["per_lbl_drops"].items():
        print(f"    {lbl}: mean_ΔAUROC={drop:+.4f}  "
              f"std={res['std_auroc_drop']:.4f}")

    if trigger and drift_group is None:
        drift_group = curr_g

# ── CUMULATIVE PERFORMANCE TRACKER ────────────────────────────────────────────
# CHG 7: std-gate removed. Fires on mean_drop > AUROC_DRIFT_THR alone.
# CHG 8: torch.cuda.empty_cache() added after del _m_tmp to prevent GPU OOM.
print("\n--- Cumulative performance tracking vs source ENSEMBLE val baseline ---")

# Use mean across all seeds as the ensemble val baseline for each label
source_val_aurocs_ensemble = {}
for lbl in LABEL_COLS:
    vals = np.array([m.get(f"{lbl}_auroc", np.nan) for m in all_source_val_mets])
    vals = vals[~np.isnan(vals)]
    source_val_aurocs_ensemble[lbl] = float(np.mean(vals)) if len(vals) else np.nan

print("  Source ensemble val baseline: " +
      " | ".join(f"{l.replace('label_','')}={v:.4f}"
                 for l, v in source_val_aurocs_ensemble.items()))

for lbl in LABEL_COLS:
    vals = np.array([m.get(f"{lbl}_auroc", np.nan) for m in all_source_val_mets])
    vals = vals[~np.isnan(vals)]
    if len(vals) > 1:
        print(f"    {lbl}: range [{vals.min():.4f}, {vals.max():.4f}]  "
              f"std={vals.std(ddof=1):.4f}")

cumulative_drift_group = None
for g in ALL_GROUPS_ORDERED:
    if g in TRAIN_GROUP_SET:
        continue

    # Evaluate ALL sources on this group and aggregate
    per_seed_grp_aurocs = {}
    for idx, state in enumerate(all_source_states):
        _m_tmp = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
        _m_tmp.load_state_dict(state)
        _m_tmp.freeze_all()
        grp_met, _, _ = eval_on_split(_m_tmp, group_norm[g], criterion)
        for lbl in LABEL_COLS:
            per_seed_grp_aurocs.setdefault(lbl, []).append(
                grp_met.get(f"{lbl}_auroc", np.nan))
        del _m_tmp
        # CHG 8: free GPU memory after each temporary model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    drops_mean, drops_std, n_dropped = {}, {}, 0
    for lbl in LABEL_COLS:
        grp_vals = np.array(per_seed_grp_aurocs[lbl])
        grp_vals = grp_vals[~np.isnan(grp_vals)]
        grp_auroc_mean = float(np.mean(grp_vals)) if len(grp_vals) else np.nan
        grp_auroc_std  = float(np.std(grp_vals, ddof=1)) if len(grp_vals) > 1 else 0.0
        src_auroc      = source_val_aurocs_ensemble[lbl]
        if np.isnan(grp_auroc_mean) or np.isnan(src_auroc):
            drops_mean[lbl] = np.nan
            drops_std[lbl]  = 0.0
            continue
        drop = src_auroc - grp_auroc_mean
        drops_mean[lbl] = drop
        drops_std[lbl]  = grp_auroc_std
        # CHG 7: fire on mean_drop > AUROC_DRIFT_THR alone (std-gate removed).
        if drop > AUROC_DRIFT_THR:
            n_dropped += 1

    all_valid = [d for d in drops_mean.values() if not np.isnan(d)]
    mean_drop  = float(np.mean(all_valid)) if all_valid else 0.0
    triggered  = n_dropped >= MIN_PERF_DROPS
    print(f"\n  {g}: ensemble mean_drop_vs_val={mean_drop:+.4f}  "
          f"n_labels_dropped={n_dropped}  "
          f"{'⚠ CUMULATIVE DRIFT' if triggered else '✓ within tolerance'}")
    for lbl in LABEL_COLS:
        d = drops_mean.get(lbl, np.nan)
        s = drops_std.get(lbl, 0.0)
        if not np.isnan(d):
            print(f"    {lbl}: mean_drop={d:+.4f}  seed_std={s:.4f}")
    if triggered and cumulative_drift_group is None:
        cumulative_drift_group = g

# CHG 11: commented-out cumulative drift override block removed entirely.
# The cumulative tracker is diagnostic only; pairwise OR-gate controls drift_group.

# ── build pre/post splits ─────────────────────────────────────────────────────
if drift_group:
    drift_idx   = ALL_GROUPS_ORDERED.index(drift_group)
    pre_groups  = ALL_GROUPS_ORDERED[:drift_idx]
    post_groups = ALL_GROUPS_ORDERED[drift_idx:]
    print(f"\n✅ Drift onset: {drift_group}")
    print(f"   Pre : {pre_groups}")
    print(f"   Post: {post_groups}")
else:
    pre_groups  = list(ALL_GROUPS_ORDERED)
    post_groups = []
    print("\n✅ No drift detected")

test_pre  = (test_df.filter(pl.col("anchor_year_group").is_in(pre_groups))
             .filter(pl.col("anchor_year_group").is_in(list(TEST_GROUP_SET)))
             if pre_groups else pl.DataFrame())
test_post = (test_df.filter(pl.col("anchor_year_group").is_in(post_groups))
             if post_groups else pl.DataFrame())

pre_cp_stays  = test_pre["stay_id"].unique().to_list()  if test_pre.height  > 0 else []
post_cp_stays = test_post["stay_id"].unique().to_list() if test_post.height > 0 else []
drift_tag     = drift_group if drift_group else "no-drift"

print(f"\nPre-drift test stays:  {len(pre_cp_stays)}")
print(f"Post-drift test stays: {len(post_cp_stays)}")

# ── collect drifted feature names for downstream use ─────────────────────────
drifted_post = []
for g in post_groups:
    if g not in TEST_GROUP_SET:
        continue
    idx = ALL_GROUPS_ORDERED.index(g)
    if idx == 0:
        continue
    prev_g = ALL_GROUPS_ORDERED[idx - 1]
    key    = (prev_g, g)
    if key in pair_results:
        drifted_post.extend([c for c, _, _ in pair_results[key]["drifted_cont"]])
        drifted_post.extend([c for c, _ in pair_results[key]["drifted_bin"]])
drifted_post = list(set(drifted_post))
print(f"\nDrifted features in post-drift groups: {len(drifted_post)}")
for f in sorted(drifted_post)[:10]:
    print(f"  ⚠ {f}")

if len(pre_cp_stays) == 0 and drift_group is not None:
    print("\n⚠  NOTE: Drift onset was detected at the earliest test group.")
    print("   No stable pre-drift test period is available for comparison.")
    print("   Val set serves as the pre-drift reference.")

# ── drift signal bar plot ──────────────────────────────────────────────────────
pair_labels = [f"{a[-7:]}\n→{b[-7:]}" for a, b in pair_results.keys()]
pair_scores = [v["dist_score"] for v in pair_results.values()]
pair_nd     = [v["n_drifted"] for v in pair_results.values()]
is_train    = [(b in TRAIN_GROUP_SET) for _, b in pair_results.keys()]

fig, ax = plt.subplots(figsize=(max(8, len(pair_labels) * 2), 4))
colors = ["#1f77b4" if t else "#d62728" for t in is_train]
bars   = ax.bar(range(len(pair_labels)), pair_scores, color=colors,
                alpha=0.75, edgecolor="black")
for idx, (bar, nd) in enumerate(zip(bars, pair_nd)):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"n={nd}", ha="center", va="bottom", fontsize=8)
ax.axhline(dist_threshold, color="red", ls="--", lw=1.5,
           label=f"Drift threshold ({dist_threshold:.3f})")
ax.axhline(baseline_dist,    color="gray", ls=":",  lw=1.2,
           label=f"Baseline ({baseline_dist:.3f})")
if drift_group is not None:
    onset_pair_idx = next(
        (i for i, (_, b) in enumerate(pair_results.keys()) if b == drift_group), None
    )
    if onset_pair_idx is not None:
        ax.axvline(onset_pair_idx - 0.5, color="red", ls="-", lw=2,
                   label=f"Drift onset: {drift_group}")
from matplotlib.patches import Patch
legend_handles = [
    Patch(color="#1f77b4", alpha=0.75, label="Train-era pair"),
    Patch(color="#d62728", alpha=0.75, label="Test-era pair"),
]
ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0][2:], fontsize=8)
ax.set_xticks(range(len(pair_labels)))
ax.set_xticklabels(pair_labels, fontsize=8)
ax.set_ylabel("Sequential drift score")
ax.set_title("Sequential pairwise drift detection — consecutive group PSI\n"
             f"(Performance leg: ensemble of {len(SOURCE_SEEDS)} source seeds)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(SAVE_PATH / "drift_changepoint.png", dpi=150, bbox_inches="tight")
print(f"Saved → {SAVE_PATH / 'drift_changepoint.png'}")

# ── DRIFT EXPLAINABILITY ───────────────────────────────────────────────────────
test_pre_raw  = test_raw.filter(pl.col("anchor_year_group").is_in(pre_groups))  \
                if pre_groups else pl.DataFrame()
test_post_raw = test_raw.filter(pl.col("anchor_year_group").is_in(post_groups)) \
                if post_groups else pl.DataFrame()

explainer = DriftExplainer(
    train_df       = train_raw,
    test_pre_df    = test_pre_raw,
    test_post_df   = test_post_raw,
    treat_features = [f for f in TREATMENT_FEATURES if f not in EXCLUDE_FROM_DRIFT],
    binary_cols    = BINARY_COLS,
    label_cols     = LABEL_COLS,
    psi_thresh     = PSI_THRESH,
    save_path      = SAVE_PATH,
)
drift_report = explainer.explain_drift()

# ── PHASE 2b: ADAPTATION (multi-seed, subject-level split) ────────────────────
adaptation_performed = False
eval_post_stays = []
adapt_train_stays, adapt_val_stays, buf_pre_stays = [], [], []

SEEDS = [42, 123, 7, 2024, 99]
seed_runs    = {"B": [], "C": []}
seed_states  = {"B": [], "C": []}   # FIX 5: cache all seed states for median selection

if len(drifted_post) > 0 and test_post.height > 0 and test_post["stay_id"].n_unique() > 50:
    print(f"\n--- Drift detected ({len(drifted_post)} features). "
          f"Running adaptation under {len(SEEDS)} seeds ---")

    # FIX 1: SUBJECT-LEVEL split (not stay-level). MIMIC-IV patients can have
    # multiple stays; random stay-id splits leak the same person across folds.
    post_subjects = test_post.filter(pl.col("hrs_from_admit") == 0)["subject_id"].unique().to_list()
    np.random.seed(SEED)
    np.random.shuffle(post_subjects)
    n_total_subj       = len(post_subjects)
    n_adapt_train_subj = int(n_total_subj * 0.30)
    n_adapt_val_subj   = int(n_total_subj * 0.10)
    adapt_train_subjects = post_subjects[:n_adapt_train_subj]
    adapt_val_subjects   = post_subjects[n_adapt_train_subj:n_adapt_train_subj + n_adapt_val_subj]
    eval_post_subjects   = post_subjects[n_adapt_train_subj + n_adapt_val_subj:]

    # Map subjects back to ALL their stays in post-drift
    adapt_train_stays = (test_post.filter(pl.col("subject_id").is_in(adapt_train_subjects))
                         ["stay_id"].unique().to_list())
    adapt_val_stays   = (test_post.filter(pl.col("subject_id").is_in(adapt_val_subjects))
                         ["stay_id"].unique().to_list())
    eval_post_stays   = (test_post.filter(pl.col("subject_id").is_in(eval_post_subjects))
                         ["stay_id"].unique().to_list())

    print(f"  Subjects (post-drift): total={n_total_subj} | "
          f"adapt_train={n_adapt_train_subj} | adapt_val={n_adapt_val_subj} | "
          f"eval={n_total_subj - n_adapt_train_subj - n_adapt_val_subj}")
    print(f"  Stays    (post-drift): adapt_train={len(adapt_train_stays)} | "
          f"adapt_val={len(adapt_val_stays)} | eval={len(eval_post_stays)}")

    # Sanity-check that subject sets really are disjoint
    eval_post_subj_set = set(eval_post_subjects)
    assert not (set(adapt_train_subjects) & eval_post_subj_set), "subject leak: train↔eval"
    assert not (set(adapt_val_subjects)   & eval_post_subj_set), "subject leak: val↔eval"

    # FIX 8: pre-drift buffer sorted by `intime` (true recency) with
    # subject-level guard so buffer subjects can't overlap eval_post subjects.
    if test_pre.height > 0:
        pre_first_row = (test_pre.filter(pl.col("hrs_from_admit") == 0)
                 .sort(["anchor_year_group", "intime"]))
        pre_stays_sorted = pre_first_row["stay_id"].to_list()
        buf_stays_cand   = (pre_stays_sorted[-BUFFER_SIZE:]
                            if len(pre_stays_sorted) > BUFFER_SIZE
                            else pre_stays_sorted)
        buf_pre_df = test_pre.filter(pl.col("stay_id").is_in(buf_stays_cand))

        # Drop any subjects that also appear in eval_post_subjects
        buf_subjects     = set(buf_pre_df["subject_id"].unique().to_list())
        leaking_subjects = buf_subjects & eval_post_subj_set
        if leaking_subjects:
            buf_pre_df = buf_pre_df.filter(~pl.col("subject_id").is_in(list(leaking_subjects)))
            print(f"  ⚠ Removed {len(leaking_subjects)} subjects from pre-drift "
                  f"buffer (also in eval_post)")
        buf_pre_stays = buf_pre_df["stay_id"].unique().to_list()
    else:
        buf_pre_stays = []
        buf_pre_df    = pl.DataFrame()

    adapt_train_df = test_post.filter(pl.col("stay_id").is_in(adapt_train_stays))
    adapt_val_df   = test_post.filter(pl.col("stay_id").is_in(adapt_val_stays))
    eval_post_df   = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))

    if buf_pre_df.height > 0:
        combined_train_df = pl.concat([buf_pre_df, adapt_train_df])
    else:
        combined_train_df = adapt_train_df
        print("  ⚠ No pre-drift buffer available — adapting on post-drift data only")

    n_buf     = buf_pre_df.filter(pl.col("hrs_from_admit") == 0).height \
                if buf_pre_df.height > 0 else 0
    n_post_tr = adapt_train_df.filter(pl.col("hrs_from_admit") == 0).height
    buf_ratio = n_buf / (n_buf + n_post_tr + 1e-9)
    print(f"  Adapt-train: {n_post_tr} post-drift + {n_buf} pre-drift buffer "
          f"({buf_ratio*100:.1f}% buffer)")
    if buf_ratio > 0.5:
        print(f"  ⚠ Buffer dominates ({buf_ratio*100:.1f}%) — "
              f"model may not adapt sufficiently")

    adapt_train_ds = ICUDataset(combined_train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    adapt_val_ds   = ICUDataset(adapt_val_df,      SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)

    adapt_pos_weights = compute_pos_weights(adapt_train_ds, max_weight=15.0)
    adapt_criterion   = FocalBCEWithLogitsLoss(
        pos_weight=adapt_pos_weights, gamma=1.0, label_smoothing=0.05)

    for s in SEEDS:
        print(f"\n=== Seed {s} ===")
        torch.manual_seed(s); np.random.seed(s)
        atr = DataLoader(adapt_train_ds, batch_size=BATCH_SIZE, shuffle=True)
        ava = DataLoader(adapt_val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        mB = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
        mB.load_state_dict(source_state); mB.unfreeze_adaptive()
        mB = train_model(mB, atr, ava, adapt_criterion,
                         LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE, f"B-s{s} ")

        mC = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
        mC.load_state_dict(source_state); mC.unfreeze_all()
        mC = train_model(mC, atr, ava, adapt_criterion,
                         LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE, f"C-s{s} ")

        mB_met, _, _ = eval_on_split(mB, eval_post_df, criterion)
        mC_met, _, _ = eval_on_split(mC, eval_post_df, criterion)
        seed_runs["B"].append(mB_met)
        seed_runs["C"].append(mC_met)
        seed_states["B"].append(copy.deepcopy(mB.state_dict()))   # FIX 5
        seed_states["C"].append(copy.deepcopy(mC.state_dict()))   # FIX 5

    adaptation_performed = True

    # Aggregate across seeds
    print("\n--- Multi-seed post-drift AUROC (mean ± std across {} seeds) ---".format(len(SEEDS)))
    for lbl in LABEL_COLS:
        for run in ["B", "C"]:
            vals = np.array([m.get(f"{lbl}_auroc", np.nan) for m in seed_runs[run]])
            vals = vals[~np.isnan(vals)]
            if len(vals):
                print(f"  {run} {lbl:<22} {vals.mean():.4f} ± {vals.std(ddof=1):.4f}  "
                      f"(seeds: {np.round(vals,4).tolist()})")

    # ── FIX 5: pick MEDIAN seed for each run (not arbitrary seed 42) ──────
    def _pick_median_idx(metric_dicts):
        means = np.array([np.nanmean([m.get(f"{l}_auroc", np.nan) for l in LABEL_COLS])
                          for m in metric_dicts])
        return int(np.argsort(means)[len(means) // 2])

    median_idx_B = _pick_median_idx(seed_runs["B"])
    median_idx_C = _pick_median_idx(seed_runs["C"])
    print(f"\nMedian-seed selection:  B → seed {SEEDS[median_idx_B]} | "
          f"C → seed {SEEDS[median_idx_C]}")

    model_B = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    model_B.load_state_dict(seed_states["B"][median_idx_B])
    model_C = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    model_C.load_state_dict(seed_states["C"][median_idx_C])

    # ── adaptation explanation (on the median-seed models) ────────────────
    # CHG 10: source model reconstructed once here and reused below in the
    #         IG export block (previously reconstructed twice with identical weights).
    src_model_shared = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    src_model_shared.load_state_dict(source_state)
    src_model_shared.freeze_all()

    if 'explainer' in dir():
        explainer.explain_adaptation_update(
            model_before   = src_model_shared,
            model_after    = model_B,
            adapt_train_ds = adapt_train_ds,
            device         = device,
            run_tag        = "RunB",
        )
        explainer.explain_adaptation_update(
            model_before   = src_model_shared,
            model_after    = model_C,
            adapt_train_ds = adapt_train_ds,
            device         = device,
            run_tag        = "RunC",
        )

    # Sanity print — variance flag (helps interpret B vs C)
    stds_B = [np.nanstd([m.get(f"{l}_auroc", np.nan) for m in seed_runs["B"]], ddof=1)
              for l in LABEL_COLS]
    stds_C = [np.nanstd([m.get(f"{l}_auroc", np.nan) for m in seed_runs["C"]], ddof=1)
              for l in LABEL_COLS]
    print(f"\nMean per-label seed-std:  B={np.nanmean(stds_B):.4f} | "
          f"C={np.nanmean(stds_C):.4f}  "
          f"({'C noisier (overfit risk)' if np.nanmean(stds_C) > np.nanmean(stds_B) else 'B noisier'})")

else:
    print("\nNo significant drift — Run B and Run C unchanged")
    if test_post.height > 0:
        post_stays = test_post.filter(pl.col("hrs_from_admit") == 0)["stay_id"].unique().to_list()
    else:
        post_stays = []
    eval_post_stays = post_stays
    # src_model_shared not needed when adaptation is not performed
    src_model_shared = None

# ── PHASE 3: FINAL COMPARISON ─────────────────────────────────────────────────
print("\n" + "="*60 + "\nPHASE 3: Run A vs Run B vs Run C\n" + "="*60)

# For pre-drift splits, all runs used the SAME source weights —
# use source-init "pre" models so the comparison is honest.
model_B_pre = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_B_pre.load_state_dict(source_state); model_B_pre.freeze_all()

model_C_pre = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_C_pre.load_state_dict(source_state); model_C_pre.freeze_all()

results = {}

print("\n--- Val (source weights, pre-adaptation — should be identical) ---")
met_a_val, _, _ = eval_on_split(model_A,     val_df, criterion)
met_b_val, _, _ = eval_on_split(model_B_pre, val_df, criterion)
met_c_val, _, _ = eval_on_split(model_C_pre, val_df, criterion)
results["Val (source)"] = {"A": met_a_val, "B": met_b_val, "C": met_c_val}

print("--- Test-Pre (source weights, pre-adaptation — should be identical) ---")
if test_pre.height > 0:
    met_a_pre, _, _ = eval_on_split(model_A,     test_pre, criterion)
    met_b_pre, _, _ = eval_on_split(model_B_pre, test_pre, criterion)
    met_c_pre, _, _ = eval_on_split(model_C_pre, test_pre, criterion)
    results[f"Test-Pre  (≤ {drift_tag})"] = {"A": met_a_pre, "B": met_b_pre, "C": met_c_pre}
else:
    print("  ⚠ test_pre is empty — drift onset was at earliest test group.")
    print("  Skipping Test-Pre evaluation. Val (source) serves as pre-drift reference.")

# Post-drift — A frozen, B partially adapted, C fully adapted (median-seed)
verdicts = {}
if len(eval_post_stays) > 0 and adaptation_performed:
    eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))
    print(f"--- Test-Post held-out ({len(eval_post_stays)} stays) "
          "— A frozen | B partial adapt | C full adapt (median-seed) ---")
    met_a_post, probs_a, labels_a = eval_on_split(model_A, eval_post_df, criterion)
    met_b_post, probs_b, labels_b = eval_on_split(model_B, eval_post_df, criterion)
    met_c_post, probs_c, labels_c = eval_on_split(model_C, eval_post_df, criterion)
    results[f"Test-Post (> {drift_tag})"] = {"A": met_a_post, "B": met_b_post, "C": met_c_post}

    if adaptation_performed:
        assert labels_a.shape == labels_b.shape == labels_c.shape, \
            "Label shape mismatch across runs — DataLoader order inconsistent"
        assert np.array_equal(labels_a, labels_b) and \
               np.array_equal(labels_a, labels_c), \
            "Ground-truth labels differ across runs — evaluation not aligned"
        print("✅ DataLoader alignment verified (labels_a == labels_b == labels_c)")

    print("\n--- Post-drift AUROC with 95% bootstrap CIs ---")
    print(f"  {'Label':<22} {'A (95% CI)':<22} {'B (95% CI)':<22} {'C (95% CI)':<22} {'B−A':>8} {'C−A':>8}")
    ci_table = {}
    for i, lbl in enumerate(LABEL_COLS):
        y_t = labels_a[:, i]
        if y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue
        a = bootstrap_ci(y_t, probs_a[:, i], roc_auc_score, seed=1)
        b = bootstrap_ci(y_t, probs_b[:, i], roc_auc_score, seed=1)
        c = bootstrap_ci(y_t, probs_c[:, i], roc_auc_score, seed=1)
        ci_table[lbl] = {"A": a, "B": b, "C": c}
        print(f"  {lbl:<22} {a[0]:.3f} [{a[1]:.3f},{a[2]:.3f}]  "
              f"{b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}]  "
              f"{c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]  "
              f"{b[0]-a[0]:+.4f} {c[0]-a[0]:+.4f}")

    # Paired bootstrap for B vs C
    rng = np.random.default_rng(2)
    n_boot, n = 1000, len(labels_a)
    print("\n--- Paired bootstrap: ΔAUROC (B − C) on held-out post-drift ---")
    for i, lbl in enumerate(LABEL_COLS):
        y_t = labels_a[:, i]
        if y_t.sum() == 0 or y_t.sum() == n: continue
        diffs = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            yt = y_t[idx]
            if yt.sum() == 0 or yt.sum() == n: continue
            diffs.append(roc_auc_score(yt, probs_b[idx, i]) -
                         roc_auc_score(yt, probs_c[idx, i]))
        diffs = np.array(diffs)
        print(f"  {lbl:<22} ΔB−C = {diffs.mean():+.4f}  "
              f"[{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}]  "
              f"P(B>C)={(diffs>0).mean():.2f}")

    # Verdicts
    for i, lbl in enumerate(LABEL_COLS):
        y_t = labels_a[:, i]
        if y_t.sum() == 0 or y_t.sum() == n: continue
        diffs = []
        rng2 = np.random.default_rng(2)
        for _ in range(n_boot):
            idx = rng2.integers(0, n, n)
            yt = y_t[idx]
            if yt.sum() == 0 or yt.sum() == n: continue
            diffs.append(roc_auc_score(yt, probs_b[idx, i]) -
                         roc_auc_score(yt, probs_c[idx, i]))
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        if lo > 0:    verdicts[lbl] = "B"
        elif hi < 0:  verdicts[lbl] = "C"
        else:         verdicts[lbl] = "tie"
else:
    print("--- No post-drift evaluation (no drift detected or no adaptation) ---")

# ── METRIC GAIN EXPLANATION ───────────────────────────────────────────────────
if 'explainer' in dir():
    explainer.explain_metric_gains(results, LABEL_COLS)

# ── PRINT RESULTS ──────────────────────────────────────────────────────────────
for split_name, r in results.items():
    print(f"\n{split_name}:")
    hdr = (f"  {'Label':<25} "
           f"{'A_AUROC':>8} {'B_AUROC':>8} {'B-A':>7} "
           f"{'C_AUROC':>8} {'C-A':>7} | "
           f"{'A_AUPRC':>8} {'B_AUPRC':>8} {'B-A':>7} "
           f"{'C_AUPRC':>8} {'C-A':>7} | n_pos")
    print(hdr)
    print("  " + "-"*len(hdr))
    for lbl in LABEL_COLS:
        a_roc = r["A"].get(f"{lbl}_auroc", float("nan"))
        b_roc = r["B"].get(f"{lbl}_auroc", float("nan"))
        c_roc = r["C"].get(f"{lbl}_auroc", float("nan"))
        a_prc = r["A"].get(f"{lbl}_auprc", float("nan"))
        b_prc = r["B"].get(f"{lbl}_auprc", float("nan"))
        c_prc = r["C"].get(f"{lbl}_auprc", float("nan"))
        n_pos = r["A"].get(f"{lbl}_n_pos", 0)

        def fmt(v):  return f"{v:.4f}" if not np.isnan(v) else "  N/A "
        def dfmt(v): return f"{v:+.4f}" if not np.isnan(v) else "  N/A "

        d_b_roc = b_roc - a_roc if not (np.isnan(b_roc) or np.isnan(a_roc)) else float("nan")
        d_c_roc = c_roc - a_roc if not (np.isnan(c_roc) or np.isnan(a_roc)) else float("nan")
        d_b_prc = b_prc - a_prc if not (np.isnan(b_prc) or np.isnan(a_prc)) else float("nan")
        d_c_prc = c_prc - a_prc if not (np.isnan(c_prc) or np.isnan(a_prc)) else float("nan")

        print(f"  {lbl:<25} "
              f"{fmt(a_roc):>8} {fmt(b_roc):>8} {dfmt(d_b_roc):>7} "
              f"{fmt(c_roc):>8} {dfmt(d_c_roc):>7} | "
              f"{fmt(a_prc):>8} {fmt(b_prc):>8} {dfmt(d_b_prc):>7} "
              f"{fmt(c_prc):>8} {dfmt(d_c_prc):>7} | {n_pos}")

# ── PLOTTING ───────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

plottable = []
for lbl in LABEL_COLS:
    has_data = any(
        not (np.isnan(results[s]["A"].get(f"{lbl}_auroc", float("nan")))
             and np.isnan(results[s]["B"].get(f"{lbl}_auroc", float("nan")))
             and np.isnan(results[s]["C"].get(f"{lbl}_auroc", float("nan"))))
        for s in results)
    if has_data: plottable.append(lbl)

if plottable:
    n_lbl = len(plottable)
    fig, axes = plt.subplots(2, n_lbl, figsize=(5 * n_lbl, 10))
    if n_lbl == 1: axes = axes.reshape(2, 1)
    split_names = list(results.keys())

    run_styles = {
        "A": dict(marker="o", color="#d62728", label="Run A (static)"),
        "B": dict(marker="s", color="#2ca02c", label="Run B (partial adapt)"),
        "C": dict(marker="^", color="#1f77b4", label="Run C (full adapt)"),
    }

    for j, lbl in enumerate(plottable):
        for row, metric in enumerate(["auroc", "auprc"]):
            ax = axes[row, j]

            for run_key, style in run_styles.items():
                vals, valid_x = [], []
                for k, s in enumerate(split_names):
                    v = results[s][run_key].get(f"{lbl}_{metric}", float("nan"))
                    if not np.isnan(v):
                        vals.append(v); valid_x.append(k)
                if len(valid_x) >= 2:
                    ax.plot(valid_x, vals, marker=style["marker"],
                            color=style["color"], label=style["label"],
                            lw=2, ms=8, linestyle="-")

            if len(split_names) >= 3:
                ax.axvline(x=len(split_names) - 1.5, color="gray",
                           linestyle="--", alpha=0.5, label="Drift")

            ax.set_xticks(range(len(split_names)))
            ax.set_xticklabels([s.split("(")[0].strip() for s in split_names],
                               fontsize=8, rotation=15)
            ax.set_ylabel(metric.upper())
            if row == 0:
                ax.set_title(lbl.replace("label_","").replace("_"," ").title(), fontsize=11)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            if metric == "auroc": ax.set_ylim(0.45, 1.0)

    fig.suptitle(
        "Run A (Static) vs Run B (Partial Adapt) vs Run C (Full Adapt)\n"
        "Pre-drift: identical source weights | Post-drift: adaptation diverges\n"
        f"(Source: median of {len(SOURCE_SEEDS)}-seed ensemble, "
        f"seed={_median_src_seed})",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "run_a_b_c_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved → {SAVE_PATH / 'run_a_b_c_comparison.png'}")

# Drift feature distributions
all_drift_feats = list(train_treat_ref.keys()) + [
    c for c in BINARY_MONITOR if c.startswith(("has_","high_","on_","early_"))]
plot_feats = all_drift_feats[:6]
n_plots = min(6, len(plot_feats))
if n_plots > 0:
    rows = (n_plots + 2) // 3
    fig2, axes2 = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes2 = axes2.flatten() if hasattr(axes2, "flatten") else [axes2]
    for i in range(len(axes2)):
        if i < n_plots:
            feat = plot_feats[i]; ax = axes2[i]
            ref    = (train_df.filter(pl.col("hrs_from_admit")==0)[feat].to_numpy().astype(float)
                      if feat in train_df.columns else np.array([]))
            pre_v  = (test_pre.filter(pl.col("hrs_from_admit")==0)[feat].to_numpy().astype(float)
                      if feat in test_pre.columns and test_pre.height > 0 else np.array([]))
            post_v = (test_post.filter(pl.col("hrs_from_admit")==0)[feat].to_numpy().astype(float)
                      if feat in test_post.columns and test_post.height > 0 else np.array([]))
            if len(ref)    > 0: ax.hist(ref,    bins=20, alpha=0.4, density=True, label="Train",      color="blue")
            if len(pre_v)  > 0: ax.hist(pre_v,  bins=20, alpha=0.4, density=True, label="Pre-drift",  color="green")
            if len(post_v) > 0: ax.hist(post_v, bins=20, alpha=0.4, density=True, label="Post-drift", color="red")
            ax.set_title(feat.replace("_"," ").title(), fontsize=9)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        else:
            axes2[i].set_visible(False)
    fig2.suptitle("Treatment Feature Distributions — Drift Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "drift_distributions.png", dpi=150, bbox_inches="tight")
    print(f"Saved → {SAVE_PATH / 'drift_distributions.png'}")

# ── SAVE MODELS ───────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SAVING TWO-STREAM MODELS — Run A + Run B (Run C in temp file)")
print("="*70)

model_A.eval()
if adaptation_performed:
    model_B.eval()
    model_C.eval()

torch.save({
    "source": source_state,
    "run_a": model_A.state_dict(),
    "run_b": model_B.state_dict() if adaptation_performed else source_state,
    "seq_dim": seq_dim,
    "treat_dim": treat_dim,
    "n_targets": len(LABEL_COLS),
    "train_stats": train_stats,
    "config": {
        "adaptation_performed": adaptation_performed,
        "source_seeds":         SOURCE_SEEDS,
        "median_source_seed":   _median_src_seed,
        "median_src_idx":       _median_src_idx,
        "source_ensemble_aurocs": _src_mean_aurocs.tolist(),
        "median_seed_B": SEEDS[median_idx_B] if adaptation_performed else None,
        "median_seed_C": SEEDS[median_idx_C] if adaptation_performed else None,
    }
}, SAVE_PATH / "two_stream_models.pt")

if adaptation_performed:
    torch.save(model_C.state_dict(), SAVE_PATH / "temp_run_c_weights.pt")
    print(f"✅ Saved temp_run_c_weights.pt (median-seed Run C)")
else:
    torch.save(source_state, SAVE_PATH / "temp_run_c_weights.pt")
    print(f"✅ Saved temp_run_c_weights.pt (source weights — no adaptation)")

print(f"✅ Saved two_stream_models.pt with Run A + median-seed Run B")

# CHG 12: majority_fired removed from eval_split.json
with open(SAVE_PATH / "eval_split.json", "w") as f:
    json.dump({
        "pre_cp_stays":             [int(x) for x in pre_cp_stays],
        "post_cp_stays":            [int(x) for x in post_cp_stays],
        "eval_post_stays":          [int(x) for x in eval_post_stays],
        "adapt_train_stays":        [int(x) for x in adapt_train_stays] if adaptation_performed else [],
        "adapt_val_stays":          [int(x) for x in adapt_val_stays]   if adaptation_performed else [],
        "buf_pre_stays":            [int(x) for x in buf_pre_stays]     if adaptation_performed else [],
        "adapt_train_subjects":     [int(x) for x in adapt_train_subjects] if adaptation_performed else [],
        "adapt_val_subjects":       [int(x) for x in adapt_val_subjects]   if adaptation_performed else [],
        "eval_post_subjects":       [int(x) for x in eval_post_subjects]   if adaptation_performed else [],
        "drift_tag":                str(drift_tag),
        "seeds":                    SEEDS if adaptation_performed else [SEED],
        "source_seeds":             SOURCE_SEEDS,
        "median_source_seed":       _median_src_seed,
        "source_ensemble_aurocs":   _src_mean_aurocs.tolist(),
        "median_seed_B":            SEEDS[median_idx_B] if adaptation_performed else None,
        "median_seed_C":            SEEDS[median_idx_C] if adaptation_performed else None,
    }, f)
print(f"✅ Saved eval_split.json")

if adaptation_performed and len(eval_post_stays) > 0:
    np.savez(SAVE_PATH / "post_drift_predictions.npz",
             labels=labels_a, probs_a=probs_a, probs_b=probs_b, probs_c=probs_c)
    print("✅ Saved post_drift_predictions.npz")
else:
    print("⚠ No post-drift predictions to save")

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
print(f"Architecture: LSTM({seq_dim}→{HIDDEN_DIM}) + MLP({treat_dim}→{TREAT_DIM}) → Fusion → {len(LABEL_COLS)}")
print(f"Source ensemble: {len(SOURCE_SEEDS)} seeds → median seed={_median_src_seed}  "
      f"(mAUROC {_src_mean_aurocs.mean():.4f} ± {_src_mean_aurocs.std(ddof=1):.4f})")
print(f"Drift: {len(drifted_post)} features | Adapted: {adaptation_performed}")

for s in results:
    a_aur = [results[s]["A"].get(f"{l}_auroc", float("nan")) for l in LABEL_COLS]
    b_aur = [results[s]["B"].get(f"{l}_auroc", float("nan")) for l in LABEL_COLS]
    c_aur = [results[s]["C"].get(f"{l}_auroc", float("nan")) for l in LABEL_COLS]
    d_b   = np.nanmean(b_aur) - np.nanmean(a_aur)
    d_c   = np.nanmean(c_aur) - np.nanmean(a_aur)
    print(f"  {s}: A={np.nanmean(a_aur):.4f} "
          f"B={np.nanmean(b_aur):.4f} (Δ={d_b:+.4f}) "
          f"C={np.nanmean(c_aur):.4f} (Δ={d_c:+.4f})")

if verdicts:
    print(f"\n  Post-drift verdicts (paired bootstrap, 95% CI):")
    b_sig = sum(1 for v in verdicts.values() if v == "B")
    c_sig = sum(1 for v in verdicts.values() if v == "C")
    ties  = sum(1 for v in verdicts.values() if v == "tie")
    print(f"    B significantly > C: {b_sig}/{len(verdicts)}")
    print(f"    C significantly > B: {c_sig}/{len(verdicts)}")
    print(f"    Statistical tie:     {ties}/{len(verdicts)}")
    for lbl, v in verdicts.items():
        print(f"      {lbl:<22} → {v}")
else:
    print("\n  No post-drift verdicts (no adaptation performed)")

# CHG 12: majority_fired removed from summary print
if pair_results:
    print("\n--- Sequential pairwise drift scores ---")
    for (prev_g, curr_g), res in pair_results.items():
        tag = "TRAIN" if curr_g in TRAIN_GROUP_SET else "TEST"
        dist_ok = res["dist_score"] > dist_threshold and res["n_drifted"] >= MIN_FEAT_DRIFTED
        perf_ok = res["n_perf_sig"] >= MIN_PERF_DROPS
        marker  = "⚠ DRIFT" if (dist_ok or perf_ok) else "✓"
        print(f"  [{tag}] {prev_g} → {curr_g}: "
              f"score={res['dist_score']:.4f}  n_drifted={res['n_drifted']}  "
              f"dist_ok={dist_ok}  perf_ok={perf_ok}  "
              f"(n_sig={res['n_perf_sig']})  "
              f"baseline={res['baseline_tag']}  {marker}")

print("\n✅ Complete")
if test_pre.height > 0:
    print(test_pre.height)
    print(test_pre["anchor_year_group"].value_counts())
    print(test_pre.filter(pl.col("hrs_from_admit") == 0).height)

if adaptation_performed and len(eval_post_stays) > 0:
    print("\n--- Paired bootstrap: ΔAUPRC (B − C) on held-out post-drift ---")
    rng3 = np.random.default_rng(3)
    for i, lbl in enumerate(LABEL_COLS):
        y_t = labels_a[:, i]
        if y_t.sum() == 0 or y_t.sum() == n: continue
        diffs = []
        for _ in range(n_boot):
            idx = rng3.integers(0, n, n)
            yt = y_t[idx]
            if yt.sum() == 0 or yt.sum() == n: continue
            diffs.append(average_precision_score(yt, probs_b[idx, i]) -
                         average_precision_score(yt, probs_c[idx, i]))
        diffs = np.array(diffs)
        print(f"  {lbl:<22} ΔB−C = {diffs.mean():+.4f}  "
              f"[{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}]  "
              f"P(B>C)={(diffs>0).mean():.2f}")
else:
    print("\n--- Paired bootstrap: ΔAUPRC skipped (no adaptation performed) ---")

# ── EXPORT: IG attribution data for fig_biological_amnesia ───────────────────
print("\n" + "="*70)
print("COMPUTING IG ATTRIBUTION DATA FOR FIG_BIOLOGICAL_AMNESIA")
print("="*70)

if adaptation_performed and len(eval_post_stays) > 0:
    import pickle

    eval_post_df_fig = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))
    ds_fig = ICUDataset(eval_post_df_fig, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)

    # CHG 13: guard ICUDataset stay_ids access with a fallback
    def _get_stay_id(ds, idx):
        """Safely retrieve stay_id from a dataset by index."""
        if hasattr(ds, "stay_ids"):
            return ds.stay_ids[idx]
        # Fallback: pull from the first-hour rows of the underlying dataframe
        return (eval_post_df_fig
                .filter(pl.col("hrs_from_admit") == 0)
                ["stay_id"].to_list()[idx])

    # Find a patient where model_B is correct (true positive for vasopressor)
    LABEL_IDX = 0  # label_vasopressor
    fig_stay_id   = None
    fig_patient_i = None

    for ci in range(len(ds_fig)):
        seq, treat, lbl = ds_fig[ci]
        if lbl[LABEL_IDX].item() != 1:
            continue
        with torch.no_grad():
            prob = torch.sigmoid(model_B(
                seq.unsqueeze(0).to(device),
                treat.unsqueeze(0).to(device)
            ))[0, LABEL_IDX].item()
        if prob >= 0.5:
            fig_patient_i = ci
            fig_stay_id   = _get_stay_id(ds_fig, ci)
            break

    if fig_patient_i is None:
        # Fallback: pick patient with highest predicted probability
        best_prob, best_ci = -1, 0
        for ci in range(min(len(ds_fig), 200)):
            seq, treat, lbl = ds_fig[ci]
            with torch.no_grad():
                prob = torch.sigmoid(model_B(
                    seq.unsqueeze(0).to(device),
                    treat.unsqueeze(0).to(device)
                ))[0, LABEL_IDX].item()
            if prob > best_prob:
                best_prob, best_ci = prob, ci
        fig_patient_i = best_ci
        fig_stay_id   = _get_stay_id(ds_fig, best_ci)

    print(f"  Selected patient stay_id={fig_stay_id} (index {fig_patient_i})")

    seq_t, treat_t, lbl_t = ds_fig[fig_patient_i]
    true_label = int(lbl_t[LABEL_IDX].item())

    # CHG 10: reuse src_model_shared (already reconstructed in Phase 2b).
    # src_model_shared was set to None when adaptation was not performed,
    # but we only reach here when adaptation_performed is True, so it is valid.
    src_model = src_model_shared

    def integrated_gradients_fig(model, xs, xt, ti, steps=30):
        # train() required for cuDNN RNN backward on GPU;
        # dropout zeroed to keep attributions deterministic (matching CPU behaviour)
        model.train()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
    
        xs, xt = xs.to(device), xt.to(device)
        bs, bt = torch.zeros_like(xs), torch.zeros_like(xt)
        sg, tg = torch.zeros_like(xs), torch.zeros_like(xt)
        for alpha in np.linspace(0, 1, steps):
            is_ = (bs + alpha * (xs - bs)).requires_grad_(True)
            it_ = (bt + alpha * (xt - bt)).requires_grad_(True)
            out = model(is_, it_)[:, ti].sum()
            g1, g2 = torch.autograd.grad(out, [is_, it_])
            sg += g1; tg += g2
        seq_attr   = ((xs - bs) * sg / steps).cpu().numpy()[0]  # (seq_len, seq_dim)
        treat_attr = ((xt - bt) * tg / steps).cpu().numpy()[0]  # (treat_dim,)
    
        # Restore original dropout and eval mode
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = DROPOUT
        model.eval()
    
        return seq_attr.sum(axis=0), treat_attr  # collapse time → (seq_dim,), (treat_dim,)

    xs_in = seq_t.unsqueeze(0)
    xt_in = treat_t.unsqueeze(0)

    # Source weights attributions
    physio_src_arr, treat_src_arr = integrated_gradients_fig(
        src_model, xs_in, xt_in, LABEL_IDX)
    # Adapted (Run B) attributions
    physio_adp_arr, treat_adp_arr = integrated_gradients_fig(
        model_B, xs_in, xt_in, LABEL_IDX)

    # Build per-feature dicts indexed by feature name
    physio_src_dict = {n: float(v) for n, v in zip(ds_fig.seq_cols,   physio_src_arr)}
    physio_adp_dict = {n: float(v) for n, v in zip(ds_fig.seq_cols,   physio_adp_arr)}
    treat_src_dict  = {n: float(v) for n, v in zip(ds_fig.treat_cols, treat_src_arr)}
    treat_adp_dict  = {n: float(v) for n, v in zip(ds_fig.treat_cols, treat_adp_arr)}

    # ── all_phys_deltas / all_treat_deltas ────────────────────────────────
    physio_src_vals = np.array(list(physio_src_dict.values()))
    treat_src_vals  = np.array(list(treat_src_dict.values()))
    runb_p95 = float(np.percentile(np.abs(
        np.concatenate([physio_src_vals, treat_src_vals])), 95))
    runb_p95 = max(runb_p95, 1e-6)

    all_phys_deltas = {}
    for n in ds_fig.seq_cols:
        src_v = physio_src_dict[n]
        adp_v = physio_adp_dict[n]
        d     = abs(adp_v - src_v) / runb_p95
        all_phys_deltas[n] = {"src": src_v, "adp": adp_v, "delta": d}

    all_treat_deltas = {}
    for n in ds_fig.treat_cols:
        src_v = treat_src_dict[n]
        adp_v = treat_adp_dict[n]
        d     = abs(adp_v - src_v) / runb_p95
        all_treat_deltas[n] = {"src": src_v, "adp": adp_v, "delta": d}

    # ── Predicted probabilities for the fig panels ────────────────────────
    with torch.no_grad():
        p_b_src = torch.sigmoid(src_model(
            xs_in.to(device), xt_in.to(device)))[0, LABEL_IDX].item()
        p_b_adp = torch.sigmoid(model_B(
            xs_in.to(device), xt_in.to(device)))[0, LABEL_IDX].item()

    # ── Normalization arrays for bottom panel ─────────────────────────────
    runb_phys_norm  = np.array([v["delta"] for v in all_phys_deltas.values()])
    runb_treat_norm = np.array([v["delta"] for v in all_treat_deltas.values()])

    # ── Patient metadata ──────────────────────────────────────────────────
    pat_row = test_raw.filter(pl.col("stay_id") == fig_stay_id) \
                      .filter(pl.col("hrs_from_admit") == 0)
    age     = float(pat_row["age"][0]) if "age" in pat_row.columns else float("nan")
    stay_id = int(fig_stay_id)

    # ── Save IG side only — script5 bridge merges XGBoost/SHAP side ──────
    ig_data = {
        "all_phys_deltas":  all_phys_deltas,
        "all_treat_deltas": all_treat_deltas,
        "p_b_src":          p_b_src,
        "p_b_adp":          p_b_adp,
        "runb_phys_norm":   runb_phys_norm,
        "runb_treat_norm":  runb_treat_norm,
        "runb_p95":         runb_p95,
        "stay_id":          stay_id,
        "age":              age,
        "true_label":       true_label,
        "seq_cols":         ds_fig.seq_cols,
        "treat_cols":       ds_fig.treat_cols,
        "label_idx":        LABEL_IDX,
    }
    with open(SAVE_PATH / "fig_amnesia_ig_data.pkl", "wb") as f:
        pickle.dump(ig_data, f)
    print(f"✅ Saved fig_amnesia_ig_data.pkl → script5 bridge will merge XGBoost/SHAP side")

else:
    print("⚠ Skipping fig attribution data — no adaptation was performed")

# ── FINAL AUDIT LOG ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL MODEL UPDATE AUDIT LOG")
print("="*70)

if 'explainer' in dir():
    human_text = explainer.generate_human_summary()
    print(human_text)
    with open(SAVE_PATH / "clinical_drift_audit.txt", "w") as f:
        f.write("FINAL MODEL UPDATE AUDIT LOG\n")
        f.write("="*70 + "\n")
        f.write(human_text)
    print(f"\n✅ Full text audit saved: {SAVE_PATH / 'clinical_drift_audit.txt'}")