#%%writefile evaluation/script6_disagreement_matrix.py
"""
script5_disagreement_matrix_with_runD.py
══════════════════════════════════════════
MC3 + MC2 Reviewer Response — Full disagreement matrix including Run D.

Extends script5_disagreement_matrix.py to include Run D (single-stream
last-layer adapt) in all pairwise comparisons.

New comparisons added:
  (iv)  Run B  vs  Run D  →  two-stream decomposition vs monolithic selective freeze
  (v)   Run C  vs  Run D  →  two-stream full adapt vs monolithic selective freeze
  (vi)  Run D  vs  XGBoost-adapted  →  monolithic last-layer vs monolithic full retrain

Run AFTER:
  - script2_three_stream_model_v4.py   (produces two_stream_models.pt)
  - script_run_d_single_stream.py      (produces run_d_model.pt)
"""

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, SingleStreamDataset, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import PhysiologyStream, TreatmentStream, FusionHead, SingleStreamModel, TwoStreamModel, SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM, BATCH_SIZE, LSTM_LAYERS, DROPOUT


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ========================== IMPORT UTILS & MODELS ==========================


torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH = Path("/kaggle/working")
BASE_PATH2 = Path("/kaggle/input/datasets/fatematamanna/ptfiles")

TAU_HIGH_PRIMARY = 0.50
TAU_LOW_PRIMARY = 0.10
THRESHOLD_PAIRS = [(0.50, 0.10), (0.50, 0.20), (0.40, 0.10), 
                   (0.30, 0.10), (0.50, 0.05)]

# ========================== LOAD MODELS ==========================
print("\nLoading two-stream checkpoint (Runs A, B)...")
ckpt_ab = torch.load(BASE_PATH2 / "two_stream_models (3).pt", 
                     map_location=device, weights_only=False)

seq_dim = ckpt_ab["seq_dim"]
treat_dim = ckpt_ab["treat_dim"]
n_targets = ckpt_ab["n_targets"]
train_stats = ckpt_ab["train_stats"]

model_A = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_B = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_A.load_state_dict(ckpt_ab["run_a"]); model_A.eval()
model_B.load_state_dict(ckpt_ab["run_b"]); model_B.eval()

print("Loading full adapt checkpoint (Runs C, D)...")
ckpt_cd = torch.load(BASE_PATH2 / "full_adapt_models.pt", 
                     map_location=device, weights_only=False)

model_C = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_C.load_state_dict(ckpt_cd["run_c"]); model_C.eval()

combined_input_dim = ckpt_cd["combined_input_dim"]
model_D = SingleStreamModel(combined_input_dim, HIDDEN_DIM, LSTM_LAYERS, 
                           DROPOUT, n_targets).to(device)
model_D.load_state_dict(ckpt_cd["run_d_adapted"]); model_D.eval()

print("✅ All models (A, B, C, D) loaded successfully")

# ========================== DATA LOADING ==========================
print("\nLoading data...")
test_df = normalize(load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES), 
                    train_stats)

test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

print(f"Total post-drift stays: {test_post['stay_id'].n_unique()}")

# Try to load locked held-out set first (preferred)
json_path = BASE_PATH2 / "eval_post_stays.json"
if json_path.exists():
    with open(json_path, "r") as f:
        eval_post_stays = json.load(f)
    print(f"✅ Loaded locked held-out set from JSON: {len(eval_post_stays)} stays")
else:
    print("⚠️  eval_post_stays.json not found → Creating dynamic split (same as Run D script)")
    
    # === Dynamic splitting logic (same rules as script_run_d) ===
    post_stays = (test_post.filter(pl.col("hrs_from_admit") == 0)
                           .sort("intime")["stay_id"].to_list())
    
    n_total = len(post_stays)
    n_adapt_train = int(n_total * 0.30)
    n_adapt_val   = int(n_total * 0.10)
    n_heldout     = n_total - n_adapt_train - n_adapt_val
    
    eval_post_stays = post_stays[n_adapt_train + n_adapt_val:]
    
    print(f"Dynamic split → Held-out: {len(eval_post_stays)} stays "
          f"({n_adapt_train} adapt train + {n_adapt_val} adapt val)")
    
    # Optional: Save it so future runs are consistent
    with open(json_path, "w") as f:
        json.dump(eval_post_stays, f)
    print(f"   → Saved new eval_post_stays.json for reproducibility")

# Final filtering
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))
print(f"✅ Final post-drift held-out set: {len(eval_post_stays)} stays")
# ── GET PREDICTIONS — TWO-STREAM MODELS ───────────────────────────────────────
@torch.no_grad()
def get_two_stream_preds(model, df):
    ds     = ICUDataset(df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_logits, all_labels = [], []
    for x_seq, x_treat, y in loader:
        all_logits.append(
            model(x_seq.to(device), x_treat.to(device)).cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    stay_ids = ds.stay_ids
    return probs, labels, stay_ids

# ── GET PREDICTIONS — SINGLE-STREAM MODEL (Run D) ─────────────────────────────
@torch.no_grad()
def get_single_stream_preds(model, df):
    ds = SingleStreamDataset(
        df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_logits, all_labels = [], []
    for x_comb, y in loader:
        all_logits.append(model(x_comb.to(device)).cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1.0 / (1.0 + np.exp(-logits))
    stay_ids = ds.stay_ids
    return probs, labels, stay_ids

print("\nGetting predictions on post-drift held-out set...")
probs_a, labels, stay_ids = get_two_stream_preds(model_A, eval_post_df)
probs_b, _,      sids_b   = get_two_stream_preds(model_B, eval_post_df)
probs_c, _,      sids_c   = get_two_stream_preds(model_C, eval_post_df)
probs_d, _,      sids_d   = get_single_stream_preds(model_D, eval_post_df)

# Verify all models evaluated on the same stay ordering
assert sids_b == stay_ids, "Run B stay-id ordering mismatch"
assert sids_c == stay_ids, "Run C stay-id ordering mismatch"
assert sids_d == stay_ids, "Run D stay-id ordering mismatch — check SingleStreamDataset sort"
print(f"  Stay-id alignment confirmed for all four models: {len(stay_ids)} stays")

# ── LOAD XGBOOST PREDICTIONS ──────────────────────────────────────────────────
xgb_data = np.load(SAVE_PATH / "xgb_predictions_corrected.npz")
probs_xgb_source_raw  = xgb_data["probs_xgb_source"]
probs_xgb_adapted_raw = xgb_data["probs_xgb_adapted"]
xgb_sids_raw = xgb_data["stay_ids"].tolist()

# Verify same set of patients
assert set(xgb_sids_raw) == set(stay_ids), \
    "Patient sets do not match between PyTorch and XGBoost"

# Realign XGBoost predictions to match PyTorch stay_id ordering
xgb_index = {sid: i for i, sid in enumerate(xgb_sids_raw)}
reorder    = [xgb_index[sid] for sid in stay_ids]

probs_xgb_source  = probs_xgb_source_raw[reorder]
probs_xgb_adapted = probs_xgb_adapted_raw[reorder]

# Verify ordering now matches
assert [xgb_sids_raw[i] for i in reorder] == stay_ids, \
    "Realignment failed"

print(f"✅ XGBoost realigned to PyTorch ordering ({len(stay_ids)} stays)")

# ── COMPLETE ORDERING VERIFICATION ────────────────────────────────────────────
print("Verifying realignment...")

# 1. Check every single position matches
all_match = all(
    xgb_sids_raw[xgb_index[stay_ids[i]]] == stay_ids[i] 
    for i in range(len(stay_ids))
)
print(f"  All {len(stay_ids)} positions correctly aligned: {all_match}")

# 2. Check first 5 and last 5 explicitly
print("\n  First 5 patients:")
for i in range(5):
    sid = stay_ids[i]
    original_xgb_pos = xgb_index[sid]
    print(f"    pos {i}: PyTorch={sid} | XGB_realigned={xgb_sids_raw[original_xgb_pos]} | match={sid == xgb_sids_raw[original_xgb_pos]}")

print("\n  Last 5 patients:")
for i in range(len(stay_ids)-5, len(stay_ids)):
    sid = stay_ids[i]
    original_xgb_pos = xgb_index[sid]
    print(f"    pos {i}: PyTorch={sid} | XGB_realigned={xgb_sids_raw[original_xgb_pos]} | match={sid == xgb_sids_raw[original_xgb_pos]}")

# 3. Verify reordered array matches PyTorch ordering exactly
reordered_sids = [xgb_sids_raw[i] for i in reorder]
assert reordered_sids == stay_ids, "CRITICAL: Realignment failed"
print("\n✅ VERIFIED: XGBoost and PyTorch arrays point to identical patients at every position")

# ── DISAGREEMENT MATRIX FUNCTIONS ─────────────────────────────────────────────
def disagreement_matrix(probs_X, probs_Y, y_true,
                        tau_high=0.5, tau_low=0.1):
    pos_mask = (y_true == 1)
    n_pos    = int(pos_mask.sum())
    if n_pos == 0:
        return {"n_pos": 0, "x_catch_y_miss": 0, "y_catch_x_miss": 0,
                "both_catch": 0, "both_miss": 0,
                "asymmetry": float("nan"),
                "tau_high": tau_high, "tau_low": tau_low}
    pX, pY          = probs_X[pos_mask], probs_Y[pos_mask]
    x_catch_y_miss  = int(np.sum((pX >= tau_high) & (pY <  tau_low)))
    y_catch_x_miss  = int(np.sum((pY >= tau_high) & (pX <  tau_low)))
    both_catch      = int(np.sum((pX >= tau_high) & (pY >= tau_high)))
    both_miss       = int(np.sum((pX <  tau_high) & (pY <  tau_high)))
    asym = (x_catch_y_miss / max(y_catch_x_miss, 1)
            if y_catch_x_miss > 0 else float("inf"))
    return {"n_pos": n_pos,
            "x_catch_y_miss": x_catch_y_miss,
            "y_catch_x_miss": y_catch_x_miss,
            "both_catch":     both_catch,
            "both_miss":      both_miss,
            "asymmetry":      asym,
            "tau_high":       tau_high,
            "tau_low":        tau_low}

def false_alarm_matrix(probs_X, probs_Y, y_true,
                       tau_high=0.5, tau_low=0.1):
    neg_mask        = (y_true == 0)
    n_neg           = int(neg_mask.sum())
    if n_neg == 0:
        return {"n_neg": 0, "x_alarm_y_quiet": 0, "y_alarm_x_quiet": 0,
                "tau_high": tau_high, "tau_low": tau_low}
    pX, pY          = probs_X[neg_mask], probs_Y[neg_mask]
    x_alarm_y_quiet = int(np.sum((pX >= tau_high) & (pY < tau_low)))
    y_alarm_x_quiet = int(np.sum((pY >= tau_high) & (pX < tau_low)))
    return {"n_neg":            n_neg,
            "x_alarm_y_quiet":  x_alarm_y_quiet,
            "y_alarm_x_quiet":  y_alarm_x_quiet,
            "tau_high":         tau_high,
            "tau_low":          tau_low}

# ── DEFINE ALL PAIRWISE COMPARISONS (now includes Run D) ──────────────────────
COMPARISONS = [
    # Original MC3 comparisons
    ("RunB_vs_XGBadapted", "Run B",  probs_b, "XGBoost-adapted", probs_xgb_adapted),
    ("RunB_vs_RunC",       "Run B",  probs_b, "Run C",           probs_c),
    ("RunC_vs_XGBadapted", "Run C",  probs_c, "XGBoost-adapted", probs_xgb_adapted),
    ("RunB_vs_RunA",       "Run B",  probs_b, "Run A",           probs_a),
    # New MC2 comparisons involving Run D
    ("RunB_vs_RunD",       "Run B",  probs_b, "Run D",           probs_d),
    ("RunC_vs_RunD",       "Run C",  probs_c, "Run D",           probs_d),
    ("RunD_vs_XGBadapted", "Run D",  probs_d, "XGBoost-adapted", probs_xgb_adapted),
    ("RunD_vs_RunA",       "Run D",  probs_d, "Run A",           probs_a),
]

# ── RUN ALL COMPARISONS ────────────────────────────────────────────────────────
print("\n" + "="*78)
print(f"DISAGREEMENT MATRICES  "
      f"(catch>={TAU_HIGH_PRIMARY}, miss<{TAU_LOW_PRIMARY})")
print("="*78)

all_results = {}

for cmp_key, name_X, p_X, name_Y, p_Y in COMPARISONS:
    print(f"\n── {name_X}  vs  {name_Y}  " + "─"*40)
    cmp_results = {
        "comparison": cmp_key,
        "model_X": name_X, "model_Y": name_Y, "by_label": {}}
    for i, lbl in enumerate(LABEL_COLS):
        y_i = labels[:, i]
        dm  = disagreement_matrix(p_X[:, i], p_Y[:, i], y_i,
                                  TAU_HIGH_PRIMARY, TAU_LOW_PRIMARY)
        fa  = false_alarm_matrix( p_X[:, i], p_Y[:, i], y_i,
                                  TAU_HIGH_PRIMARY, TAU_LOW_PRIMARY)
        cmp_results["by_label"][lbl] = {"positives": dm, "negatives": fa}
        lbl_short = lbl.replace("label_", "")
        n_pos     = dm["n_pos"]
        xY, yX    = dm["x_catch_y_miss"], dm["y_catch_x_miss"]
        bc, bm    = dm["both_catch"],     dm["both_miss"]
        asym_str  = (f"{dm['asymmetry']:.2f}x"
                     if np.isfinite(dm["asymmetry"]) else "∞")
        print(f"\n  {lbl_short.upper():<16} (n_pos={n_pos})")
        print(f"    {name_X} catches, {name_Y} misses : {xY:>4}  "
              f"({100*xY/max(n_pos,1):>5.1f}%)")
        print(f"    {name_Y} catches, {name_X} misses : {yX:>4}  "
              f"({100*yX/max(n_pos,1):>5.1f}%)")
        print(f"    Both catch                        : {bc:>4}  "
              f"({100*bc/max(n_pos,1):>5.1f}%)")
        print(f"    Both miss                         : {bm:>4}  "
              f"({100*bm/max(n_pos,1):>5.1f}%)")
        print(f"    Asymmetry ratio ({name_X}/{name_Y})  : {asym_str}")
        print(f"    False-alarm (n_neg={fa['n_neg']}):  "
              f"{name_X} alarms/{name_Y} quiet={fa['x_alarm_y_quiet']}  |  "
              f"{name_Y} alarms/{name_X} quiet={fa['y_alarm_x_quiet']}")
    all_results[cmp_key] = cmp_results

# ── THRESHOLD SENSITIVITY — primary and Run D comparisons ─────────────────────
print("\n" + "="*78)
print("THRESHOLD SENSITIVITY")
print("="*78)

sensitivity_keys = [
    ("RunB_vs_XGBadapted", "Run B",  probs_b, "XGBoost-adapted", probs_xgb_adapted),
    ("RunD_vs_XGBadapted", "Run D",  probs_d, "XGBoost-adapted", probs_xgb_adapted),
    ("RunB_vs_RunD",       "Run B",  probs_b, "Run D",           probs_d),
]

sensitivity = {}
for cmp_key, name_X, p_X, name_Y, p_Y in sensitivity_keys:
    print(f"\n  ── {name_X} vs {name_Y}")
    sensitivity[cmp_key] = {}
    for tau_h, tau_l in THRESHOLD_PAIRS:
        print(f"    catch>={tau_h:.2f}  miss<{tau_l:.2f}")
        sensitivity[cmp_key][f"tau_h={tau_h}_tau_l={tau_l}"] = {}
        for i, lbl in enumerate(LABEL_COLS):
            dm = disagreement_matrix(p_X[:, i], p_Y[:, i], labels[:, i],
                                     tau_h, tau_l)
            sensitivity[cmp_key][f"tau_h={tau_h}_tau_l={tau_l}"][lbl] = dm
            lbl_short = lbl.replace("label_", "")
            asym_str  = (f"{dm['asymmetry']:.2f}x"
                         if np.isfinite(dm["asymmetry"]) else "∞")
            print(f"      {lbl_short:<16}  "
                  f"{name_X}-catch/{name_Y}-miss={dm['x_catch_y_miss']:>3}   "
                  f"{name_Y}-catch/{name_X}-miss={dm['y_catch_x_miss']:>3}   "
                  f"asymmetry={asym_str}")

# ── WORKED-EXAMPLE PATIENTS — all five model probabilities ────────────────────
print("\n" + "="*78)
print("WORKED-EXAMPLE PATIENTS — Run A / B / C / D / XGBoost probabilities")
print("="*78)

WORKED_EXAMPLES = [
    ("Patient A — vasopressor",  "label_vasopressor",  34731610),
    ("Patient B — septic shock", "label_septic_shock", 31210595),
    ("Patient C — intubation",   "label_intubation",   38854605),
]

stay_id_arr  = np.array(stay_ids)
worked_table = []
for desc, lbl, sid in WORKED_EXAMPLES:
    matches = np.where(stay_id_arr == sid)[0]
    if len(matches) == 0:
        print(f"  {desc} (stay_id={sid}): NOT in held-out set")
        worked_table.append({"desc": desc, "stay_id": sid, "found": False})
        continue
    idx = int(matches[0])
    li  = LABEL_COLS.index(lbl)
    row = {
        "desc": desc, "stay_id": sid, "label": lbl, "found": True,
        "true_label":      int(labels[idx, li]),
        "p_run_a":         float(probs_a[idx, li]),
        "p_run_b":         float(probs_b[idx, li]),
        "p_run_c":         float(probs_c[idx, li]),
        "p_run_d":         float(probs_d[idx, li]),
        "p_xgb_source":    float(probs_xgb_source[idx, li]),
        "p_xgb_adapted":   float(probs_xgb_adapted[idx, li]),
    }
    worked_table.append(row)
    print(f"\n  {desc} (stay_id={sid}, label={lbl}, true={row['true_label']})")
    print(f"    Run A  (frozen)              : {row['p_run_a']:.4f}")
    print(f"    Run B  (two-stream selective): {row['p_run_b']:.4f}")
    print(f"    Run C  (two-stream full)     : {row['p_run_c']:.4f}")
    print(f"    Run D  (single-stream head)  : {row['p_run_d']:.4f}")
    print(f"    XGBoost (source)             : {row['p_xgb_source']:.4f}")
    print(f"    XGBoost (adapted)            : {row['p_xgb_adapted']:.4f}")

# ── VISUALIZATION: 3×4 GRID (original 3 cols + Run D comparisons) ──────────────
print("\nGenerating disagreement-matrix figures...")

# Figure 1: original three comparisons (for paper Table 6)
fig1, axes1 = plt.subplots(3, 3, figsize=(15, 11))
key_comparisons_orig = [
    "RunB_vs_XGBadapted",
    "RunB_vs_RunC",
    "RunC_vs_XGBadapted",
]
for col, cmp_key in enumerate(key_comparisons_orig):
    cmp      = all_results[cmp_key]
    name_X   = cmp["model_X"]
    name_Y   = cmp["model_Y"]
    for row, lbl in enumerate(LABEL_COLS):
        ax  = axes1[row, col]
        dm  = cmp["by_label"][lbl]["positives"]
        bars  = [dm["x_catch_y_miss"], dm["y_catch_x_miss"],
                 dm["both_catch"],      dm["both_miss"]]
        names = [f"{name_X} catches\n{name_Y} misses",
                 f"{name_Y} catches\n{name_X} misses",
                 "Both catch", "Both miss"]
        colors = ["#16a34a","#dc2626","#2563eb","#9ca3af"]
        bars_obj = ax.bar(range(4), bars, color=colors,
                          alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("Count" if col == 0 else "")
        if row == 0:
            ax.set_title(f"{name_X} vs {name_Y}", fontsize=10, fontweight="bold")
        for b, v in zip(bars_obj, bars):
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + max(bars)*0.01,
                    str(v), ha="center", va="bottom", fontsize=8)
        lbl_short = lbl.replace("label_","").replace("_"," ").title()
        ax.text(0.02, 0.95, f"{lbl_short}\n(n_pos={dm['n_pos']})",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.grid(True, alpha=0.3, axis="y")

fig1.suptitle(
    f"Disagreement matrices — original three comparisons\n"
    f"(n={len(stay_ids)} post-drift stays, "
    f"catch≥{TAU_HIGH_PRIMARY}, miss<{TAU_LOW_PRIMARY})",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(SAVE_PATH / "disagreement_matrices_abc.png",
            dpi=150, bbox_inches="tight")
print(f"Saved → {SAVE_PATH / 'disagreement_matrices_abc.png'}")

# Figure 2: Run D comparisons (for MC2 architectural argument)
fig2, axes2 = plt.subplots(3, 3, figsize=(15, 11))
key_comparisons_runD = [
    "RunB_vs_RunD",
    "RunC_vs_RunD",
    "RunD_vs_XGBadapted",
]
for col, cmp_key in enumerate(key_comparisons_runD):
    cmp    = all_results[cmp_key]
    name_X = cmp["model_X"]
    name_Y = cmp["model_Y"]
    for row, lbl in enumerate(LABEL_COLS):
        ax  = axes2[row, col]
        dm  = cmp["by_label"][lbl]["positives"]
        bars  = [dm["x_catch_y_miss"], dm["y_catch_x_miss"],
                 dm["both_catch"],      dm["both_miss"]]
        names = [f"{name_X} catches\n{name_Y} misses",
                 f"{name_Y} catches\n{name_X} misses",
                 "Both catch", "Both miss"]
        colors = ["#16a34a","#dc2626","#2563eb","#9ca3af"]
        bars_obj = ax.bar(range(4), bars, color=colors,
                          alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("Count" if col == 0 else "")
        if row == 0:
            ax.set_title(f"{name_X} vs {name_Y}", fontsize=10, fontweight="bold")
        for b, v in zip(bars_obj, bars):
            ax.text(b.get_x() + b.get_width()/2,
                    b.get_height() + max(bars)*0.01,
                    str(v), ha="center", va="bottom", fontsize=8)
        lbl_short = lbl.replace("label_","").replace("_"," ").title()
        ax.text(0.02, 0.95, f"{lbl_short}\n(n_pos={dm['n_pos']})",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.grid(True, alpha=0.3, axis="y")

fig2.suptitle(
    f"Disagreement matrices — Run D (single-stream) comparisons\n"
    f"(n={len(stay_ids)} post-drift stays, "
    f"catch≥{TAU_HIGH_PRIMARY}, miss<{TAU_LOW_PRIMARY})",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(SAVE_PATH / "disagreement_matrices_runD.png",
            dpi=150, bbox_inches="tight")
print(f"Saved → {SAVE_PATH / 'disagreement_matrices_runD.png'}")

# ── SAVE ALL RESULTS ───────────────────────────────────────────────────────────
np.savez(
    SAVE_PATH / "post_drift_predictions_with_runD.npz",
    stay_ids         = np.array(stay_ids),
    labels           = labels,
    probs_run_a      = probs_a,
    probs_run_b      = probs_b,
    probs_run_c      = probs_c,
    probs_run_d      = probs_d,
    probs_xgb_source  = probs_xgb_source,
    probs_xgb_adapted = probs_xgb_adapted,
)
print(f"Saved prediction arrays → "
      f"{SAVE_PATH / 'post_drift_predictions_with_runD.npz'}")

def to_serialisable(obj):
    if isinstance(obj, dict):
        return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serialisable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj: return None          # NaN
        if obj == float("inf"): return "inf"
        return round(obj, 6)
    if isinstance(obj, (np.float32, np.float64)):
        if obj != obj: return None
        return float(round(obj, 6))
    if isinstance(obj, (np.int32, np.int64, int)):
        return int(obj)
    return obj

output = {
    "thresholds": {"primary_tau_high": TAU_HIGH_PRIMARY,
                   "primary_tau_low":  TAU_LOW_PRIMARY},
    "n_held_out_stays":  len(stay_ids),
    "comparisons":       all_results,
    "threshold_sensitivity": sensitivity,
    "worked_examples":   worked_table,
}
with open(SAVE_PATH / "disagreement_matrix_results_with_runD.json", "w") as f:
    json.dump(to_serialisable(output), f, indent=2)
print(f"Saved results → "
      f"{SAVE_PATH / 'disagreement_matrix_results_with_runD.json'}")

# ── PAPER-READY SUMMARY ────────────────────────────────────────────────────────
print("\n" + "="*78)
print("PAPER-READY SUMMARY")
print("="*78)

print(f"\nAcross the {len(stay_ids):,} held-out post-drift stays:\n")

for cmp_key, name_X, p_X, name_Y, p_Y in [
    ("RunB_vs_XGBadapted", "Run B", probs_b, "XGBoost-adapted", probs_xgb_adapted),
    ("RunB_vs_RunD",       "Run B", probs_b, "Run D",           probs_d),
    ("RunC_vs_RunD",       "Run C", probs_c, "Run D",           probs_d),
]:
    print(f"  ── {name_X} vs {name_Y}")
    for i, lbl in enumerate(LABEL_COLS):
        dm = all_results[cmp_key]["by_label"][lbl]["positives"]
        fa = all_results[cmp_key]["by_label"][lbl]["negatives"]
        lbl_short = lbl.replace("label_","").replace("_"," ")
        asym_str  = (f"{dm['asymmetry']:.1f}x"
                     if np.isfinite(dm["asymmetry"]) else "∞")
        print(f"    {lbl_short:<18}  "
              f"{name_X}-catches/{name_Y}-misses = {dm['x_catch_y_miss']:>3}  "
              f"{name_Y}-catches/{name_X}-misses = {dm['y_catch_x_miss']:>3}  "
              f"asym={asym_str}  |  "
              f"FA: {name_X}={fa['x_alarm_y_quiet']} {name_Y}={fa['y_alarm_x_quiet']}")
    print()

print("\n✅ Complete — two figures and one JSON saved to", SAVE_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# RUN B VS ALL 5 OTHER MODELS (MASTER SUMMARY)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*95)
print("RUN B vs ALL 5 OTHER MODELS (Disagreement Summary)")
print(f"Thresholds: Catch >= {TAU_HIGH_PRIMARY}, Miss < {TAU_LOW_PRIMARY}")
print("="*95)

# List of all models to compare against Run B
other_models = [
    ("Run A (Static)", probs_a),
    ("Run C (Full Two-Stream)", probs_c),
    ("Run D (Single-Stream)", probs_d),
    ("XGBoost (Source)", probs_xgb_source),
    ("XGBoost (Adapted)", probs_xgb_adapted)
]

for i, lbl in enumerate(LABEL_COLS):
    lbl_short = lbl.replace("label_", "").upper()
    print(f"\n── LABEL: {lbl_short} " + "─"*70)
    print(f"  {'Model compared to Run B':<25} | {'Run B Catches / Other Misses':>28} | {'Other Catches / Run B Misses':>28} | {'Asymmetry':>9}")
    print("  " + "-"*96)
    
    for name_other, probs_other in other_models:
        dm = disagreement_matrix(
            probs_b[:, i], probs_other[:, i], labels[:, i],
            tau_high=TAU_HIGH_PRIMARY, tau_low=TAU_LOW_PRIMARY
        )
        
        x_y = dm['x_catch_y_miss']
        y_x = dm['y_catch_x_miss']
        asym = dm['asymmetry']
        asym_str = f"{asym:.2f}x" if np.isfinite(asym) else "∞"
        
        print(f"  {name_other:<25} | {x_y:>28} | {y_x:>28} | {asym_str:>9}")