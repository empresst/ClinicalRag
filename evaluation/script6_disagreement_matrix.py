%%writefile evaluation/script6_disagreement_matrix.py
"""
script6_disagreement_matrix.py
══════════════════════════════
MC3 + MC2 Reviewer Response — Full disagreement matrix including Run D.
Run AFTER script2 (v5), script3, script5 (corrected).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, SingleStreamDataset, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import TwoStreamModel, SingleStreamModel

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# FIX 1 + FIX 2: Define all constants locally — do NOT import from architectures.
# These must exactly match script2 and script3 values.
SEED         = 42
SEQ_LEN      = 6
HIDDEN_DIM   = 64
TREAT_DIM    = 32
LSTM_LAYERS  = 2
BATCH_SIZE   = 64
DROPOUT      = 0.3
LABEL_COLS   = ["label_vasopressor", "label_intubation", "label_septic_shock"]

torch.manual_seed(SEED)
np.random.seed(SEED)

BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH  = Path("/kaggle/working")
BASE_PATH2 = Path("/kaggle/input/datasets/fatematamanna/ptfiles")

TAU_HIGH_PRIMARY = 0.50
TAU_LOW_PRIMARY  = 0.10
THRESHOLD_PAIRS  = [(0.50, 0.10), (0.50, 0.20), (0.40, 0.10),
                    (0.30, 0.10), (0.50, 0.05)]

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("\nLoading two-stream checkpoint (Runs A, B)...")
ckpt_ab = torch.load(SAVE_PATH / "two_stream_models.pt",
                     map_location=device, weights_only=False)

seq_dim     = ckpt_ab["seq_dim"]
treat_dim   = ckpt_ab["treat_dim"]
n_targets   = ckpt_ab["n_targets"]
train_stats = ckpt_ab["train_stats"]


_all_trained_features = list(ckpt_ab["train_stats"].keys())


SEQ_FEATURES_CKPT   = [f for f in SEQ_FEATURES   if f in _all_trained_features]
TREAT_FEATURES_CKPT = [f for f in TREATMENT_FEATURES if f in _all_trained_features]

# Verify dims match the saved model
assert len(TREAT_FEATURES_CKPT) == treat_dim, (
    f"treat_dim mismatch: checkpoint has {treat_dim} but reconstructed "
    f"TREAT_FEATURES_CKPT has {len(TREAT_FEATURES_CKPT)}.\n"
    f"Check that TREATMENT_FEATURES in constants.py matches what was used during training.")

print(f"  Reconstructed: {len(SEQ_FEATURES_CKPT)} seq features, "
      f"{len(TREAT_FEATURES_CKPT)} treat features")


model_A = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_B = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_A.load_state_dict(ckpt_ab["run_a"]); model_A.eval()
model_B.load_state_dict(ckpt_ab["run_b"]); model_B.eval()

print("Loading full adapt checkpoint (Runs C, D)...")
ckpt_cd = torch.load(SAVE_PATH / "full_adapt_models.pt",
                     map_location=device, weights_only=False)

seq_dim_cd   = ckpt_cd.get("seq_dim",   seq_dim)
treat_dim_cd = ckpt_cd.get("treat_dim", treat_dim)

model_C = TwoStreamModel(seq_dim_cd, treat_dim_cd, n_targets).to(device)
model_C.load_state_dict(ckpt_cd["run_c"]); model_C.eval()

combined_input_dim = ckpt_cd["combined_input_dim"]
model_D = SingleStreamModel(combined_input_dim, HIDDEN_DIM, LSTM_LAYERS,
                             DROPOUT, n_targets).to(device)
model_D.load_state_dict(ckpt_cd["run_d_adapted"]); model_D.eval()

if seq_dim_cd != seq_dim or treat_dim_cd != treat_dim:
    print(f"  ⚠ Dim mismatch between checkpoints: "
          f"ckpt_ab=({seq_dim},{treat_dim}) vs ckpt_cd=({seq_dim_cd},{treat_dim_cd})")
    print(f"  ⚠ Runs A/B and C/D were trained with different feature sets — "
          f"comparison is only valid if the eval data matches ckpt_cd dims.")
    # Use ckpt_cd dims as authoritative for eval (C and D are the adapted models)
    seq_dim   = seq_dim_cd
    treat_dim = treat_dim_cd

print("✅ All models (A, B, C, D) loaded successfully")

# ── DATA LOADING ──────────────────────────────────────────────────────────────
print("\nLoading data...")
test_df = normalize(
    load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES),
    train_stats)

test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")
print(f"Total post-drift stays: {test_post['stay_id'].n_unique()}")

# ── LOAD EVAL SPLIT ───────────────────────────────────────────────────────────
# FIX 3 + FIX 7: Always load from eval_split.json saved by script2 (subject-level
# split). The old dynamic fallback used a stay-level intime sort which diverges
# from script2 v5's subject-level logic, causing label misalignment.
# Check both possible locations: working dir first, then ptfiles input.
json_candidates = [
    SAVE_PATH  / "eval_split.json",
    BASE_PATH2 / "eval_split.json",
]
split_path = next((p for p in json_candidates if p.exists()), None)

if split_path is None:
    raise FileNotFoundError(
        "eval_split.json not found in SAVE_PATH or BASE_PATH2.\n"
        "Run script2 (v5) first — it saves eval_split.json with subject-level splits.")

with open(split_path) as f:
    split = json.load(f)

eval_post_stays = list(map(int, split["eval_post_stays"]))
drift_tag       = split.get("drift_tag", "unknown")

print(f"✅ Loaded eval_split.json from {split_path} (drift_tag={drift_tag})")
print(f"   eval_post_stays: {len(eval_post_stays)} stays")

eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))
print(f"✅ Final post-drift held-out set: {eval_post_df['stay_id'].n_unique()} stays")

# ── PREDICTION HELPERS ─────────────────────────────────────────────────────────
@torch.no_grad()
def get_two_stream_preds(model, df):
    ds     = ICUDataset(df, SEQ_FEATURES_CKPT, TREAT_FEATURES_CKPT, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_logits, all_labels = [], []
    for x_seq, x_treat, y in loader:
        all_logits.append(model(x_seq.to(device), x_treat.to(device)).cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return 1.0 / (1.0 + np.exp(-logits)), labels, ds.stay_ids

@torch.no_grad()
def get_single_stream_preds(model, df):
    ds     = SingleStreamDataset(df, SEQ_FEATURES_CKPT, TREAT_FEATURES_CKPT, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    all_logits, all_labels = [], []
    for x_comb, y in loader:
        all_logits.append(model(x_comb.to(device)).cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return 1.0 / (1.0 + np.exp(-logits)), labels, ds.stay_ids

print("\nGetting predictions on post-drift held-out set...")
probs_a, labels, stay_ids = get_two_stream_preds(model_A, eval_post_df)
probs_b, _,      sids_b   = get_two_stream_preds(model_B, eval_post_df)
probs_c, _,      sids_c   = get_two_stream_preds(model_C, eval_post_df)
probs_d, _,      sids_d   = get_single_stream_preds(model_D, eval_post_df)

assert sids_b == stay_ids, "Run B stay-id ordering mismatch"
assert sids_c == stay_ids, "Run C stay-id ordering mismatch"
assert sids_d == stay_ids, "Run D stay-id ordering mismatch"
print(f"  Stay-id alignment confirmed for all four models: {len(stay_ids)} stays")

# ── LOAD XGBOOST PREDICTIONS ──────────────────────────────────────────────────
# FIX 5: xgb_predictions_corrected.npz is saved to SAVE_PATH by script5.
# If running in a new session where SAVE_PATH was cleared, copy the file to
# BASE_PATH2 and add that as a fallback here.
xgb_npz_candidates = [
    SAVE_PATH  / "xgb_predictions_corrected.npz",
    BASE_PATH2 / "xgb_predictions_corrected.npz",
]
xgb_npz_path = next((p for p in xgb_npz_candidates if p.exists()), None)
if xgb_npz_path is None:
    raise FileNotFoundError(
        "xgb_predictions_corrected.npz not found. Run script5 first.")

xgb_data = np.load(xgb_npz_path)
probs_xgb_source_raw  = xgb_data["probs_xgb_source"]
probs_xgb_adapted_raw = xgb_data["probs_xgb_adapted"]
xgb_sids_raw          = xgb_data["stay_ids"].tolist()

assert set(xgb_sids_raw) == set(stay_ids), \
    "Patient sets do not match between PyTorch and XGBoost"

xgb_index = {sid: i for i, sid in enumerate(xgb_sids_raw)}
reorder   = [xgb_index[sid] for sid in stay_ids]
probs_xgb_source  = probs_xgb_source_raw[reorder]
probs_xgb_adapted = probs_xgb_adapted_raw[reorder]

assert [xgb_sids_raw[i] for i in reorder] == stay_ids, "Realignment failed"
print(f"✅ XGBoost realigned to PyTorch ordering ({len(stay_ids)} stays)")

# ── DISAGREEMENT MATRIX FUNCTIONS ─────────────────────────────────────────────
def disagreement_matrix(probs_X, probs_Y, y_true,
                        tau_high=0.5, tau_low=0.1):
    pos_mask       = (y_true == 1)
    n_pos          = int(pos_mask.sum())
    if n_pos == 0:
        return {"n_pos": 0, "x_catch_y_miss": 0, "y_catch_x_miss": 0,
                "both_catch": 0, "both_miss": 0,
                "asymmetry": float("nan"),
                "tau_high": tau_high, "tau_low": tau_low}
    pX, pY         = probs_X[pos_mask], probs_Y[pos_mask]
    x_catch_y_miss = int(np.sum((pX >= tau_high) & (pY <  tau_low)))
    y_catch_x_miss = int(np.sum((pY >= tau_high) & (pX <  tau_low)))
    both_catch     = int(np.sum((pX >= tau_high) & (pY >= tau_high)))
    both_miss      = int(np.sum((pX <  tau_high) & (pY <  tau_high)))
    asym           = (x_catch_y_miss / max(y_catch_x_miss, 1)
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

# ── PAIRWISE COMPARISONS ──────────────────────────────────────────────────────
COMPARISONS = [
    ("RunB_vs_XGBadapted", "Run B",  probs_b, "XGBoost-adapted", probs_xgb_adapted),
    ("RunB_vs_RunC",       "Run B",  probs_b, "Run C",           probs_c),
    ("RunC_vs_XGBadapted", "Run C",  probs_c, "XGBoost-adapted", probs_xgb_adapted),
    ("RunB_vs_RunA",       "Run B",  probs_b, "Run A",           probs_a),
    ("RunB_vs_RunD",       "Run B",  probs_b, "Run D",           probs_d),
    ("RunC_vs_RunD",       "Run C",  probs_c, "Run D",           probs_d),
    ("RunD_vs_XGBadapted", "Run D",  probs_d, "XGBoost-adapted", probs_xgb_adapted),
    ("RunD_vs_RunA",       "Run D",  probs_d, "Run A",           probs_a),
]

print("\n" + "="*78)
print(f"DISAGREEMENT MATRICES  (catch>={TAU_HIGH_PRIMARY}, miss<{TAU_LOW_PRIMARY})")
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
        print(f"    {name_X} catches, {name_Y} misses : {xY:>4}  ({100*xY/max(n_pos,1):>5.1f}%)")
        print(f"    {name_Y} catches, {name_X} misses : {yX:>4}  ({100*yX/max(n_pos,1):>5.1f}%)")
        print(f"    Both catch                        : {bc:>4}  ({100*bc/max(n_pos,1):>5.1f}%)")
        print(f"    Both miss                         : {bm:>4}  ({100*bm/max(n_pos,1):>5.1f}%)")
        print(f"    Asymmetry ratio ({name_X}/{name_Y})  : {asym_str}")
        print(f"    False-alarm (n_neg={fa['n_neg']}):  "
              f"{name_X} alarms/{name_Y} quiet={fa['x_alarm_y_quiet']}  |  "
              f"{name_Y} alarms/{name_X} quiet={fa['y_alarm_x_quiet']}")
    all_results[cmp_key] = cmp_results

# ── THRESHOLD SENSITIVITY ─────────────────────────────────────────────────────
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

# ── WORKED-EXAMPLE PATIENTS ───────────────────────────────────────────────────
# FIX 6: Worked examples print "NOT in held-out set" gracefully instead of
# crashing if subject-level split moved these stays out of eval_post.
# The original hardcoded stay_ids are kept as the primary lookup; if not found,
# a fallback picks the highest-confidence true-positive from that label.
print("\n" + "="*78)
print("WORKED-EXAMPLE PATIENTS — Run A / B / C / D / XGBoost probabilities")
print("="*78)

# ── Dynamically select best worked-example patient per label ──────────────────
# Criteria: true positive where Run B and XGBoost-adapted disagree most
# (Run B high confidence, XGBoost-adapted low) — most narratively useful.
WORKED_EXAMPLES = []
label_descs = {
    "label_vasopressor":  "Patient A — vasopressor",
    "label_intubation":   "Patient C — intubation",
    "label_septic_shock": "Patient B — septic shock",
}

for lbl, desc in label_descs.items():
    li       = LABEL_COLS.index(lbl)
    pos_mask = labels[:, li] == 1
    if pos_mask.sum() == 0:
        print(f"  ⚠ No positives for {lbl} in held-out set — skipping")
        continue
    # Score = Run B confidence - XGBoost-adapted confidence (among true positives)
    score    = (probs_b[:, li] - probs_xgb_adapted[:, li]) * pos_mask
    best_idx = int(np.argmax(score))
    best_sid = stay_ids[best_idx]
    print(f"  Selected {desc}: stay_id={best_sid}, "
          f"Run B={probs_b[best_idx, li]:.4f}, "
          f"XGBoost-adapted={probs_xgb_adapted[best_idx, li]:.4f}, "
          f"true={int(labels[best_idx, li])}")
    WORKED_EXAMPLES.append((desc, lbl, best_sid))

stay_id_arr = np.array(stay_ids)
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
    
# ── VISUALIZATIONS ────────────────────────────────────────────────────────────
print("\nGenerating disagreement-matrix figures...")

def _plot_disagreement_grid(cmp_keys, title, filename):
    fig, axes = plt.subplots(3, len(cmp_keys), figsize=(5 * len(cmp_keys), 11))
    if len(cmp_keys) == 1:
        axes = axes.reshape(3, 1)
    for col, cmp_key in enumerate(cmp_keys):
        cmp    = all_results[cmp_key]
        name_X = cmp["model_X"]
        name_Y = cmp["model_Y"]
        for row, lbl in enumerate(LABEL_COLS):
            ax  = axes[row, col]
            dm  = cmp["by_label"][lbl]["positives"]
            bars_vals = [dm["x_catch_y_miss"], dm["y_catch_x_miss"],
                         dm["both_catch"],      dm["both_miss"]]
            bar_names = [f"{name_X} catches\n{name_Y} misses",
                         f"{name_Y} catches\n{name_X} misses",
                         "Both catch", "Both miss"]
            colors    = ["#16a34a","#dc2626","#2563eb","#9ca3af"]
            bars_obj  = ax.bar(range(4), bars_vals, color=colors,
                               alpha=0.8, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(4))
            ax.set_xticklabels(bar_names, fontsize=7)
            ax.set_ylabel("Count" if col == 0 else "")
            if row == 0:
                ax.set_title(f"{name_X} vs {name_Y}", fontsize=10, fontweight="bold")
            for b, v in zip(bars_obj, bars_vals):
                ax.text(b.get_x() + b.get_width()/2,
                        b.get_height() + max(bars_vals)*0.01,
                        str(v), ha="center", va="bottom", fontsize=8)
            lbl_short = lbl.replace("label_","").replace("_"," ").title()
            ax.text(0.02, 0.95, f"{lbl_short}\n(n_pos={dm['n_pos']})",
                    transform=ax.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
            ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle(
        f"{title}\n(n={len(stay_ids)} post-drift stays, "
        f"catch≥{TAU_HIGH_PRIMARY}, miss<{TAU_LOW_PRIMARY})",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SAVE_PATH / filename, dpi=150, bbox_inches="tight")
    print(f"Saved → {SAVE_PATH / filename}")
    plt.close(fig)

_plot_disagreement_grid(
    ["RunB_vs_XGBadapted", "RunB_vs_RunC", "RunC_vs_XGBadapted"],
    "Disagreement matrices — original three comparisons",
    "disagreement_matrices_abc.png")

_plot_disagreement_grid(
    ["RunB_vs_RunD", "RunC_vs_RunD", "RunD_vs_XGBadapted"],
    "Disagreement matrices — Run D (single-stream) comparisons",
    "disagreement_matrices_runD.png")

# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────
np.savez(
    SAVE_PATH / "post_drift_predictions_with_runD.npz",
    stay_ids          = np.array(stay_ids),
    labels            = labels,
    probs_run_a       = probs_a,
    probs_run_b       = probs_b,
    probs_run_c       = probs_c,
    probs_run_d       = probs_d,
    probs_xgb_source  = probs_xgb_source,
    probs_xgb_adapted = probs_xgb_adapted,
)
print(f"Saved → {SAVE_PATH / 'post_drift_predictions_with_runD.npz'}")

def to_serialisable(obj):
    if isinstance(obj, dict):   return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [to_serialisable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:          return None
        if obj == float("inf"): return "inf"
        return round(obj, 6)
    if isinstance(obj, (np.float32, np.float64)):
        if obj != obj:          return None
        return float(round(obj, 6))
    if isinstance(obj, (np.int32, np.int64, int)): return int(obj)
    return obj

output = {
    "thresholds":            {"primary_tau_high": TAU_HIGH_PRIMARY,
                               "primary_tau_low":  TAU_LOW_PRIMARY},
    "n_held_out_stays":      len(stay_ids),
    "comparisons":           all_results,
    "threshold_sensitivity": sensitivity,
    "worked_examples":       worked_table,
}
with open(SAVE_PATH / "disagreement_matrix_results_with_runD.json", "w") as f:
    json.dump(to_serialisable(output), f, indent=2)
print(f"Saved → {SAVE_PATH / 'disagreement_matrix_results_with_runD.json'}")

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
        dm        = all_results[cmp_key]["by_label"][lbl]["positives"]
        fa        = all_results[cmp_key]["by_label"][lbl]["negatives"]
        lbl_short = lbl.replace("label_","").replace("_"," ")
        asym_str  = (f"{dm['asymmetry']:.1f}x"
                     if np.isfinite(dm["asymmetry"]) else "∞")
        print(f"    {lbl_short:<18}  "
              f"{name_X}-catches/{name_Y}-misses = {dm['x_catch_y_miss']:>3}  "
              f"{name_Y}-catches/{name_X}-misses = {dm['y_catch_x_miss']:>3}  "
              f"asym={asym_str}  |  "
              f"FA: {name_X}={fa['x_alarm_y_quiet']} {name_Y}={fa['y_alarm_x_quiet']}")
    print()

# ── RUN B VS ALL 5 OTHER MODELS (MASTER SUMMARY) ─────────────────────────────
print("\n" + "="*95)
print("RUN B vs ALL 5 OTHER MODELS (Disagreement Summary)")
print(f"Thresholds: Catch >= {TAU_HIGH_PRIMARY}, Miss < {TAU_LOW_PRIMARY}")
print("="*95)

other_models = [
    ("Run A (Static)",         probs_a),
    ("Run C (Full Two-Stream)", probs_c),
    ("Run D (Single-Stream)",  probs_d),
    ("XGBoost (Source)",       probs_xgb_source),
    ("XGBoost (Adapted)",      probs_xgb_adapted),
]

for i, lbl in enumerate(LABEL_COLS):
    lbl_short = lbl.replace("label_", "").upper()
    print(f"\n── LABEL: {lbl_short} " + "─"*70)
    print(f"  {'Model compared to Run B':<25} | {'Run B Catches / Other Misses':>28} | "
          f"{'Other Catches / Run B Misses':>28} | {'Asymmetry':>9}")
    print("  " + "-"*96)
    for name_other, probs_other in other_models:
        dm       = disagreement_matrix(probs_b[:, i], probs_other[:, i], labels[:, i],
                                       tau_high=TAU_HIGH_PRIMARY, tau_low=TAU_LOW_PRIMARY)
        asym_str = f"{dm['asymmetry']:.2f}x" if np.isfinite(dm["asymmetry"]) else "∞"
        print(f"  {name_other:<25} | {dm['x_catch_y_miss']:>28} | "
              f"{dm['y_catch_x_miss']:>28} | {asym_str:>9}")

print("\n✅ Complete")