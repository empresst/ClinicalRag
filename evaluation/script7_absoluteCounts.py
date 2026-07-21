%%writefile evaluation/script7_absolute_counts.py
"""
script7_absolute_counts.py
══════════════════════════
Absolute TP / FP / TN / FN counts for every model individually.
Run AFTER script6 (which saves post_drift_predictions_with_runD.npz).

Models: Run A (frozen), Run B (two-stream selective), Run C (two-stream full),
        Run D (single-stream), XGBoost-source, XGBoost-adapted

Labels: vasopressor, intubation, septic_shock

Threshold: prob >= 0.5  →  predicted positive  (standard 0.5 cut-off)
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAVE_PATH   = Path("/kaggle/working")
THRESHOLD   = 0.5          # classification threshold for TP/FP/TN/FN
LABEL_COLS  = ["label_vasopressor", "label_intubation", "label_septic_shock"]
LABEL_SHORT = ["vasopressor", "intubation", "septic_shock"]

# Additional thresholds to sweep (optional sensitivity block)
EXTRA_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

# ── LOAD PREDICTIONS ──────────────────────────────────────────────────────────
npz_candidates = [
    SAVE_PATH / "post_drift_predictions_with_runD.npz",
]
npz_path = next((p for p in npz_candidates if p.exists()), None)
if npz_path is None:
    raise FileNotFoundError(
        "post_drift_predictions_with_runD.npz not found.\n"
        "Run script6_disagreement_matrix.py first.")

data     = np.load(npz_path)
labels   = data["labels"]           # shape (N, 3)
probs_a  = data["probs_run_a"]      # shape (N, 3)
probs_b  = data["probs_run_b"]
probs_c  = data["probs_run_c"]
probs_d  = data["probs_run_d"]
probs_xs = data["probs_xgb_source"]
probs_xa = data["probs_xgb_adapted"]
N        = labels.shape[0]

print(f"✅ Loaded predictions: {N} post-drift held-out stays")

# ── MODELS DICT ───────────────────────────────────────────────────────────────
MODELS = {
    "Run A  (frozen)              ": probs_a,
    "Run B  (two-stream selective)": probs_b,
    "Run C  (two-stream full)     ": probs_c,
    "Run D  (single-stream head)  ": probs_d,
    "XGBoost (source)             ": probs_xs,
    "XGBoost (adapted)            ": probs_xa,
}

# ── HELPER ────────────────────────────────────────────────────────────────────
def compute_counts(probs_col, y_true, tau=0.5):
    """Return TP, FP, TN, FN and derived metrics for one model × one label."""
    preds = (probs_col >= tau).astype(int)
    TP = int(np.sum((preds == 1) & (y_true == 1)))
    FP = int(np.sum((preds == 1) & (y_true == 0)))
    TN = int(np.sum((preds == 0) & (y_true == 0)))
    FN = int(np.sum((preds == 0) & (y_true == 1)))
    sensitivity = TP / max(TP + FN, 1)   # recall / TPR
    specificity = TN / max(TN + FP, 1)
    precision   = TP / max(TP + FP, 1)
    f1          = (2 * precision * sensitivity) / max(precision + sensitivity, 1e-9)
    try:
        auroc = roc_auc_score(y_true, probs_col)
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(y_true, probs_col)
    except ValueError:
        auprc = float("nan")
    return dict(TP=TP, FP=FP, TN=TN, FN=FN,
                sensitivity=sensitivity, specificity=specificity,
                precision=precision, F1=f1,
                AUROC=auroc, AUPRC=auprc)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PER-LABEL ABSOLUTE COUNTS  (primary threshold = 0.5)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*90)
print(f"ABSOLUTE TP / FP / TN / FN  —  threshold = {THRESHOLD}")
print(f"Post-drift held-out set  (N = {N} stays)")
print("="*90)

all_results = {}

for li, (lbl_col, lbl_short) in enumerate(zip(LABEL_COLS, LABEL_SHORT)):
    y = labels[:, li]
    n_pos = int(y.sum())
    n_neg = N - n_pos

    print(f"\n{'─'*90}")
    print(f"  LABEL: {lbl_short.upper():<20}  n_pos={n_pos}  n_neg={n_neg}  "
          f"prevalence={100*n_pos/N:.1f}%")
    print(f"{'─'*90}")
    print(f"  {'Model':<34}  {'TP':>5}  {'FP':>6}  {'TN':>6}  {'FN':>5}  "
          f"{'Sens':>6}  {'Spec':>6}  {'Prec':>6}  {'F1':>6}  "
          f"{'AUROC':>7}  {'AUPRC':>7}")
    print(f"  {'─'*34}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*5}  "
          f"{'─'*6}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")

    label_results = {}
    for model_name, probs in MODELS.items():
        r = compute_counts(probs[:, li], y, tau=THRESHOLD)
        print(f"  {model_name}  "
              f"{r['TP']:>5}  {r['FP']:>6}  {r['TN']:>6}  {r['FN']:>5}  "
              f"{r['sensitivity']:>6.3f}  {r['specificity']:>6.3f}  "
              f"{r['precision']:>6.3f}  {r['F1']:>6.3f}  "
              f"{r['AUROC']:>7.4f}  {r['AUPRC']:>7.4f}")
        label_results[model_name.strip()] = r

    all_results[lbl_short] = label_results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — THRESHOLD SWEEP  (TP / FP only, most clinically relevant)
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*90)
print("THRESHOLD SWEEP  — TP and FP counts at multiple cut-offs")
print("="*90)

for li, (lbl_col, lbl_short) in enumerate(zip(LABEL_COLS, LABEL_SHORT)):
    y     = labels[:, li]
    n_pos = int(y.sum())
    n_neg = N - n_pos

    print(f"\n── {lbl_short.upper()}  (n_pos={n_pos}, n_neg={n_neg}) ──────────────────────────────────────────")

    # Header
    hdr = f"  {'Model':<34}"
    for tau in EXTRA_THRESHOLDS:
        hdr += f"  τ={tau:.1f}(TP/FP)"
    print(hdr)
    print(f"  {'─'*34}" + ("  " + "─"*12) * len(EXTRA_THRESHOLDS))

    for model_name, probs in MODELS.items():
        row = f"  {model_name}"
        for tau in EXTRA_THRESHOLDS:
            r   = compute_counts(probs[:, li], y, tau=tau)
            row += f"  {r['TP']:>4}/{r['FP']:<5}"
        print(row)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — COMPACT SUMMARY TABLE  (paper-friendly)
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*90)
print(f"COMPACT SUMMARY  (threshold={THRESHOLD})  — paper-friendly view")
print("="*90)

COL_W = 22
print(f"\n  {'':34}", end="")
for lbl_short in LABEL_SHORT:
    print(f"  {lbl_short.upper():^{COL_W}}", end="")
print()

print(f"  {'Model':34}", end="")
for _ in LABEL_SHORT:
    print(f"  {'TP':>4}  {'FP':>5}  {'FN':>4}  {'Sens':>5}", end="")
print()

print(f"  {'─'*34}", end="")
for _ in LABEL_SHORT:
    print(f"  {'─'*4}  {'─'*5}  {'─'*4}  {'─'*5}", end="")
print()

for model_name, probs in MODELS.items():
    print(f"  {model_name}", end="")
    for li, lbl_short in enumerate(LABEL_SHORT):
        y = labels[:, li]
        r = compute_counts(probs[:, li], y, tau=THRESHOLD)
        print(f"  {r['TP']:>4}  {r['FP']:>5}  {r['FN']:>4}  {r['sensitivity']:>5.3f}", end="")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SAVE JSON
# ─────────────────────────────────────────────────────────────────────────────
def to_serialisable(obj):
    if isinstance(obj, dict):   return {k: to_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [to_serialisable(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj:           return None          # NaN
        if obj == float("inf"):  return "inf"
        return round(obj, 6)
    if isinstance(obj, (np.float32, np.float64)):
        return None if (obj != obj) else float(round(obj, 6))
    if isinstance(obj, (np.int32, np.int64, int)): return int(obj)
    return obj

output = {
    "threshold":         THRESHOLD,
    "n_held_out_stays":  N,
    "label_results":     all_results,
}
out_path = SAVE_PATH / "absolute_counts_per_model.json"
with open(out_path, "w") as f:
    json.dump(to_serialisable(output), f, indent=2)

print(f"\n✅ Saved → {out_path}")
print("✅ Complete")s