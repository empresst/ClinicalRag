"""
compute_brier_scores.py
═══════════════════════
Standalone Brier score table for all 5 models on post-drift held-out set.

Inputs (all from /kaggle/working):
  post_drift_predictions.npz     → probs_a, probs_b, probs_c, labels
  xgb_predictions_corrected.npz → probs_xgb_source, probs_xgb_adapted, labels, stay_ids

No forward passes, no model loading, no data loading needed.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

SAVE_PATH  = Path("/kaggle/working")
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]

# ── Load ───────────────────────────────────────────────────────────────────────
ts  = np.load(SAVE_PATH / "post_drift_predictions.npz")
xgb = np.load(SAVE_PATH / "xgb_predictions_corrected.npz", allow_pickle=True)

labels    = ts["labels"]          # ground truth — same across both files
probs_a   = ts["probs_a"]
probs_b   = ts["probs_b"]
probs_c   = ts["probs_c"]
probs_xgs = xgb["probs_xgb_source"]
probs_xga = xgb["probs_xgb_adapted"]

# Sanity: labels must agree between the two npz files
assert np.array_equal(labels, xgb["labels"]), (
    "Label mismatch between post_drift_predictions.npz and "
    "xgb_predictions_corrected.npz — stay ordering differs. "
    "Re-run script5 to regenerate xgb_predictions_corrected.npz."
)
print("✅ Label alignment verified across both npz files")
print(f"   Patients: {labels.shape[0]} | Labels: {labels.shape[1]}\n")

# ── Compute ────────────────────────────────────────────────────────────────────
models = {
    "Run A (static)"      : probs_a,
    "Run B (partial adapt)": probs_b,
    "Run C (full adapt)"  : probs_c,
    "XGBoost (source)"    : probs_xgs,
    "XGBoost (adapted)"   : probs_xga,
}

results = {name: {} for name in models}

for name, probs in models.items():
    for i, lbl in enumerate(LABEL_COLS):
        y, p = labels[:, i], probs[:, i]
        n_pos = int(y.sum())
        if n_pos > 0 and n_pos < len(y):
            results[name][lbl] = {
                "auroc" : float(roc_auc_score(y, p)),
                "auprc" : float(average_precision_score(y, p)),
                "brier" : float(brier_score_loss(y, p)),
                "n_pos" : n_pos,
            }
        else:
            results[name][lbl] = {
                "auroc": float("nan"), "auprc": float("nan"),
                "brier": float("nan"), "n_pos": n_pos,
            }

# ── Print ──────────────────────────────────────────────────────────────────────
SHORT = [l.replace("label_", "") for l in LABEL_COLS]

def fmt(v):
    return f"{v:.4f}" if not np.isnan(v) else "  N/A"

# --- AUROC ---
print("=" * 72)
print("POST-DRIFT HELD-OUT — AUROC")
print("=" * 72)
print(f"  {'Model':<26} {SHORT[0]:>12} {SHORT[1]:>14} {SHORT[2]:>14}")
print("  " + "-" * 66)
for name, res in results.items():
    vals = [fmt(res[l]["auroc"]) for l in LABEL_COLS]
    print(f"  {name:<26} {vals[0]:>12} {vals[1]:>14} {vals[2]:>14}")

# --- AUPRC ---
print(f"\n{'=' * 72}")
print("POST-DRIFT HELD-OUT — AUPRC")
print("=" * 72)
print(f"  {'Model':<26} {SHORT[0]:>12} {SHORT[1]:>14} {SHORT[2]:>14}")
print("  " + "-" * 66)
for name, res in results.items():
    vals = [fmt(res[l]["auprc"]) for l in LABEL_COLS]
    print(f"  {name:<26} {vals[0]:>12} {vals[1]:>14} {vals[2]:>14}")

# --- Brier ---
print(f"\n{'=' * 72}")
print("POST-DRIFT HELD-OUT — BRIER SCORE  (lower = better)")
print("=" * 72)
print(f"  {'Model':<26} {SHORT[0]:>12} {SHORT[1]:>14} {SHORT[2]:>14}")
print("  " + "-" * 66)
for name, res in results.items():
    vals = [fmt(res[l]["brier"]) for l in LABEL_COLS]
    print(f"  {name:<26} {vals[0]:>12} {vals[1]:>14} {vals[2]:>14}")

# --- Brier deltas vs Run A ---
print(f"\n{'=' * 72}")
print("BRIER DELTA vs Run A  (negative = better than Run A)")
print("=" * 72)
print(f"  {'Model':<26} {SHORT[0]:>12} {SHORT[1]:>14} {SHORT[2]:>14}")
print("  " + "-" * 66)
ref = results["Run A (static)"]
for name, res in results.items():
    if name == "Run A (static)":
        continue
    deltas = []
    for l in LABEL_COLS:
        b = res[l]["brier"]
        a = ref[l]["brier"]
        deltas.append(f"{b - a:+.4f}" if not (np.isnan(b) or np.isnan(a)) else "  N/A")
    print(f"  {name:<26} {deltas[0]:>12} {deltas[1]:>14} {deltas[2]:>14}")

# --- n_pos reminder ---
print(f"\n  n_pos per label (same for all models):")
for l in LABEL_COLS:
    print(f"    {l}: {results['Run A (static)'][l]['n_pos']}")

print("\n✅ Done")