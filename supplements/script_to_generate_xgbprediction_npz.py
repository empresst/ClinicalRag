import numpy as np
import polars as pl
from pathlib import Path

BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")
s2        = Path("/kaggle/input/datasets/fatematamanna/ptfiles")
SAVE_PATH = Path("/kaggle/working")

# Load eval_post_stays to get the held-out set
import json
with open(s2 / "eval_post_stays.json") as f:
    eval_post_stays = json.load(f)

# Load and normalize test data
test_df = normalize(load_split("test"), train_stats)
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))

# Get PyTorch ordering — this is the reference ordering
from torch.utils.data import DataLoader
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