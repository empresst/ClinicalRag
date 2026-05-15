#%%writefile models/script3_run_d_single_stream.py
"""
script_run_d_single_stream.py
══════════════════════════════
Run D: Single-Stream Late-Fusion baseline.

Architecture:
  - ONE LSTM on all 105 features (seq + treatment concatenated at input).
  - A small fusion head on top of the LSTM hidden state.
  - During adaptation: LSTM frozen, only the final 1-2 dense layers are updated.

Purpose (MC2 response):
  This baseline isolates the contribution of the TWO-STREAM STRUCTURAL
  DECOMPOSITION itself, independent of the freeze strategy.

  Run B = two-stream architecture + LSTM freeze
  Run C = two-stream architecture + no freeze
  Run D = single-stream architecture + last-layer freeze (monolithic selective adapt)

  If Run B/C > Run D on post-drift metrics, the architectural decomposition
  (not just the freeze) is doing real work. If Run D ≈ Run B/C, then selective
  freezing of any layer is sufficient and the two-stream prior adds less value.

Usage:
  Load the same source train/val/test splits as script2_three_stream_model_v4.py.
  This script is designed to be run AFTER that script has already defined
  SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, and the data loaders.
  Alternatively, run standalone by copying the data-loading block below.

  The script produces:
    - run_d_model.pt          (saved model weights)
    - run_d_results.json      (AUROC / AUPRC / Brier for pre- and post-drift)
    - run_d_comparison.png    (Run A vs B vs C vs D AUROC plot)
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
from models.architectures import PhysiologyStream, TreatmentStream, FusionHead, SingleStreamModel


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CONFIG (must match script2_three_stream_model_v4.py exactly) ──────────────
SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 42, 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LR_INIT, LR_ADAPT = 1e-3, 3e-4
EPOCHS, ADAPT_EPOCHS = 50, 40
PATIENCE, ADAPT_PATIENCE = 8, 8
BUFFER_SIZE, PSI_THRESH = 500, 0.20
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH  = Path("/kaggle/working")

torch.manual_seed(SEED); np.random.seed(SEED)

# ── DATA LOADING (identical to v4) ────────────────────────────────────────────
# 1. Pass the constants directly into your new tool
train_df = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
val_df   = load_enriched_split(BASE_PATH, "val",   SEQ_FEATURES, TREATMENT_FEATURES)
test_df  = load_enriched_split(BASE_PATH, "test",  SEQ_FEATURES, TREATMENT_FEATURES)

print("Calculating statistics and normalizing...")
# 2. Get the stats using your new tool
all_norm_cols = list(set(SEQ_FEATURES + TREATMENT_FEATURES))
train_stats = calculate_train_stats(train_df, all_norm_cols)

# 3. Apply the normalization function you already saved!
train_df = normalize(train_df, train_stats)
val_df   = normalize(val_df,   train_stats)
test_df  = normalize(test_df,  train_stats)
print("✅ Data Loaded and Normalized!")

print("\nBuilding single-stream datasets...")
train_ss_ds = SingleStreamDataset(train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
val_ss_ds   = SingleStreamDataset(val_df,   SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
test_ss_ds  = SingleStreamDataset(test_df,  SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)

combined_input_dim = train_ss_ds.combined_data.shape[2]
print(f"Single-stream input dim (seq + treat): {combined_input_dim}")
print(f"Train: {len(train_ss_ds)} | Val: {len(val_ss_ds)} | Test: {len(test_ss_ds)}")

train_ss_loader = DataLoader(train_ss_ds, batch_size=BATCH_SIZE, shuffle=True)
val_ss_loader   = DataLoader(val_ss_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_ss_loader  = DataLoader(test_ss_ds,  batch_size=BATCH_SIZE, shuffle=False)

# ── LOSS (identical to v4) ────────────────────────────────────────────────────

pos_weights = compute_pos_weights(train_ss_ds, max_weight=20.0)
criterion   = FocalBCEWithLogitsLoss(
    pos_weight=pos_weights, gamma=1.5, label_smoothing=0.02)

# ── TRAINING UTILITIES ─────────────────────────────────────────────────────────
def train_epoch_ss(model, loader, optimizer, crit):
    """Single-stream training step: loader yields (x_combined, y)."""
    model.train()
    total_loss, n = 0, 0
    for x_comb, y in loader:
        x_comb, y = x_comb.to(device), y.to(device)
        loss = crit(model(x_comb), y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        n          += y.size(0)
    return total_loss / n


@torch.no_grad()
def evaluate_ss(model, loader, crit):
    """Single-stream evaluation: loader yields (x_combined, y)."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0, 0
    for x_comb, y in loader:
        x_comb, y = x_comb.to(device), y.to(device)
        logits      = model(x_comb)
        total_loss += crit(logits, y).item() * y.size(0)
        n          += y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))

    metrics = {"loss": total_loss / max(n, 1)}
    for i, lbl in enumerate(LABEL_COLS):
        y_true, y_prob = labels[:, i], probs[:, i]
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - y_true.sum())
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


def train_model_ss(model, train_loader, val_loader, crit, lr,
                   epochs, patience, tag=""):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    best_loss, best_state, wait = float("inf"), None, 0
    for ep in range(1, epochs + 1):
        t_loss          = train_epoch_ss(model, train_loader, optimizer, crit)
        v_met, _, _     = evaluate_ss(model, val_loader, crit)
        scheduler.step(v_met["loss"])

        if v_met["loss"] < best_loss:
            best_loss  = v_met["loss"]
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1

        if ep % 5 == 0 or wait == 0:
            aurocs = [v_met.get(f"{l}_auroc", 0) for l in LABEL_COLS]
            print(f"  {tag}Ep {ep:2d} | tL={t_loss:.4f} vL={v_met['loss']:.4f} "
                  f"mAUROC={np.nanmean(aurocs):.4f} {'*' if wait == 0 else ''}")
        if wait >= patience:
            print(f"  {tag}Early stop at epoch {ep}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ── PHASE 1: TRAIN SOURCE (Run D base model) ──────────────────────────────────
print("\n" + "="*60)
print("PHASE 1: Training single-stream source model (Run D base)")
print("="*60)

model_D = SingleStreamModel(
    input_dim  = combined_input_dim,
    hidden_dim = HIDDEN_DIM,
    n_layers   = LSTM_LAYERS,
    dropout    = DROPOUT,
    n_targets  = len(LABEL_COLS)
).to(device)

model_D = train_model_ss(
    model_D, train_ss_loader, val_ss_loader,
    criterion, LR_INIT, EPOCHS, PATIENCE, "D-SRC ")

val_met_d, _, _ = evaluate_ss(model_D, val_ss_loader, criterion)
print(f"\nRun D source on VAL:")
for lbl in LABEL_COLS:
    print(f"  {lbl}: AUROC={val_met_d.get(f'{lbl}_auroc', 0):.4f} "
          f"AUPRC={val_met_d.get(f'{lbl}_auprc', 0):.4f} "
          f"(n_pos={val_met_d.get(f'{lbl}_n_pos', 0)})")

source_state_D = copy.deepcopy(model_D.state_dict())

# ── PHASE 2: DRIFT DETECTION (reuse logic from v4) ────────────────────────────
# (Assumes test_pre and test_post are already defined from the shared data loading)
test_pre  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")
print(f"\nPre-drift:  {test_pre['stay_id'].n_unique()} stays")
print(f"Post-drift: {test_post['stay_id'].n_unique()} stays")

# ── PHASE 3: ADAPTATION (Run D — last-layer only) ─────────────────────────────
print("\n" + "="*60)
print("PHASE 3: Run D adaptation (LSTM frozen, head trainable)")
print("="*60)

adaptation_performed_D = False
eval_post_stays_D = []

if test_post["stay_id"].n_unique() > 50:
    post_stays_D   = (test_post.filter(pl.col("hrs_from_admit") == 0)
                               .sort("intime")["stay_id"].to_list())
    n_total_post   = len(post_stays_D)

    n_adapt_train  = int(n_total_post * 0.30)
    n_adapt_val    = int(n_total_post * 0.10)

    adapt_train_stays_D = post_stays_D[:n_adapt_train]
    adapt_val_stays_D   = post_stays_D[n_adapt_train:n_adapt_train + n_adapt_val]
    eval_post_stays_D   = post_stays_D[n_adapt_train + n_adapt_val:]

    print(f"Post-drift split: adapt-train={len(adapt_train_stays_D)} "
          f"adapt-val={len(adapt_val_stays_D)} "
          f"held-out={len(eval_post_stays_D)}")

    # Pre-drift stability buffer (same logic as v4)
    pre_stays_D   = (test_pre.select("stay_id").unique()
                             .sort("stay_id")["stay_id"].to_list())
    buf_pre_stays_D = (pre_stays_D[-BUFFER_SIZE:]
                       if len(pre_stays_D) > BUFFER_SIZE else pre_stays_D)
    buf_pre_df_D  = test_pre.filter(pl.col("stay_id").is_in(buf_pre_stays_D))

    adapt_train_df_D = test_post.filter(
        pl.col("stay_id").is_in(adapt_train_stays_D))
    adapt_val_df_D   = test_post.filter(
        pl.col("stay_id").is_in(adapt_val_stays_D))

    combined_adapt_train_D = pl.concat([buf_pre_df_D, adapt_train_df_D])

    adapt_train_ss_ds = SingleStreamDataset(
        combined_adapt_train_D, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    adapt_val_ss_ds   = SingleStreamDataset(
        adapt_val_df_D, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)

    adapt_train_ss_loader = DataLoader(
        adapt_train_ss_ds, batch_size=BATCH_SIZE, shuffle=True)
    adapt_val_ss_loader   = DataLoader(
        adapt_val_ss_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Adapt train: {len(adapt_train_ss_ds)} "
          f"({buf_pre_df_D['stay_id'].n_unique()} pre + "
          f"{adapt_train_df_D['stay_id'].n_unique()} post)")
    print(f"Adapt val:   {len(adapt_val_ss_ds)} (post-drift, for early stopping)")

    adapt_pos_weights_D = compute_pos_weights(adapt_train_ss_ds, max_weight=15.0)
    adapt_criterion_D   = FocalBCEWithLogitsLoss(
        pos_weight=adapt_pos_weights_D, gamma=1.0, label_smoothing=0.05)

    # Freeze LSTM, unfreeze head only
    model_D.unfreeze_head_only()
    trainable = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model_D.parameters())
    print(f"\nRun D: {trainable:,} / {total:,} parameters trainable during adaptation "
          f"({100*trainable/total:.1f}%)")

    model_D = train_model_ss(
        model_D, adapt_train_ss_loader, adapt_val_ss_loader,
        adapt_criterion_D, LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE, "D-ADAPT ")
    print("✅ Run D adapted (LSTM frozen, head fine-tuned)")
    adaptation_performed_D = True

else:
    print("Insufficient post-drift stays — Run D unchanged")
    post_stays_D      = (test_post.filter(pl.col("hrs_from_admit") == 0)
                                  .sort("intime")["stay_id"].to_list())
    eval_post_stays_D = post_stays_D

# ── PHASE 4: EVALUATION ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("PHASE 4: Final evaluation — Run D")
print("="*60)

# Frozen source copy for pre-drift evaluation (honest: no adaptation applied)
model_D_pre = SingleStreamModel(
    combined_input_dim, HIDDEN_DIM, LSTM_LAYERS, DROPOUT, len(LABEL_COLS)
).to(device)
model_D_pre.load_state_dict(source_state_D)
model_D_pre.freeze_all()


def eval_on_split_ss(model, df, crit):
    ds     = SingleStreamDataset(
        df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return evaluate_ss(model, loader, crit)


results_D = {}

print("\n--- Val (source weights) ---")
met_d_val, _, _ = eval_on_split_ss(model_D_pre, val_df, criterion)
results_D["Val (source)"] = met_d_val

print("--- Test-Pre (source weights, pre-adaptation) ---")
met_d_pre, _, _ = eval_on_split_ss(model_D_pre, test_pre, criterion)
results_D["Test-Pre (2017-19)"] = met_d_pre

if len(eval_post_stays_D) > 0:
    eval_post_df_D = test_post.filter(
        pl.col("stay_id").is_in(eval_post_stays_D))
    print(f"--- Test-Post held-out ({len(eval_post_stays_D)} stays) ---")
    met_d_post, probs_d, labels_d = eval_on_split_ss(
        model_D, eval_post_df_D, criterion)
    results_D["Test-Post (2020-22)"] = met_d_post

# ── PRINT RESULTS ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RUN D RESULTS")
print("="*60)
for split_name, met in results_D.items():
    print(f"\n{split_name}:")
    for lbl in LABEL_COLS:
        auroc = met.get(f"{lbl}_auroc", float("nan"))
        auprc = met.get(f"{lbl}_auprc", float("nan"))
        brier = met.get(f"{lbl}_brier", float("nan"))
        n_pos = met.get(f"{lbl}_n_pos", 0)
        print(f"  {lbl:<30} AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
              f"Brier={brier:.4f}  (n_pos={n_pos})")

# ── SAVE RESULTS ───────────────────────────────────────────────────────────────
# Convert to JSON-serialisable format
def to_serialisable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, float):
            out[k] = None if (v != v) else round(v, 6)   # NaN → None
        elif isinstance(v, (np.float32, np.float64)):
            out[k] = None if (v != v) else float(round(v, 6))
        elif isinstance(v, (np.int32, np.int64, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out

serialisable_results_D = {
    split: to_serialisable(met) for split, met in results_D.items()
}
with open(SAVE_PATH / "run_d_results.json", "w") as f:
    json.dump(serialisable_results_D, f, indent=2)
print(f"\nResults saved → {SAVE_PATH / 'run_d_results.json'}")

torch.save({
    "source":       source_state_D,
    "run_d_adapted": model_D.state_dict(),
    "input_dim":    combined_input_dim,
    "hidden_dim":   HIDDEN_DIM,
    "n_targets":    len(LABEL_COLS),
    "train_stats":  train_stats,
}, SAVE_PATH / "run_d_model.pt")
print(f"Model saved → {SAVE_PATH / 'run_d_model.pt'}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD Run C + SAVE Run C + Run D together
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SAVING FULL ADAPT MODELS — Run C + Run D")
print("="*70)

# Load Run C from the temporary file we created in Script 1
try:
    run_c_state = torch.load(SAVE_PATH / "temp_run_c_weights.pt", map_location=device)
except FileNotFoundError:
    raise FileNotFoundError("Could not find temp_run_c_weights.pt. Make sure you ran Script 1 first!")

# Load two_stream just to grab the dimensions
two_ckpt = torch.load(SAVE_PATH / "two_stream_models.pt", map_location=device)

# Save C and D together
torch.save({
    "run_c": run_c_state,
    "run_d_source": source_state_D,
    "run_d_adapted": model_D.state_dict(),
    "combined_input_dim": combined_input_dim,
    "seq_dim": two_ckpt.get("seq_dim"),
    "treat_dim": two_ckpt.get("treat_dim"),
    "n_targets": len(LABEL_COLS),
    "train_stats": train_stats,
}, SAVE_PATH / "full_adapt_models.pt")

print(f"✅ Saved full_adapt_models.pt with Run C + Run D")
# ── COMPARISON TABLE (Run D vs Run A / B / C if those results are available) ──
# To compare all four runs, load the previously saved Run A/B/C results or
# re-evaluate them here using the two-stream model checkpoint.
# This block prints a standalone Run D summary for easy copy-paste into paper.

print("\n" + "="*60)
print("SUMMARY TABLE — Run D standalone")
print("="*60)
print(f"Architecture: Single-stream LSTM({combined_input_dim}→{HIDDEN_DIM})")
print(f"Adaptation:   LSTM frozen, head fine-tuned ({LR_ADAPT} LR)")
print(f"Adaptation performed: {adaptation_performed_D}")
print()

header = (f"  {'Split':<22} {'Label':<30} "
          f"{'AUROC':>8} {'AUPRC':>8} {'Brier':>8} {'n_pos':>6}")
print(header)
print("  " + "-" * (len(header) - 2))
for split_name, met in results_D.items():
    for lbl in LABEL_COLS:
        auroc = met.get(f"{lbl}_auroc", float("nan"))
        auprc = met.get(f"{lbl}_auprc", float("nan"))
        brier = met.get(f"{lbl}_brier", float("nan"))
        n_pos = met.get(f"{lbl}_n_pos", 0)
        def fmt(v): return f"{v:.4f}" if not np.isnan(v) else "  N/A "
        print(f"  {split_name:<22} {lbl:<30} "
              f"{fmt(auroc):>8} {fmt(auprc):>8} {fmt(brier):>8} {n_pos:>6}")

print("\n✅ Run D complete")
print()
print("Next step: add Run D mAUROC to Table 3 in the paper alongside Runs A/B/C.")
print("Expected interpretation:")
print("  If Run D << Run B/C  → two-stream decomposition adds value beyond")
print("                          selective freezing alone.")
print("  If Run D ≈ Run B/C   → last-layer freezing is sufficient; the")
print("                          structural prior argument needs qualifying.")