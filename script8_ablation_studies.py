"""
script8_ablation_studies.py — CORRECTED
══════════════════════════════════════════════════════════════════════
Fixes vs. original draft:
  1. SUBJECT-LEVEL split (not stay-level) for adapt_train/adapt_val/eval,
     mirroring script2 FIX 1 exactly — prevents the same patient's
     multiple stays leaking across train/val/eval at any split ratio.
  2. Pre-drift buffer sorted by [anchor_year_group, intime] (true
     chronological recency), mirroring script2 FIX 8 exactly — NOT
     sorted by stay_id, which is not time-ordered.
  3. Subject-level leakage guard: any subject_id in the replay buffer
     that also appears in eval_post_subjects is dropped, mirroring
     script2's leaking_subjects check exactly.
  4. eval set is held fixed at the SAME stays as the main pipeline's
     eval_post_stays (loaded from eval_split.json) for every split-ratio
     run, so split-ratio ablation only varies the train/val boundary
     within the *remaining* subjects, keeping the held-out evaluation
     partition identical to Table 2 / Table 3 across all ablation rows.
     This isolates the effect of split ratio without changing what is
     being evaluated on.

Produces ablation_results.json + two paper-ready markdown tables.
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
import polars as pl
from torch.utils.data import DataLoader

import copy
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import load_enriched_split, normalize, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import (
    TwoStreamModel, SEQ_LEN, BATCH_SIZE, LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
warnings.filterwarnings("ignore")

import os
BASE_PATH = Path(os.environ.get("DATA_DIR", "/kaggle/input/datasets/fatematamanna/allnew"))
s2        = Path("/kaggle/input/datasets/fatematamanna/ptfiles")
SAVE_PATH = Path(os.environ.get("OUT_DIR", "/kaggle/working"))

SEED = 42
np.random.seed(SEED)

# ── LOAD SOURCE CHECKPOINT ────────────────────────────────────────────────────

ckpt_path = SAVE_PATH / "two_stream_models.pt"
assert ckpt_path.exists(), (
    f"{ckpt_path} not found — run script2_two_stream_model.py first in this "
    f"session so the current-architecture checkpoint exists in /kaggle/working."
)
source_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
source_state = source_ckpt["source"]
seq_dim      = source_ckpt["seq_dim"]
treat_dim    = source_ckpt["treat_dim"]
n_targets    = source_ckpt["n_targets"]
train_stats  = source_ckpt["train_stats"]

# ── LOAD THE *SAME* EVAL SPLIT USED IN THE MAIN PIPELINE ─────────────────────
# This guarantees the held-out evaluation partition reported here is the
# identical 5,749-stay set used in Table 2 / Table 3 of the main paper.
with open(SAVE_PATH / "eval_split.json") as f:
    main_split = json.load(f)

eval_post_stays_fixed = set(map(int, main_split["eval_post_stays"]))

# ── TRAIN / EVAL HELPERS ──────────────────────────────────────────────────────
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
            metrics[f"{lbl}_auroc"]  = roc_auc_score(y_true, y_prob)
            metrics[f"{lbl}_auprc"]  = average_precision_score(y_true, y_prob)
            metrics[f"{lbl}_brier"]  = brier_score_loss(y_true, y_prob)
        else:
            metrics[f"{lbl}_auroc"]  = float("nan")
            metrics[f"{lbl}_auprc"]  = float("nan")
            metrics[f"{lbl}_brier"]  = float("nan")
        metrics[f"{lbl}_n_pos"] = n_pos
    return metrics, probs, labels


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
                  f"mAUROC={np.nanmean(aurocs):.4f} {'*' if wait == 0 else ''}")
        if wait >= patience:
            print(f"  {tag}Early stop at epoch {ep}"); break
    if best_state: model.load_state_dict(best_state)
    return model


# ── LOAD DATA ──────────────────────────────────────────────────────────────────
test_df = normalize(load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES),
                     train_stats)

test_pre  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")


# ── CORE EXPERIMENT — subject-level split + intime-sorted, leakage-guarded buffer ──
def run_adaptation_experiment(train_ratio, buffer_size, label_tag):
    print(f"\n--- Running Ablation: {label_tag} (Train={train_ratio}, Buffer={buffer_size}) ---")

    # 1. SUBJECT-LEVEL split — mirrors script2 FIX 1 exactly.
    #    Train/val are re-drawn at the requested ratio from the post-drift
    #    subject pool that EXCLUDES the fixed eval_post subjects, so the
    #    held-out evaluation partition never changes across ablation rows.
    eval_post_df_fixed = test_post.filter(pl.col("stay_id").is_in(eval_post_stays_fixed))
    eval_post_subjects_fixed = set(
        eval_post_df_fixed.filter(pl.col("hrs_from_admit") == 0)["subject_id"].unique().to_list()
    )

    remaining_post = test_post.filter(~pl.col("subject_id").is_in(eval_post_subjects_fixed))
    remaining_subjects = (remaining_post.filter(pl.col("hrs_from_admit") == 0)
                           ["subject_id"].unique().to_list())

    rng = np.random.RandomState(SEED)
    rng.shuffle(remaining_subjects)

    n_total_subj = len(remaining_subjects)
    n_train_subj = int(n_total_subj * train_ratio)
    n_val_subj   = int(n_total_subj * 0.10)  # val ratio held stable, as in main pipeline

    train_subjects = remaining_subjects[:n_train_subj]
    val_subjects   = remaining_subjects[n_train_subj:n_train_subj + n_val_subj]
    # any leftover remaining subjects beyond train+val are simply unused for
    # this ablation row (mirrors the fact that eval is fixed, not re-derived)

    train_stays = (remaining_post.filter(pl.col("subject_id").is_in(train_subjects))
                   ["stay_id"].unique().to_list())
    val_stays   = (remaining_post.filter(pl.col("subject_id").is_in(val_subjects))
                   ["stay_id"].unique().to_list())

    # Sanity: no subject leakage between train/val and the fixed eval set
    assert not (set(train_subjects) & eval_post_subjects_fixed), "subject leak: train↔eval"
    assert not (set(val_subjects) & eval_post_subjects_fixed),   "subject leak: val↔eval"

    print(f"  Subjects: train={n_train_subj} val={n_val_subj} "
          f"(eval fixed at {len(eval_post_subjects_fixed)} subjects / "
          f"{len(eval_post_stays_fixed)} stays)")

    # 2. Pre-drift replay buffer — mirrors script2 FIX 8 exactly:
    #    sorted by [anchor_year_group, intime], then leakage-guarded against
    #    the fixed eval_post subject set.
    buf_pre_df = pl.DataFrame()
    if buffer_size > 0 and test_pre.height > 0:
        pre_first_row = (test_pre.filter(pl.col("hrs_from_admit") == 0)
                          .sort(["anchor_year_group", "intime"]))
        pre_stays_sorted = pre_first_row["stay_id"].to_list()
        buf_stays_cand = (pre_stays_sorted[-buffer_size:]
                           if len(pre_stays_sorted) > buffer_size
                           else pre_stays_sorted)
        buf_pre_df = test_pre.filter(pl.col("stay_id").is_in(buf_stays_cand))

        buf_subjects = set(buf_pre_df["subject_id"].unique().to_list())
        leaking_subjects = buf_subjects & eval_post_subjects_fixed
        if leaking_subjects:
            buf_pre_df = buf_pre_df.filter(~pl.col("subject_id").is_in(list(leaking_subjects)))
            print(f"  ⚠ Removed {len(leaking_subjects)} subjects from pre-drift "
                  f"buffer (also in eval_post)")

    n_buf = (buf_pre_df.filter(pl.col("hrs_from_admit") == 0).height
             if buf_pre_df.height > 0 else 0)
    print(f"  Buffer: requested={buffer_size} actual_stays={n_buf}")

    # 3. Build dataloaders
    train_df_sub = (pl.concat([buf_pre_df, remaining_post.filter(pl.col("stay_id").is_in(train_stays))])
                     if buf_pre_df.height > 0
                     else remaining_post.filter(pl.col("stay_id").is_in(train_stays)))

    ds_train = ICUDataset(train_df_sub, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    ds_val   = ICUDataset(remaining_post.filter(pl.col("stay_id").is_in(val_stays)),
                           SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    ds_eval  = ICUDataset(eval_post_df_fixed, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)

    ldr_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    ldr_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False)
    ldr_eval  = DataLoader(ds_eval,  batch_size=BATCH_SIZE, shuffle=False)

    # 4. Reset to source state, configure for selective (Run B) adaptation
    exp_model = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    exp_model.load_state_dict(source_state)
    exp_model.unfreeze_adaptive()   # freeze physio, unfreeze treat + fusion — same as Run B

    ablation_patience = 15
    crit = FocalBCEWithLogitsLoss(pos_weight=compute_pos_weights(ds_train), gamma=1.0,
                                   label_smoothing=0.05)
    exp_model = train_model(exp_model, ldr_train, ldr_val, crit, LR_ADAPT,
                             ADAPT_EPOCHS, ablation_patience, f"{label_tag} ")
    

    # 6. Evaluate on the FIXED post-drift held-out set
    post_metrics, _, _ = evaluate(exp_model, ldr_eval, crit)

    # 7. Evaluate on full pre-drift test set (catastrophic forgetting check)
    pre_ds  = ICUDataset(test_pre, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    pre_ldr = DataLoader(pre_ds, batch_size=BATCH_SIZE, shuffle=False)
    pre_metrics, _, _ = evaluate(exp_model, pre_ldr, crit)

    return post_metrics, pre_metrics, n_buf


# ── RUN ABLATIONS ──────────────────────────────────────────────────────────────
ablation_results = {}

print("\nRunning Baseline (Split_30 + Buffer_500)...")
post_m, pre_m, n_buf = run_adaptation_experiment(train_ratio=0.30, buffer_size=500,
                                                   label_tag="Split_30_Baseline")
ablation_results["Split_30_Baseline"] = {
    "post_auroc": float(np.nanmean([post_m[f"{l}_auroc"] for l in LABEL_COLS])),
    "pre_auroc":  float(np.nanmean([pre_m[f"{l}_auroc"] for l in LABEL_COLS])),
    "post_per_label_auroc": {l: float(post_m[f"{l}_auroc"]) for l in LABEL_COLS},
    "actual_buffer_stays": n_buf,
}

# Experiment 1: Split Ratio Ablation (Buffer Fixed at 500)
for ratio in [0.20, 0.40]:
    tag = f"Split_{int(ratio * 100)}"
    post_m, pre_m, n_buf = run_adaptation_experiment(train_ratio=ratio, buffer_size=500, label_tag=tag)
    ablation_results[tag] = {
        "post_auroc": float(np.nanmean([post_m[f"{l}_auroc"] for l in LABEL_COLS])),
        "pre_auroc":  float(np.nanmean([pre_m[f"{l}_auroc"] for l in LABEL_COLS])),
        "post_per_label_auroc": {l: float(post_m[f"{l}_auroc"]) for l in LABEL_COLS},
        "actual_buffer_stays": n_buf,
    }

# Experiment 2: Buffer Size Ablation (Ratio Fixed at 0.30)
for buf in [0, 250]:
    tag = f"Buffer_{buf}"
    post_m, pre_m, n_buf = run_adaptation_experiment(train_ratio=0.30, buffer_size=buf, label_tag=tag)
    ablation_results[tag] = {
        "post_auroc": float(np.nanmean([post_m[f"{l}_auroc"] for l in LABEL_COLS])),
        "pre_auroc":  float(np.nanmean([pre_m[f"{l}_auroc"] for l in LABEL_COLS])),
        "post_per_label_auroc": {l: float(post_m[f"{l}_auroc"]) for l in LABEL_COLS},
        "actual_buffer_stays": n_buf,
    }

print("\nRaw Ablation Results:", json.dumps(ablation_results, indent=2))

# ── SAVE ────────────────────────────────────────────────────────────────────────
with open(SAVE_PATH / "ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)
print(f"\n✅ Saved → {SAVE_PATH / 'ablation_results.json'}")

# ── PAPER-READY TABLES ────────────────────────────────────────────────────────
print("\n--- TABLE A: Split Ratio Ablation (buffer fixed = 500) ---")
split_data = {
    "Split_20":           ablation_results.get("Split_20", {}),
    "Split_30 (Baseline)": ablation_results.get("Split_30_Baseline", {}),
    "Split_40":           ablation_results.get("Split_40", {}),
}
df_split = pd.DataFrame.from_dict(split_data, orient='index')[['post_auroc', 'pre_auroc']].round(4)
print(df_split.to_markdown())

print("\n--- TABLE B: Buffer Size Ablation (split fixed = 0.30) ---")
buffer_data = {
    "Buffer_0":             ablation_results.get("Buffer_0", {}),
    "Buffer_250":           ablation_results.get("Buffer_250", {}),
    "Buffer_500 (Baseline)": ablation_results.get("Split_30_Baseline", {}),
}
df_buffer = pd.DataFrame.from_dict(buffer_data, orient='index')[['post_auroc', 'pre_auroc']].round(4)
print(df_buffer.to_markdown())

print("\n--- Variance summary for paper text ---")
split_aurocs = [ablation_results[k]["post_auroc"] for k in
                ["Split_20", "Split_30_Baseline", "Split_40"] if k in ablation_results]
print(f"  Split ratio post-AUROC range: {min(split_aurocs):.4f} - {max(split_aurocs):.4f} "
      f"(spread = {max(split_aurocs) - min(split_aurocs):.4f})")

buf_post_aurocs = {k: ablation_results[k]["post_auroc"] for k in
                   ["Buffer_0", "Buffer_250", "Split_30_Baseline"] if k in ablation_results}
print(f"  Buffer post-AUROC: {buf_post_aurocs}")
buf_pre_aurocs = {k: ablation_results[k]["pre_auroc"] for k in
                  ["Buffer_0", "Buffer_250", "Split_30_Baseline"] if k in ablation_results}
print(f"  Buffer pre-AUROC (forgetting check): {buf_pre_aurocs}")

print("\n✅ script8_ablation_studies.py (corrected) complete")