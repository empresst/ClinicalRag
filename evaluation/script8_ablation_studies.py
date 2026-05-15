#%%writefile evaluation/script8_ablation_studies.py
import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
import polars as pl
from torch.utils.data import Dataset, DataLoader

import copy
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from utils.train_utils import EarlyStopping   # Make sure this exists in your utils

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, SingleStreamDataset, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import PhysiologyStream, TreatmentStream, FusionHead, SingleStreamModel, TwoStreamModel, SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM, BATCH_SIZE, LSTM_LAYERS, DROPOUT, LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
warnings.filterwarnings("ignore")
BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")
s2       = Path("/kaggle/input/datasets/fatematamanna/ptfiles")
SAVE_PATH = Path("/kaggle/working")

source_ckpt = torch.load("/kaggle/input/datasets/fatematamanna/ptfiles/two_stream_models (3).pt", map_location=device, weights_only=False)
source_state = source_ckpt["source"] 

seq_dim = source_ckpt["seq_dim"]
treat_dim = source_ckpt["treat_dim"]
n_targets = source_ckpt["n_targets"]
train_stats = source_ckpt["train_stats"]

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


test_df = normalize(load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES), 
                    train_stats)

test_pre  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

def run_adaptation_experiment(train_ratio, buffer_size, label_tag):
    print(f"\n--- Running Ablation: {label_tag} (Train={train_ratio}, Buffer={buffer_size}) ---")
    
    # 1. Setup Data
    post_stays = test_post.filter(pl.col("hrs_from_admit") == 0).sort("intime")["stay_id"].to_list()
    n_total = len(post_stays)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * 0.10) # Keep Val stable at 10%

    train_ids = post_stays[:n_train]
    val_ids = post_stays[n_train:n_train + n_val]
    eval_ids = post_stays[n_train + n_val:]

    # Buffer logic
    pre_stays = test_pre.select("stay_id").unique().sort("stay_id")["stay_id"].to_list()
    buf_stays = pre_stays[-buffer_size:] if buffer_size > 0 else []
    
    # Build Dataloaders
    train_df_sub = pl.concat([test_pre.filter(pl.col("stay_id").is_in(buf_stays)),
                             test_post.filter(pl.col("stay_id").is_in(train_ids))]) if buf_stays else test_post.filter(pl.col("stay_id").is_in(train_ids))
    
    ds_train = ICUDataset(train_df_sub, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    ds_val = ICUDataset(test_post.filter(pl.col("stay_id").is_in(val_ids)), SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    ds_eval = ICUDataset(test_post.filter(pl.col("stay_id").is_in(eval_ids)), SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    
    ldr_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    ldr_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)
    ldr_eval = DataLoader(ds_eval, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Reset Model B to Source State
    exp_model = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
    exp_model.load_state_dict(source_state)
    exp_model.unfreeze_adaptive()

    # 3. Adapt
    crit = FocalBCEWithLogitsLoss(pos_weight=compute_pos_weights(ds_train), gamma=1.0)
    exp_model = train_model(exp_model, ldr_train, ldr_val, crit, LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE, f"{label_tag} ")

    # 4. Evaluate on Post-Drift
    metrics, _, _ = evaluate(exp_model, ldr_eval, crit)
    
    # 5. Evaluate on PRE-Drift (To check Catastrophic Forgetting)
    pre_ds = ICUDataset(test_pre, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    pre_ldr = DataLoader(pre_ds, batch_size=BATCH_SIZE, shuffle=False)
    pre_metrics, _, _ = evaluate(exp_model, pre_ldr, crit)

    return metrics, pre_metrics


ablation_results = {}

# --- Experiment 1: Split Ratio Ablation (Buffer Fixed at 500) ---
for ratio in [0.20, 0.40]:
    tag = f"Split_{int(ratio*100)}"
    post_m, pre_m = run_adaptation_experiment(train_ratio=ratio, buffer_size=500, label_tag=tag)
    ablation_results[tag] = {"post_auroc": np.mean([post_m[f"{l}_auroc"] for l in LABEL_COLS]),
                             "pre_auroc": np.mean([pre_m[f"{l}_auroc"] for l in LABEL_COLS])}

# --- Experiment 2: Buffer Size Ablation (Ratio Fixed at 0.30) ---
for buf in [0, 250]:
    tag = f"Buffer_{buf}"
    post_m, pre_m = run_adaptation_experiment(train_ratio=0.30, buffer_size=buf, label_tag=tag)
    ablation_results[tag] = {"post_auroc": np.mean([post_m[f"{l}_auroc"] for l in LABEL_COLS]),
                             "pre_auroc": np.mean([pre_m[f"{l}_auroc"] for l in LABEL_COLS])}

import pandas as pd

print("Raw Ablation Results:", ablation_results)

print("\nRunning Baseline (Split_30 + Buffer_500)...")
post_m, pre_m = run_adaptation_experiment(train_ratio=0.30, buffer_size=500, label_tag="Split_30_Baseline")

ablation_results["Split_30_Baseline"] = {
    "post_auroc": np.mean([post_m[f"{l}_auroc"] for l in LABEL_COLS]),
    "pre_auroc": np.mean([pre_m[f"{l}_auroc"] for l in LABEL_COLS])
}

print("\n--- TABLE A: Split Ratio Ablation ---")
split_data = {
    "Split_20": ablation_results.get("Split_20", {}),
    "Split_30 (Baseline)": ablation_results.get("Split_30_Baseline", {}),
    "Split_40": ablation_results.get("Split_40", {})
}
df_split = pd.DataFrame.from_dict(split_data, orient='index')[['post_auroc', 'pre_auroc']].round(4)
print(df_split.to_markdown())

print("\n--- TABLE B: Buffer Size Ablation ---")
buffer_data = {
    "Buffer_0": ablation_results.get("Buffer_0", {}),
    "Buffer_250": ablation_results.get("Buffer_250", {}),
    "Buffer_500 (Baseline)": ablation_results.get("Split_30_Baseline", {})   # reusing baseline
}
df_buffer = pd.DataFrame.from_dict(buffer_data, orient='index')[['post_auroc', 'pre_auroc']].round(4)
print(df_buffer.to_markdown())
