#%%writefile models/script2_two_stream_model.py
"""
script2_two_stream_model_v4.py (from three_stream)
═════════════════════════════════
Three-run clinical AI — Run A vs Run B vs Run C.

Run A: Fully frozen source model (no adaptation).
Run B: Two-stream, LSTM frozen, treatment+fusion trainable during adaptation.
Run C: Two-stream, NOTHING frozen — all layers trainable during adaptation.
       This lets the physiology stream also adjust to the post-drift distribution.

Key fix carried from v3:
  - Adaptation uses post-drift validation for early stopping, not the pre-drift val set.
  - Separate adapt-train / adapt-val / held-out-eval split from post-drift data.
  - All three runs evaluated ONLY on the same held-out post-drift set.
  - Pre-drift: Run A, B, C all use identical source weights (honest baseline).
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

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
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


# ── DATA LOADING ───────────────────────────────────────────────────────────────
print("Loading data...")
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
# ── DATASET ────────────────────────────────────────────────────────────────────


print("Building datasets...")
train_ds = ICUDataset(train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
val_ds   = ICUDataset(val_df,   SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
test_ds  = ICUDataset(test_df,  SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Seq features: {len(train_ds.seq_cols)} | Treat features: {len(train_ds.treat_cols)}")

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

# ── PHASE 1: TRAIN SOURCE ─────────────────────────────────────────────────────
print("\n" + "="*60 + "\nPHASE 1: Training source model on 2014-2016\n" + "="*60)

model = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model = train_model(model, train_loader, val_loader, criterion, LR_INIT, EPOCHS, PATIENCE, "SRC ")

val_met, _, _ = evaluate(model, val_loader, criterion)
print(f"\nSource on VAL:")
for lbl in LABEL_COLS:
    print(f"  {lbl}: AUROC={val_met.get(f'{lbl}_auroc',0):.4f} AUPRC={val_met.get(f'{lbl}_auprc',0):.4f}"
          f" (n_pos={val_met.get(f'{lbl}_n_pos',0)})")

source_state = copy.deepcopy(model.state_dict())

# ── INITIALISE ALL THREE RUNS ─────────────────────────────────────────────────
# Run A: frozen forever.
# Run B: LSTM frozen, treat+fusion trainable.
# Run C: nothing frozen — all layers trainable.
model_A = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_A.load_state_dict(source_state); model_A.freeze_all()

model_B = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_B.load_state_dict(source_state); model_B.freeze_physio()

model_C = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_C.load_state_dict(source_state)
# No freezing for C yet — adaptation will call unfreeze_all()

print("\nRun A: fully frozen")
print("Run B: LSTM frozen, treat+fusion trainable")
print("Run C: all layers trainable (full fine-tune on post-drift data)")

# ── DRIFT DETECTION ───────────────────────────────────────────────────────────
def compute_psi(ref, cur, bins=10):
    if len(cur) < 10: return 0.0
    mn, mx = min(ref.min(), cur.min()) - 1e-6, max(ref.max(), cur.max()) + 1e-6
    edges = np.linspace(mn, mx, bins + 1)
    ref_h, _ = np.histogram(ref, bins=edges)
    cur_h, _ = np.histogram(cur, bins=edges)
    ref_p = (ref_h + 1) / (ref_h.sum() + bins)
    cur_p = (cur_h + 1) / (cur_h.sum() + bins)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

TREAT_ONLY = [c for c in TREATMENT_FEATURES if c not in BINARY_COLS and c != "age"]
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

# ── PHASE 2: DRIFT DETECTION ──────────────────────────────────────────────────
print("\n" + "="*60 + "\nPHASE 2: Drift detection\n" + "="*60)

test_pre  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")
print(f"Pre-drift:  {test_pre['stay_id'].n_unique()} stays")
print(f"Post-drift: {test_post['stay_id'].n_unique()} stays")

def check_drift(chunk_df, label):
    psi_vals = {}
    for c, ref in train_treat_ref.items():
        if c in chunk_df.columns:
            cur = chunk_df.filter(pl.col("hrs_from_admit") == 0)[c].to_numpy().astype(float)
            psi_vals[c] = compute_psi(ref, cur)
    binary_shifts = {}
    for c, ref_rate in train_binary_ref.items():
        if c in chunk_df.columns:
            cur_rate = chunk_df.filter(pl.col("hrs_from_admit") == 0)[c].to_numpy().astype(float).mean()
            binary_shifts[c] = abs(cur_rate - ref_rate)
    mean_psi = np.mean(list(psi_vals.values())) if psi_vals else 0
    drifted_cont = [k for k, v in psi_vals.items() if v > PSI_THRESH]
    drifted_bin  = [k for k, v in binary_shifts.items() if v > 0.05]
    print(f"  {label}: mean_PSI={mean_psi:.4f} | drifted: {len(drifted_cont)} cont, {len(drifted_bin)} bin")
    for d in (drifted_cont + drifted_bin)[:8]:
        if d in psi_vals:        print(f"    ⚠ {d}: PSI={psi_vals[d]:.4f}")
        elif d in binary_shifts: print(f"    ⚠ {d}: Δrate={binary_shifts[d]:.4f}")
    return drifted_cont + drifted_bin

drifted_pre  = check_drift(test_pre,  "Pre-drift ")
drifted_post = check_drift(test_post, "Post-drift")

# ── PHASE 2b: BUILD SHARED ADAPTATION LOADERS ────────────────────────────────
# The same adapt-train / adapt-val / held-out split is shared by Run B and Run C
# so the comparison is perfectly fair.
adaptation_performed = False
eval_post_stays = []

if len(drifted_post) > 0 and test_post["stay_id"].n_unique() > 50:
    print(f"\n--- Drift detected ({len(drifted_post)} features)! Building adaptation data ---")

    post_stays    = test_post.filter(pl.col("hrs_from_admit") == 0).sort("intime")["stay_id"].to_list()
    n_total_post  = len(post_stays)

    # 30% adapt-train | 10% adapt-val | 60% held-out eval
    n_adapt_train = int(n_total_post * 0.30)
    n_adapt_val   = int(n_total_post * 0.10)

    adapt_train_stays = post_stays[:n_adapt_train]
    adapt_val_stays   = post_stays[n_adapt_train:n_adapt_train + n_adapt_val]
    eval_post_stays   = post_stays[n_adapt_train + n_adapt_val:]

    print(f"  Post-drift split: adapt-train={len(adapt_train_stays)} "
          f"adapt-val={len(adapt_val_stays)} held-out={len(eval_post_stays)}")

    # Pre-drift buffer for stability (prevents catastrophic forgetting in Run C)
    pre_stays     = test_pre.select("stay_id").unique().sort("stay_id")["stay_id"].to_list()
    buf_pre_stays = pre_stays[-BUFFER_SIZE:] if len(pre_stays) > BUFFER_SIZE else pre_stays
    buf_pre_df    = test_pre.filter(pl.col("stay_id").is_in(buf_pre_stays))

    adapt_train_df = test_post.filter(pl.col("stay_id").is_in(adapt_train_stays))
    adapt_val_df   = test_post.filter(pl.col("stay_id").is_in(adapt_val_stays))

    # Combine: pre-drift buffer + post-drift adapt-train (shared by B and C)
    combined_train_df = pl.concat([buf_pre_df, adapt_train_df])

    adapt_train_ds = ICUDataset(combined_train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    adapt_val_ds   = ICUDataset(adapt_val_df,      SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)

    adapt_train_loader = DataLoader(adapt_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    adapt_val_loader   = DataLoader(adapt_val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Adapt train: {len(adapt_train_ds)} ({buf_pre_df['stay_id'].n_unique()} pre + "
          f"{adapt_train_df['stay_id'].n_unique()} post)")
    print(f"  Adapt val:   {len(adapt_val_ds)} (post-drift, for early stopping)")

    adapt_pos_weights = compute_pos_weights(adapt_train_ds, max_weight=15.0)
    adapt_criterion   = FocalBCEWithLogitsLoss(
        pos_weight=adapt_pos_weights, gamma=1.0, label_smoothing=0.05)

    # ── Adapt Run B (LSTM frozen) ─────────────────────────────────────────────
    print("\n--- Adapting Run B (physio frozen) ---")
    model_B.unfreeze_adaptive()
    model_B = train_model(model_B, adapt_train_loader, adapt_val_loader, adapt_criterion,
                          LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE, "B-ADAPT ")
    print("  ✅ Run B adapted")

    # ── Adapt Run C (fully unfrozen) ──────────────────────────────────────────
    print("\n--- Adapting Run C (all layers trainable) ---")
    model_C.unfreeze_all()
    # Use a smaller LR for C to reduce the risk of catastrophic forgetting of
    # general physiology patterns while still allowing full adaptation.
    LR_ADAPT_C = LR_ADAPT * 0.5
    model_C = train_model(model_C, adapt_train_loader, adapt_val_loader, adapt_criterion,
                          LR_ADAPT_C, ADAPT_EPOCHS, ADAPT_PATIENCE, "C-ADAPT ")
    print("  ✅ Run C adapted")

    adaptation_performed = True

else:
    print("\nNo significant drift — Run B and Run C unchanged")
    post_stays      = test_post.filter(pl.col("hrs_from_admit") == 0).sort("intime")["stay_id"].to_list()
    eval_post_stays = post_stays

# ── PHASE 3: FINAL COMPARISON ─────────────────────────────────────────────────
print("\n" + "="*60 + "\nPHASE 3: Run A vs Run B vs Run C\n" + "="*60)

# For pre-drift splits, all runs used the SAME source weights —
# use model_B_pre (source weights) for B and C so the comparison is honest.
model_B_pre = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_B_pre.load_state_dict(source_state); model_B_pre.freeze_all()

model_C_pre = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_C_pre.load_state_dict(source_state); model_C_pre.freeze_all()

def eval_on_split(model, df, crit):
    ds     = ICUDataset(df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    return evaluate(model, loader, crit)

results = {}

# Val — all three use source weights
print("\n--- Val (source weights, pre-adaptation — should be identical) ---")
met_a_val,   _, _ = eval_on_split(model_A,     val_df, criterion)
met_b_val,   _, _ = eval_on_split(model_B_pre, val_df, criterion)
met_c_val,   _, _ = eval_on_split(model_C_pre, val_df, criterion)
results["Val (source)"] = {"A": met_a_val, "B": met_b_val, "C": met_c_val}

# Pre-drift — all three use source weights
print("--- Test-Pre (source weights, pre-adaptation — should be identical) ---")
met_a_pre, _, _ = eval_on_split(model_A,     test_pre, criterion)
met_b_pre, _, _ = eval_on_split(model_B_pre, test_pre, criterion)
met_c_pre, _, _ = eval_on_split(model_C_pre, test_pre, criterion)
results["Test-Pre (2017-19)"] = {"A": met_a_pre, "B": met_b_pre, "C": met_c_pre}

# Post-drift — A frozen, B partially adapted, C fully adapted
if len(eval_post_stays) > 0:
    eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))
    print(f"--- Test-Post held-out ({len(eval_post_stays)} stays) "
          "— A frozen | B partial adapt | C full adapt ---")
    met_a_post, probs_a, labels_a = eval_on_split(model_A, eval_post_df, criterion)
    met_b_post, probs_b, labels_b = eval_on_split(model_B, eval_post_df, criterion)
    met_c_post, probs_c, labels_c = eval_on_split(model_C, eval_post_df, criterion)
    results["Test-Post (2020-22)"] = {"A": met_a_post, "B": met_b_post, "C": met_c_post}

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

            # Drift line
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
        "Pre-drift: identical source weights | Post-drift: adaptation diverges",
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

print("\n" + "="*70)
print("SAVING TWO-STREAM MODELS — ONLY Run A + Run B")
print("="*70)

model_A.eval()
model_B.eval()
model_C.eval()

# 1. Save A and B to the main file exactly as you wanted
torch.save({
    "source": source_state,
    "run_a": model_A.state_dict(),
    "run_b": model_B.state_dict(),
    "seq_dim": seq_dim,
    "treat_dim": treat_dim,
    "n_targets": len(LABEL_COLS),
    "train_stats": train_stats,
    "config": {"adaptation_performed": adaptation_performed}
}, SAVE_PATH / "two_stream_models.pt")

# 2. Save Run C to a temporary file so it survives until the next script
torch.save(model_C.state_dict(), SAVE_PATH / "temp_run_c_weights.pt")

print(f"✅ Saved two_stream_models.pt with ONLY Run A + Run B")
print(f"✅ Saved temp_run_c_weights.pt (holding Run C for later)")

# ── SUMMARY ────────────────────────────────────────────────────────────────────
print("\n" + "="*60 + "\nSUMMARY\n" + "="*60)
print(f"Architecture: LSTM({seq_dim}→{HIDDEN_DIM}) + MLP({treat_dim}→{TREAT_DIM}) → Fusion → {len(LABEL_COLS)}")
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

if "Test-Post (2020-22)" in results:
    r = results["Test-Post (2020-22)"]
    valid_lbls = [l for l in LABEL_COLS
                  if not np.isnan(r["A"].get(f"{l}_auroc", float("nan")))]
    b_wins = sum(1 for l in valid_lbls
                 if r["B"].get(f"{l}_auroc", 0) > r["A"].get(f"{l}_auroc", 0))
    c_wins = sum(1 for l in valid_lbls
                 if r["C"].get(f"{l}_auroc", 0) > r["A"].get(f"{l}_auroc", 0))
    b_vs_c = sum(1 for l in valid_lbls
                 if r["B"].get(f"{l}_auroc", 0) > r["C"].get(f"{l}_auroc", 0))
    n_valid = len(valid_lbls)
    print(f"\n  Post-drift label wins (AUROC vs Run A):")
    print(f"    Run B > Run A: {b_wins}/{n_valid}")
    print(f"    Run C > Run A: {c_wins}/{n_valid}")
    print(f"    Run B > Run C: {b_vs_c}/{n_valid}  "
          f"(Run C > Run B: {n_valid - b_vs_c}/{n_valid})")

print("\n✅ Complete")