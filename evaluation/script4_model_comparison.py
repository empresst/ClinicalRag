#%%writefile evaluation/script4_model_comparison.py
import torch
import json, warnings, copy
from pathlib import Path
import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from torch.utils.data import DataLoader
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from utils.data_utils import ICUDataset, SingleStreamDataset
from models.architectures import PhysiologyStream, TreatmentStream, FusionHead, SingleStreamModel, TwoStreamModel

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("\n" + "="*140)
print("CLEAN 6-MODEL COMPARISON — Pre & Post")
print("="*140)

SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 42, 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LR_INIT, LR_ADAPT = 1e-3, 3e-4
EPOCHS, ADAPT_EPOCHS = 50, 40
PATIENCE, ADAPT_PATIENCE = 8, 8
BUFFER_SIZE, PSI_THRESH = 500, 0.20
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")

seq_dim = len(SEQ_FEATURES)
treat_dim = len(TREATMENT_FEATURES)
combined_input_dim = seq_dim + treat_dim

# --- Load Data First! ---
print("Loading Data...")
train_df = load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES)
test_df  = load_enriched_split(BASE_PATH, "test",  SEQ_FEATURES, TREATMENT_FEATURES)

# Calculate stats from train, apply to test
print("Normalizing Data...")
all_norm_cols = list(set(SEQ_FEATURES + TREATMENT_FEATURES))
train_stats = calculate_train_stats(train_df, all_norm_cols)
test_df = normalize(test_df, train_stats)

# Now it is safe to split!
test_pre_df  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post_df = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

print(f"Pre rows : {len(test_pre_df):,}")
print(f"Post rows: {len(test_post_df):,}\n")

# --- Load Checkpoints ---
print("Loading Checkpoints...")
# Make sure these filenames match exactly what you saved in script2 and script3!
two_ckpt = torch.load("two_stream_models.pt", map_location=device)
full_ckpt = torch.load("full_adapt_models.pt", map_location=device)

# Models
model_A = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_B = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_C = TwoStreamModel(seq_dim, treat_dim, len(LABEL_COLS)).to(device)
model_D = SingleStreamModel(combined_input_dim, HIDDEN_DIM, LSTM_LAYERS, DROPOUT, len(LABEL_COLS)).to(device)

# Data
test_pre_df  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post_df = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

print(f"Pre rows : {len(test_pre_df):,}")
print(f"Post rows: {len(test_post_df):,}\n")

@torch.no_grad()
def get_predictions(model, df, is_single_stream=False):
    model.eval()
    all_logits = []
    
    if is_single_stream:
        ds = SingleStreamDataset(df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        for x_comb, y in loader:
            logits = model(x_comb.to(device))
            all_logits.append(logits.cpu())
    else:
        ds = ICUDataset(df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        for x_seq, x_treat, y in loader:
            logits = model(x_seq.to(device), x_treat.to(device))
            all_logits.append(logits.cpu())
            
    logits = torch.cat(all_logits).numpy()
    probs = 1 / (1 + np.exp(-logits))
    return probs

def get_metrics(probs, labels):
    results = {}
    for i, lbl in enumerate(LABEL_COLS):
        y = labels[:, i]
        p = probs[:, i]
        if y.sum() > 0 and y.sum() < len(y):
            auroc = roc_auc_score(y, p)
            auprc = average_precision_score(y, p)
        else:
            auroc = auprc = float("nan")
        results[lbl] = {"auroc": auroc, "auprc": auprc}
    return results

print("Computing all predictions...")

# ---------- Labels (extracted once, consistently) ----------
labels_pre  = ICUDataset(test_pre_df,  SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN).labels
labels_post = ICUDataset(test_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN).labels

# ---------- Run A ----------
model_A.load_state_dict(two_ckpt["run_a"])
probs_a_pre  = get_predictions(model_A, test_pre_df,  is_single_stream=False)
probs_a_post = get_predictions(model_A, test_post_df, is_single_stream=False)

# ---------- Run B ----------
model_B.load_state_dict(two_ckpt["run_b"])
probs_b_pre  = get_predictions(model_B, test_pre_df,  is_single_stream=False)
probs_b_post = get_predictions(model_B, test_post_df, is_single_stream=False)

# ---------- Run C ----------
model_C.load_state_dict(full_ckpt["run_c"])
probs_c_pre  = get_predictions(model_C, test_pre_df,  is_single_stream=False)
probs_c_post = get_predictions(model_C, test_post_df, is_single_stream=False)

# ---------- Run D ----------
# Pre: use source (pre-adaptation) weights evaluated on pre-drift data
model_D.load_state_dict(full_ckpt["run_d_source"])
probs_d_pre  = get_predictions(model_D, test_pre_df,  is_single_stream=True)

# Post: use adapted weights evaluated on post-drift data
model_D.load_state_dict(full_ckpt["run_d_adapted"])
probs_d_post = get_predictions(model_D, test_post_df, is_single_stream=True)

# ---------- Collect & print ----------
metrics = {
    "Run A (Pre)":  get_metrics(probs_a_pre,  labels_pre),
    "Run A (Post)": get_metrics(probs_a_post, labels_post),
    "Run B (Pre)":  get_metrics(probs_b_pre,  labels_pre),
    "Run B (Post)": get_metrics(probs_b_post, labels_post),
    "Run C (Pre)":  get_metrics(probs_c_pre,  labels_pre),
    "Run C (Post)": get_metrics(probs_c_post, labels_post),
    "Run D (Pre)":  get_metrics(probs_d_pre,  labels_pre),
    "Run D (Post)": get_metrics(probs_d_post, labels_post),
}

print(f"\n{'Model':<20} {'Vaso AUROC':>12} {'Intub AUROC':>14} {'Shock AUROC':>14} | "
      f"{'Vaso AUPRC':>12} {'Intub AUPRC':>14} {'Shock AUPRC':>14}")
print("-" * 145)

for name, m in metrics.items():
    v = m['label_vasopressor']
    i = m['label_intubation']
    s = m['label_septic_shock']
    print(f"{name:<20} {v['auroc']:>12.4f} {i['auroc']:>14.4f} {s['auroc']:>14.4f} | "
          f"{v['auprc']:>12.4f} {i['auprc']:>14.4f} {s['auprc']:>14.4f}")

print("\n✅ Done.")