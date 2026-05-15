#%%writefile utils/data_utils.py
import polars as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils.constants import BINARY_COLS
from pathlib import Path

SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 42, 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LR_INIT, LR_ADAPT = 1e-3, 3e-4
EPOCHS, ADAPT_EPOCHS = 50, 40
PATIENCE, ADAPT_PATIENCE = 8, 8
BUFFER_SIZE, PSI_THRESH = 500, 0.20
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]


def load_enriched_split(base_path: Path, split_name: str, seq_features: list, treat_features: list) -> pl.DataFrame:
    """Loads a parquet split and ensures all required columns and encodings exist."""
    df = pl.read_parquet(base_path / f"{split_name}_final_enriched.parquet")
    
    if "gender" in df.columns:
        df = df.with_columns((pl.col("gender") == "M").cast(pl.Float32).alias("gender_M"))
        
    for eth in ["WHITE","BLACK","HISPANIC","ASIAN"]:
        if "ethnicity" in df.columns:
            df = df.with_columns((pl.col("ethnicity") == eth).cast(pl.Float32).alias(f"eth_{eth}"))
            
    for c in seq_features + treat_features:
        if c not in df.columns:
            df = df.with_columns(pl.lit(0.0).cast(pl.Float32).alias(c))
            
    return df

def calculate_train_stats(train_df: pl.DataFrame, norm_cols: list) -> dict:
    """Calculates Z-score statistics exclusively from the training set."""
    train_stats = {}
    for c in norm_cols:
        if c in train_df.columns:
            vals = train_df[c].cast(pl.Float64)
            mu, sd = vals.mean(), vals.std()
            if sd is None or sd == 0 or sd != sd: sd = 1.0
            train_stats[c] = (mu if mu is not None else 0.0, sd)
    return train_stats
    
def normalize(df, stats):
    exprs = []
    for c, (mu, sd) in stats.items():
        if c in df.columns:
            if c.endswith("_mask") or c in BINARY_COLS:
                exprs.append(pl.col(c).cast(pl.Float32))
            else:
                exprs.append(((pl.col(c).cast(pl.Float64) - mu) / sd).cast(pl.Float32).alias(c))
    return df.with_columns(exprs)


# ── DATASET ────────────────────────────────────────────────────────────────────
class ICUDataset(Dataset):
    def __init__(self, df, seq_features, treat_features, label_cols, seq_len=6):
        self.seq_len, self.label_cols = seq_len, label_cols
        self.seq_cols   = [c for c in seq_features  if c in df.columns]
        self.treat_cols = [c for c in treat_features if c in df.columns]
        stays = df.sort(["stay_id","hrs_from_admit"])
        self.stay_ids = stays.select("stay_id").unique().sort("stay_id")["stay_id"].to_list()
        self.seq_data, self.treat_data, self.labels, self.groups = [], [], [], []
        for sid in self.stay_ids:
            s = stays.filter(pl.col("stay_id") == sid)
            seq = s.select(self.seq_cols).to_numpy().astype(np.float32)
            if seq.shape[0] < seq_len:
                seq = np.vstack([seq, np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)])
            else:
                seq = seq[:seq_len]
            self.seq_data.append(seq)
            self.treat_data.append(np.array(s.select(self.treat_cols).row(0), dtype=np.float32))
            self.labels.append(np.array(s.select(label_cols).row(0), dtype=np.float32))
            self.groups.append(s["anchor_year_group"][0] if "anchor_year_group" in s.columns else "UNK")
        self.seq_data   = np.stack(self.seq_data)
        self.treat_data = np.stack(self.treat_data)
        self.labels     = np.stack(self.labels)

    def __len__(self): return len(self.stay_ids)
    def __getitem__(self, idx):
        return (torch.from_numpy(self.seq_data[idx]),
                torch.from_numpy(self.treat_data[idx]),
                torch.from_numpy(self.labels[idx]))


class SingleStreamDataset(Dataset):
    """
    Concatenates treatment features (static, broadcast across time) with
    sequential physiological features so the LSTM sees all 105 features
    at every timestep.
    """
    def __init__(self, df, seq_features, treat_features, label_cols, seq_len=6):
        self.seq_len     = seq_len
        self.label_cols  = label_cols
        self.seq_cols    = [c for c in seq_features   if c in df.columns]
        self.treat_cols  = [c for c in treat_features if c in df.columns]

        stays = df.sort(["stay_id", "hrs_from_admit"])
        self.stay_ids = (stays.select("stay_id").unique()
                              .sort("stay_id")["stay_id"].to_list())

        self.combined_data = []   # shape: (n_stays, seq_len, seq_dim + treat_dim)
        self.labels        = []
        self.groups        = []

        for sid in self.stay_ids:
            s = stays.filter(pl.col("stay_id") == sid)

            # Physiological sequence: (seq_len, seq_dim)
            seq = s.select(self.seq_cols).to_numpy().astype(np.float32)
            if seq.shape[0] < seq_len:
                pad = np.zeros((seq_len - seq.shape[0], seq.shape[1]), dtype=np.float32)
                seq = np.vstack([seq, pad])
            else:
                seq = seq[:seq_len]

            # Treatment vector (static snapshot from first row): (treat_dim,)
            treat_vec = np.array(s.select(self.treat_cols).row(0), dtype=np.float32)

            # Broadcast treatment across time: (seq_len, treat_dim)
            treat_tiled = np.tile(treat_vec, (seq_len, 1))

            # Concatenate: (seq_len, seq_dim + treat_dim)
            combined = np.concatenate([seq, treat_tiled], axis=1)

            self.combined_data.append(combined)
            self.labels.append(np.array(s.select(label_cols).row(0), dtype=np.float32))
            self.groups.append(s["anchor_year_group"][0]
                               if "anchor_year_group" in s.columns else "UNK")

        self.combined_data = np.stack(self.combined_data)  # (N, seq_len, seq+treat)
        self.labels        = np.stack(self.labels)          # (N, n_targets)

    def __len__(self): return len(self.stay_ids)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.combined_data[idx]),
                torch.from_numpy(self.labels[idx]))
