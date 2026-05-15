#%%writefile models/architectures.py
import torch
import torch.nn as nn
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


# ── MODEL ──────────────────────────────────────────────────────────────────────
class PhysiologyStream(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.norm(h[-1])

class TreatmentStream(nn.Module):
    def __init__(self, input_dim, output_dim=32, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, output_dim), nn.ReLU(), nn.LayerNorm(output_dim))
    def forward(self, x): return self.mlp(x)

class FusionHead(nn.Module):
    def __init__(self, physio_dim=64, treat_dim=32, n_targets=4, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(physio_dim + treat_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout * 0.75),
            nn.Linear(32, n_targets))
    def forward(self, p, t): return self.head(torch.cat([p, t], dim=1))

class TwoStreamModel(nn.Module):
    def __init__(self, seq_input_dim, treat_input_dim, n_targets=4):
        super().__init__()
        self.physio = PhysiologyStream(seq_input_dim, HIDDEN_DIM, LSTM_LAYERS, DROPOUT)
        self.treat  = TreatmentStream(treat_input_dim, TREAT_DIM, DROPOUT)
        self.fusion = FusionHead(HIDDEN_DIM, TREAT_DIM, n_targets, DROPOUT)

    def forward(self, x_seq, x_treat):
        return self.fusion(self.physio(x_seq), self.treat(x_treat))

    def freeze_physio(self):
        for p in self.physio.parameters(): p.requires_grad = False

    def freeze_all(self):
        for p in self.parameters(): p.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze every parameter — used by Run C."""
        for p in self.parameters(): p.requires_grad = True

    def unfreeze_adaptive(self):
        """Freeze physio, unfreeze treat+fusion — used by Run B."""
        self.freeze_all()
        for p in self.treat.parameters():  p.requires_grad = True
        for p in self.fusion.parameters(): p.requires_grad = True


class SingleStreamModel(nn.Module):
    """
    Monolithic LSTM on all (seq + treatment) features, followed by a
    two-layer dense head.

    Selective adaptation strategy (Run D):
      - self.lstm    → FROZEN during adaptation
      - self.head    → TRAINABLE during adaptation (last-layer fine-tune)

    This mirrors the freeze strategy used in Run B but applied to a
    monolithic model instead of a structurally decomposed one.
    """
    def __init__(self, input_dim, hidden_dim=64, n_layers=2,
                 dropout=0.3, n_targets=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Late-fusion head — these layers are updated during adaptation
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(32, n_targets)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h, _) = self.lstm(x)
        h_last = self.lstm_norm(h[-1])   # (batch, hidden_dim)
        return self.head(h_last)          # (batch, n_targets)

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze_head_only(self):
        """
        Freeze LSTM, unfreeze only the dense head.
        This is the Run D adaptation strategy — selective last-layer fine-tuning
        on a monolithic model, analogous to Run B's treatment-stream freeze
        but without structural decomposition.
        """
        self.freeze_all()
        for p in self.head.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

