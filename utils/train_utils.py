%%writefile utils/train_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    """Tracks validation loss and triggers early stopping."""
    def __init__(self, patience: int = 8, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=1.5, label_smoothing=0.02):
        super().__init__()
        self.gamma, self.label_smoothing, self.pos_weight = gamma, label_smoothing, pos_weight
    def forward(self, logits, targets):
        ts = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, ts, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        return ((1 - pt) ** self.gamma * bce).mean()


def compute_pos_weights(dataset, max_weight=20.0):
    labels  = dataset.labels
    pos     = labels.sum(axis=0)
    neg     = labels.shape[0] - pos
    weights = neg / (pos + 1e-6)
    return torch.tensor(np.clip(weights, 1.0, max_weight),
                        dtype=torch.float32).to(device)
