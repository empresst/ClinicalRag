"""
script20_multiseed_adaptation.py
════════════════════════════════
Re-runs ADAPTATION ONLY across all five seeds for Run B and Run C, so that the
attribution / retrieval contrasts (Tables 7-9) can be reported with the same
across-seed spread that Table 5 already gives for AUROC.

Design constraints (fidelity over convenience):
  * The SOURCE model is LOADED from two_stream_models.pt, never retrained.
    script2 was run on Kaggle (BASE_PATH=/kaggle/input/...), so a local CPU
    retrain would not reproduce it bitwise and every published number would
    shift. Loading guarantees comparability with Tables 3-9.
  * train_stats are LOADED from the same checkpoint, so normalisation is
    identical to the published run rather than recomputed.
  * Loss, optimiser, dataset and training loop are IMPORTED from utils/ or
    copied verbatim from script2 (train_epoch / evaluate / train_model),
    not reimplemented from the paper's prose.
  * The era split comes from eval_split.json, which regen_eval_split.py already
    verified reproduces script2:743-862 exactly (pre=30641, post=9602,
    eval=5749, eval_subj=4752).

Validation gate: seed 2024 Run B and seed 123 Run C must land near the
published 0.8947 / 0.8779 vasopressor AUROC. Exact equality is NOT expected
(Kaggle GPU vs local CPU); a gap beyond ~0.005 means something is wrong and the
other seeds should not be trusted.

Outputs  multiseed_adapted_weights.pt   (10 state_dicts + metadata)
         multiseed_adaptation.json      (per-seed metrics)
"""
import json, copy, sys, time, warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent))
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
from utils.data_utils import load_enriched_split, normalize, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import TwoStreamModel

warnings.filterwarnings("ignore")
device = torch.device("cpu")

import os
BASE_PATH = Path(os.environ.get("DATA_DIR", "/home/tamanna/Downloads/paper"))
SAVE_PATH = Path(os.environ.get("OUT_DIR", "/home/tamanna/Documents/try3"))

# script2:74-79 — verbatim
SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 42, 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LR_INIT, LR_ADAPT = 1e-3, 3e-4
EPOCHS, ADAPT_EPOCHS = 50, 40
PATIENCE, ADAPT_PATIENCE = 8, 8
BUFFER_SIZE = 500
SEEDS = [42, 123, 7, 2024, 99]

LEAKAGE_TIMING_FEATS = ["time_to_first_abx_order_hrs"]
SENTINEL_NO_EARLY_EVENT = float(SEQ_LEN + 1)

PUBLISHED = {("B", 2024): 0.8947, ("C", 123): 0.8779}   # vasopressor AUROC


# ── training functions: copied verbatim from script2:226-289 ──────────────────
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
        if wait >= patience:
            print(f"    {tag}early stop @ep{ep}"); break
    if best_state: model.load_state_dict(best_state)
    return model


def main():
    t_start = time.time()

    # ── 1. checkpoint: source weights + train_stats ──────────────────────────
    ck = torch.load(SAVE_PATH / "two_stream_models.pt", map_location="cpu", weights_only=False)
    source_state = ck["source"]
    train_stats  = ck["train_stats"]
    seq_dim, treat_dim, n_targets = ck["seq_dim"], ck["treat_dim"], ck["n_targets"]
    print(f"source loaded: seq_dim={seq_dim} treat_dim={treat_dim} targets={n_targets}")
    print(f"train_stats: {len(train_stats)} columns (reused, not recomputed)")

    # ── 2. data, with script2's leakage clipping, then published normalisation ─
    test_df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
    for feat in LEAKAGE_TIMING_FEATS:
        if feat in test_df.columns:
            test_df = test_df.with_columns(
                pl.when(pl.col(feat) > SEQ_LEN).then(SENTINEL_NO_EARLY_EVENT)
                  .otherwise(pl.col(feat)).alias(feat))
    test_df = normalize(test_df, train_stats)
    test_df = test_df.drop([c for c in ["vasopressor_flag", "ventilation_flag"]
                            if c in test_df.columns])
    print(f"test_df normalised: {test_df.shape}")

    # ── 3. split from eval_split.json (verified to reproduce script2) ────────
    sp = json.load(open(SAVE_PATH / "eval_split.json"))
    pre_cp_stays   = set(sp["pre_cp_stays"])
    post_cp_stays  = set(sp["post_cp_stays"])
    eval_post_stays = set(sp["eval_post_stays"])
    eval_post_subj  = set(sp["eval_post_subjects"])

    test_pre  = test_df.filter(pl.col("stay_id").is_in(list(pre_cp_stays)))
    test_post = test_df.filter(pl.col("stay_id").is_in(list(post_cp_stays)))

    # script2:847-862 splits post_subjects 30/10/60 in polars `.unique()` order.
    # That order is NOT reproducible across runs (regen_eval_split.py flags this),
    # and re-deriving it here overlaps the published eval set by only 61%.
    # So: PIN the evaluation population from eval_split.json (exact), and take
    # adapt_train/adapt_val from the complement in a deterministic sorted order.
    # Consequence: the evaluation population matches the published run exactly;
    # the train/val boundary inside the non-eval subjects is not recoverable and
    # affects only which epoch early stopping selects.
    post_subjects = (test_post.filter(pl.col("hrs_from_admit") == 0)["subject_id"]
                     .unique().to_list())
    n_total = len(post_subjects)
    n_tr = int(n_total * 0.30); n_va = int(n_total * 0.10)

    non_eval = sorted(set(post_subjects) - eval_post_subj)
    adapt_train_subjects = non_eval[:n_tr]
    adapt_val_subjects   = non_eval[n_tr:n_tr + n_va]

    print(f"post subjects={n_total} -> train={n_tr} val={n_va} eval={len(eval_post_subj)}")
    print(f"  non-eval pool={len(non_eval)} (train+val={n_tr+n_va})")
    print(f"  eval population PINNED from eval_split.json "
          f"({len(eval_post_stays)} stays / {len(eval_post_subj)} subjects)")
    assert not (set(adapt_train_subjects) & eval_post_subj), "subject leak: train<->eval"
    assert not (set(adapt_val_subjects)   & eval_post_subj), "subject leak: val<->eval"

    adapt_train_df = test_post.filter(pl.col("subject_id").is_in(adapt_train_subjects))
    adapt_val_df   = test_post.filter(pl.col("subject_id").is_in(adapt_val_subjects))
    eval_post_df   = test_post.filter(pl.col("stay_id").is_in(list(eval_post_stays)))

    # ── 4. replay buffer: script2:876-895 ────────────────────────────────────
    pre_first = (test_pre.filter(pl.col("hrs_from_admit") == 0)
                 .sort(["anchor_year_group", "intime"]))
    pre_sorted = pre_first["stay_id"].to_list()
    buf_cand = pre_sorted[-BUFFER_SIZE:] if len(pre_sorted) > BUFFER_SIZE else pre_sorted
    buf_df = test_pre.filter(pl.col("stay_id").is_in(buf_cand))
    leaking = set(buf_df["subject_id"].unique().to_list()) & eval_post_subj
    if leaking:
        buf_df = buf_df.filter(~pl.col("subject_id").is_in(list(leaking)))
        print(f"  removed {len(leaking)} leaking buffer subjects")

    combined_train_df = pl.concat([buf_df, adapt_train_df]) if buf_df.height else adapt_train_df
    n_buf = buf_df.filter(pl.col("hrs_from_admit") == 0).height if buf_df.height else 0
    n_post_tr = adapt_train_df.filter(pl.col("hrs_from_admit") == 0).height
    print(f"adapt-train: {n_post_tr} post + {n_buf} buffer = {n_post_tr+n_buf} "
          f"({100*n_buf/(n_buf+n_post_tr):.1f}% buffer)")

    # ── 5. datasets + adaptation loss (script2:925-928) ──────────────────────
    adapt_train_ds = ICUDataset(combined_train_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    adapt_val_ds   = ICUDataset(adapt_val_df,      SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    eval_ds        = ICUDataset(eval_post_df,      SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    print(f"datasets: adapt_train={len(adapt_train_ds)} adapt_val={len(adapt_val_ds)} eval={len(eval_ds)}")

    adapt_pos_weights = compute_pos_weights(adapt_train_ds, max_weight=15.0)
    adapt_criterion = FocalBCEWithLogitsLoss(pos_weight=adapt_pos_weights,
                                             gamma=1.0, label_smoothing=0.05)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── 6. adapt across seeds ────────────────────────────────────────────────
    states, results = {"B": {}, "C": {}}, {"B": {}, "C": {}}
    for s in SEEDS:
        print(f"\n=== seed {s} ===")
        torch.manual_seed(s); np.random.seed(s)
        atr = DataLoader(adapt_train_ds, batch_size=BATCH_SIZE, shuffle=True)
        ava = DataLoader(adapt_val_ds,   batch_size=BATCH_SIZE, shuffle=False)

        for arm, unfreeze in (("B", "unfreeze_adaptive"), ("C", "unfreeze_all")):
            m = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
            m.load_state_dict(source_state)
            getattr(m, unfreeze)()
            m = train_model(m, atr, ava, adapt_criterion,
                            LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE, f"{arm}-s{s} ")
            met, _, _ = evaluate(m, eval_loader, adapt_criterion)
            states[arm][s] = copy.deepcopy(m.state_dict())
            results[arm][s] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in met.items()}
            au = [met.get(f"{l}_auroc", np.nan) for l in LABEL_COLS]
            print(f"  {arm}-s{s}: " + "  ".join(f"{l.replace('label_','')}={a:.4f}"
                                                for l, a in zip(LABEL_COLS, au)))

    # ── 7. validation gate ───────────────────────────────────────────────────
    vaso_key = [l for l in LABEL_COLS if "vaso" in l][0] + "_auroc"
    print("\n" + "=" * 62)
    print("VALIDATION GATE (local CPU vs published Kaggle run)")
    ok = True
    for (arm, seed), pub in PUBLISHED.items():
        got = results[arm][seed][vaso_key]
        d = abs(got - pub)
        flag = "OK" if d <= 0.005 else "MISMATCH"
        if d > 0.005: ok = False
        print(f"  Run {arm} seed {seed}: got {got:.4f}  published {pub:.4f}  "
              f"delta {d:+.4f}  [{flag}]")
    print("=" * 62)

    for arm in ("B", "C"):
        for l in LABEL_COLS:
            v = np.array([results[arm][s][f"{l}_auroc"] for s in SEEDS])
            print(f"  {arm} {l.replace('label_',''):<16} {v.mean():.4f} +/- {v.std(ddof=1):.4f}  "
                  f"{np.round(v,4).tolist()}")

    torch.save({"states": states, "seeds": SEEDS, "seq_dim": seq_dim,
                "treat_dim": treat_dim, "n_targets": n_targets,
                "source": source_state, "train_stats": train_stats,
                "gate_passed": ok},
               SAVE_PATH / "multiseed_adapted_weights.pt")
    json.dump({"results": results, "seeds": SEEDS, "gate_passed": ok,
               "published_reference": {f"{a}_s{s}": p for (a, s), p in PUBLISHED.items()}},
              open(SAVE_PATH / "multiseed_adaptation.json", "w"), indent=2)
    print(f"\nsaved multiseed_adapted_weights.pt + multiseed_adaptation.json")
    print(f"gate_passed={ok}   elapsed {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
