"""
script23_faithfulness.py
════════════════════════
Does the paper's Integrated-Gradients attribution actually identify the channels
the model relies on?  The manuscript measures explanation STABILITY five ways
but never checks explanation FAITHFULNESS, so the attributions are asserted
rather than validated.

Protocol (standard deletion test, per-instance):
  * for each of the 300 published evaluation cases, take the model's IG
    attribution over the 86 physiological channels for that case's own label
  * mask the top-k channels by |phi| (masking = set to the IG baseline, 0,
    across all six timesteps) and record the change in that label's predicted
    probability
  * repeat with k RANDOM channels as a control, same k, same case
  * paired comparison over the 300 cases

If the attributions are faithful, deleting the top-k must move the prediction
substantially more than deleting k random channels. If the two are comparable,
the attributions do not identify what the model uses, and every stability
result computed on them is measuring the drift of an uninformative vector.

Run for the source model and the two published adapted arms, so faithfulness
can be compared across update regimes.

Outputs  faithfulness.json
"""
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
from utils.data_utils import load_enriched_split, normalize, ICUDataset
from models.architectures import TwoStreamModel

warnings.filterwarnings("ignore")
device = torch.device("cpu")
import os
BASE_PATH = Path(os.environ.get("DATA_DIR", "/home/tamanna/Downloads/paper"))
SAVE_PATH = Path(os.environ.get("OUT_DIR", "/home/tamanna/Documents/try3"))

SEQ_LEN, IG_STEPS = 6, 20
KS = (1, 3, 5, 10, 20)
N_RANDOM_REPEATS = 5
rng = np.random.default_rng(42)
LEAKAGE_TIMING_FEATS = ["time_to_first_abx_order_hrs"]
SENTINEL = float(SEQ_LEN + 1)


def integrated_gradients(model, xs, xt, target_idx, steps=IG_STEPS):
    torch.backends.cudnn.enabled = False
    model.eval()
    xs, xt = xs.to(device), xt.to(device)
    bs, bt = torch.zeros_like(xs), torch.zeros_like(xt)
    sg, tg = torch.zeros_like(xs), torch.zeros_like(xt)
    with torch.enable_grad():
        for alpha in np.linspace(0, 1, steps):
            is_ = (bs + alpha * (xs - bs)).requires_grad_(True)
            it_ = (bt + alpha * (xt - bt)).requires_grad_(True)
            out = model(is_, it_)[:, target_idx].sum()
            g1, g2 = torch.autograd.grad(out, [is_, it_])
            sg += g1; tg += g2
    return ((xs - bs) * sg / steps).cpu().numpy(), ((xt - bt) * tg / steps).cpu().numpy()


@torch.no_grad()
def prob_of(model, xs, xt, ti):
    return torch.sigmoid(model(xs.to(device), xt.to(device)))[0, ti].item()


@torch.no_grad()
def prob_masked(model, xs, xt, ti, cols):
    """Mask the given channel indices (set to IG baseline 0) across all timesteps."""
    x = xs.clone()
    x[0, :, list(cols)] = 0.0
    return torch.sigmoid(model(x.to(device), xt.to(device)))[0, ti].item()


def paired_p(a, b, n_perm=10_000):
    d = np.asarray(a) - np.asarray(b); obs = d.mean()
    perm = (rng.choice([-1, 1], size=(n_perm, len(d))) * d).mean(axis=1)
    return (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)


def main():
    t0 = time.time()
    ck = torch.load(SAVE_PATH / "two_stream_models.pt", map_location="cpu", weights_only=False)
    run_c_state = torch.load(SAVE_PATH / "temp_run_c_weights.pt", map_location="cpu", weights_only=False)
    seq_dim, treat_dim, n_targets = ck["seq_dim"], ck["treat_dim"], ck["n_targets"]
    train_stats = ck["train_stats"]

    expl = json.load(open(SAVE_PATH / "pubmed_rag_explanations.json"))
    cases = [{"stay_id": c["stay_id"], "label": c["label"]} for c in expl["source"]]
    print(f"cases: {len(cases)}")

    test_df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
    for f in LEAKAGE_TIMING_FEATS:
        if f in test_df.columns:
            test_df = test_df.with_columns(
                pl.when(pl.col(f) > SEQ_LEN).then(SENTINEL).otherwise(pl.col(f)).alias(f))
    test_df = normalize(test_df, train_stats)
    test_df = test_df.drop([c for c in ["vasopressor_flag", "ventilation_flag"]
                            if c in test_df.columns])
    sub = test_df.filter(pl.col("stay_id").is_in([c["stay_id"] for c in cases]))
    ds = ICUDataset(sub, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    idx = {int(s): i for i, s in enumerate(ds.stay_ids)}
    n_ch = len(ds.seq_cols)
    print(f"channels: {n_ch}   aligned: {sum(1 for c in cases if c['stay_id'] in idx)}/{len(cases)}")

    arms = {"source": ck["source"], "run_b": ck["run_b"], "run_c": run_c_state}
    out = {}

    for arm, state in arms.items():
        m = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
        m.load_state_dict(state); m.eval()
        top_d = {k: [] for k in KS}
        rnd_d = {k: [] for k in KS}
        for c in cases:
            i = idx.get(c["stay_id"])
            if i is None: continue
            xs, xt, _ = ds[i]
            xs = xs.unsqueeze(0); xt = xt.unsqueeze(0)
            ti = LABEL_COLS.index(c["label"])
            p0 = prob_of(m, xs, xt, ti)
            sa, _ = integrated_gradients(m, xs, xt, ti)
            phi = np.abs(sa[0].sum(axis=0))
            order = np.argsort(-phi)
            for k in KS:
                top_d[k].append(abs(prob_masked(m, xs, xt, ti, order[:k]) - p0))
                rr = [abs(prob_masked(m, xs, xt, ti,
                                      rng.choice(n_ch, size=k, replace=False)) - p0)
                      for _ in range(N_RANDOM_REPEATS)]
                rnd_d[k].append(float(np.mean(rr)))
        out[arm] = {str(k): {"top_mean": float(np.mean(top_d[k])),
                             "rnd_mean": float(np.mean(rnd_d[k])),
                             "ratio": float(np.mean(top_d[k]) / max(np.mean(rnd_d[k]), 1e-12)),
                             "p": float(paired_p(top_d[k], rnd_d[k])),
                             "win_rate": float(np.mean(np.array(top_d[k]) > np.array(rnd_d[k])))}
                    for k in KS}
        print(f"  {arm} done ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 78)
    print("FAITHFULNESS — mean |change in predicted probability| when masking k channels")
    print("top-k = highest |IG|; random-k = control, averaged over 5 draws; n=300 cases")
    for arm in arms:
        print(f"\n--- {arm} ---")
        print(f"{'k':>4}{'top-k':>12}{'random-k':>12}{'ratio':>9}{'top>rnd':>10}{'p':>12}")
        for k in KS:
            r = out[arm][str(k)]
            print(f"{k:>4}{r['top_mean']:>12.4f}{r['rnd_mean']:>12.4f}"
                  f"{r['ratio']:>9.2f}x{r['win_rate']*100:>9.0f}%{r['p']:>12.4g}")

    json.dump({"results": out, "ks": list(KS), "n_cases": len(cases),
               "n_channels": n_ch, "random_repeats": N_RANDOM_REPEATS},
              open(SAVE_PATH / "faithfulness.json", "w"), indent=2)
    print(f"\nsaved faithfulness.json   elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
