"""
script22_relocation_check.py
═════════════════════════════════
Across-seed version of Table 8 (physiological attribution similarity to the
source model) and Table 7 (relative weight change by block).

Uses the ten adapted checkpoints written by script20_multiseed_adaptation.py
(5 seeds x {Run B, Run C}), the SAME 300 evaluation cases as the published run
(read from pubmed_rag_explanations.json), and the SAME Integrated Gradients
procedure copied verbatim from script13Rag_BvsC.py:369 (zero baseline, 20
steps, eval mode, attributions summed over the six timesteps).

Caveat carried from script20: the adaptation train/validation boundary is not
recoverable from the published artefacts, so absolute AUROCs differ slightly
from Table 5. The paired Run B - Run C contrast is unaffected, because both
arms share source weights, adaptation data, train/val split and seed.

Outputs  relocation_check.json (physiology and treatment streams, per seed)
"""
import json, sys, time, warnings
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).parent))
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
from utils.data_utils import load_enriched_split, normalize, ICUDataset
from models.architectures import TwoStreamModel

warnings.filterwarnings("ignore")
device = torch.device("cpu")
import os
BASE_PATH = Path(os.environ.get("DATA_DIR", "/home/tamanna/Downloads/paper"))
SAVE_PATH = Path(os.environ.get("OUT_DIR", "/home/tamanna/Documents/try3"))

SEQ_LEN, IG_STEPS, BATCH = 6, 20, 64
N_PERM, N_BOOT = 10_000, 10_000
rng = np.random.default_rng(42)
LEAKAGE_TIMING_FEATS = ["time_to_first_abx_order_hrs"]
SENTINEL = float(SEQ_LEN + 1)

PUBLISHED_T8 = {"rho_signed": +0.063, "rho_abs": +0.034,
                "tau": +0.053, "j5": +0.122, "j10": +0.108}


# ── IG: verbatim from script13Rag_BvsC.py:369 ────────────────────────────────
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


def paired(B, C):
    """Paired randomization test + percentile bootstrap CI — script18 verbatim."""
    d_ = np.asarray(B) - np.asarray(C); obs = d_.mean()
    perm = (rng.choice([-1, 1], size=(N_PERM, len(d_))) * d_).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (N_PERM + 1)
    bs = d_[rng.integers(0, len(d_), size=(N_BOOT, len(d_)))].mean(axis=1)
    return obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p


def block_delta(src, adapted):
    """Relative weight change per block (Table 7)."""
    out = {}
    for blk in ("physio", "treat", "fusion"):
        rel = []
        for k in src:
            if k.startswith(blk + "."):
                a, b = src[k].float(), adapted[k].float()
                den = a.norm().item()
                rel.append(((b - a).norm().item() / den) if den > 0 else 0.0)
        out[blk] = {"mean": float(np.mean(rel)), "max": float(np.max(rel))} if rel else None
    return out


def main():
    t0 = time.time()
    mk = torch.load(SAVE_PATH / "multiseed_adapted_weights.pt", map_location="cpu", weights_only=False)
    states, SEEDS = mk["states"], mk["seeds"]
    src_state, train_stats = mk["source"], mk["train_stats"]
    seq_dim, treat_dim, n_targets = mk["seq_dim"], mk["treat_dim"], mk["n_targets"]
    print(f"checkpoints: {len(SEEDS)} seeds x B,C   gate_passed={mk.get('gate_passed')}")

    # the same 300 cases as the published run
    expl = json.load(open(SAVE_PATH / "pubmed_rag_explanations.json"))
    cases = [{"stay_id": c["stay_id"], "label": c["label"]} for c in expl["source"]]
    print(f"cases: {len(cases)} (from pubmed_rag_explanations.json)")

    # data, normalised exactly as in script20
    test_df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
    for f in LEAKAGE_TIMING_FEATS:
        if f in test_df.columns:
            test_df = test_df.with_columns(
                pl.when(pl.col(f) > SEQ_LEN).then(SENTINEL).otherwise(pl.col(f)).alias(f))
    test_df = normalize(test_df, train_stats)
    test_df = test_df.drop([c for c in ["vasopressor_flag", "ventilation_flag"]
                            if c in test_df.columns])

    stay_ids = [c["stay_id"] for c in cases]
    sub = test_df.filter(pl.col("stay_id").is_in(stay_ids))
    ds = ICUDataset(sub, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
    seq_cols = ds.seq_cols
    id_index = {int(s): i for i, s in enumerate(ds.stay_ids)} if hasattr(ds, "stay_ids") else None
    if id_index is None:
        raise SystemExit("ICUDataset exposes no stay_ids; cannot align cases")
    print(f"aligned {sum(1 for c in cases if c['stay_id'] in id_index)}/{len(cases)} cases")

    def build(state):
        m = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
        m.load_state_dict(state); m.eval(); return m

    def attrs_for(model):
        """(86-dim physiology, 12-dim treatment) attribution vectors per case."""
        out = []
        for c in cases:
            i = id_index.get(c["stay_id"])
            if i is None: out.append(None); continue
            xs, xt, _ = ds[i]
            xs = xs.unsqueeze(0); xt = xt.unsqueeze(0)
            ti = LABEL_COLS.index(c["label"])
            sa, ta = integrated_gradients(model, xs, xt, ti)
            out.append((sa[0].sum(axis=0), ta[0]))
        return out

    print("computing source attributions...")
    m_src = build(src_state)
    A_src = attrs_for(m_src)
    P_src = [a[0] for a in A_src]; T_src = [a[1] for a in A_src]
    print(f"  done ({time.time()-t0:.0f}s)")

    def metrics_vs_src(A_s, A_a, names, ks=(3, 5)):
        M = {"rho_signed": [], "rho_abs": [], "tau": []}
        for k in ks: M[f"j{k}"] = []
        for vs, va in zip(A_s, A_a):
            M["rho_signed"].append(spearmanr(vs, va)[0])
            M["rho_abs"].append(spearmanr(np.abs(vs), np.abs(va))[0])
            M["tau"].append(kendalltau(np.abs(vs), np.abs(va))[0])
            for k in ks:
                ts = set(np.array(names)[np.argsort(-np.abs(vs))[:k]])
                ta_ = set(np.array(names)[np.argsort(-np.abs(va))[:k]])
                M[f"j{k}"].append(len(ts & ta_) / len(ts | ta_))
        return M

    results, t7 = {}, {}
    treat_cols = ds.treat_cols
    print(f"treatment features: {len(treat_cols)}")
    for s_ in SEEDS:
        per_arm = {}
        for arm in ("B", "C"):
            m = build(states[arm][s_])
            A = attrs_for(m)
            P = [a[0] for a in A]; T = [a[1] for a in A]
            per_arm[arm] = {"physio": metrics_vs_src(P_src, P, seq_cols, ks=(5, 10)),
                            "treat":  metrics_vs_src(T_src, T, treat_cols, ks=(3, 5))}
            t7.setdefault(arm, {})[s_] = block_delta(src_state, states[arm][s_])
        results[s_] = per_arm
        print(f"  seed {s_}: physio j5 B={np.mean(per_arm['B']['physio']['j5']):.3f} "
              f"C={np.mean(per_arm['C']['physio']['j5']):.3f} | "
              f"treat j3 B={np.mean(per_arm['B']['treat']['j3']):.3f} "
              f"C={np.mean(per_arm['C']['treat']['j3']):.3f}  ({time.time()-t0:.0f}s)")

    print("\n" + "=" * 78)
    print("RELOCATION CHECK - attribution similarity to source, per stream, 5 seeds")
    print("Positive delta = the FREEZE keeps that stream closer to the source model.")
    summary = {}
    for stream, mets in (("physio", ("rho_signed", "rho_abs", "tau", "j5", "j10")),
                         ("treat",  ("rho_signed", "rho_abs", "tau", "j3", "j5"))):
        print(f"\n--- {stream} stream ---")
        print(f"{'metric':<12}{'Run B':>15}{'Run C':>15}{'delta (mean+/-SD)':>22}{'+ in':>7}")
        for k in mets:
            b = np.array([np.mean(results[s_][ "B"][stream][k]) for s_ in SEEDS])
            c = np.array([np.mean(results[s_][ "C"][stream][k]) for s_ in SEEDS])
            d = b - c
            summary[f"{stream}_{k}"] = {"B_mean": float(b.mean()), "C_mean": float(c.mean()),
                                        "delta_mean": float(d.mean()), "delta_sd": float(d.std(ddof=1)),
                                        "per_seed": [float(x) for x in d],
                                        "n_pos": int((d > 0).sum())}
            print(f"{k:<12}{b.mean():>8.3f}+/-{b.std(ddof=1):<5.3f}"
                  f"{c.mean():>8.3f}+/-{c.std(ddof=1):<5.3f}"
                  f"{d.mean():>+13.4f}+/-{d.std(ddof=1):<7.4f}{int((d>0).sum()):>4}/5")

    print("\n" + "=" * 78)
    print("DIFFERENCE-IN-DIFFERENCES  (physio delta) - (treat delta)")
    print("Positive = the freeze helps the FROZEN stream more than the adapted one.")
    for pk, tk in (("rho_signed", "rho_signed"), ("rho_abs", "rho_abs"),
                   ("tau", "tau"), ("j5", "j5")):
        pd_ = np.array(summary[f"physio_{pk}"]["per_seed"])
        td_ = np.array(summary[f"treat_{tk}"]["per_seed"])
        did = pd_ - td_
        summary[f"DiD_{pk}"] = {"mean": float(did.mean()), "sd": float(did.std(ddof=1)),
                                "per_seed": [float(x) for x in did],
                                "n_pos": int((did > 0).sum())}
        print(f"  {pk:<12} {did.mean():>+9.4f} +/- {did.std(ddof=1):<8.4f} "
              f"positive in {int((did>0).sum())}/5   per-seed {np.round(did,4).tolist()}")

    json.dump({"relocation": summary,
               "table7": {a: {str(s): t7[a][s] for s in SEEDS} for a in ("B", "C")},
               "seeds": SEEDS, "n_cases": len(cases)},
              open(SAVE_PATH / "relocation_check.json", "w"), indent=2)
    print(f"\nsaved relocation_check.json   elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
