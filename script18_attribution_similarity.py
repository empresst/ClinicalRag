"""
script18_attribution_similarity.py
══════════════════════════════════
Produces the two attribution results the manuscript reports beyond retrieval:

  Table `tbl:attrsim`  source-vs-adapted attribution similarity on the physiology
                       stream, with the metrics conventionally used to compare
                       explanations across models (rank correlation over the full
                       86-dim vector, signed and on magnitudes; top-k Jaccard),
                       each with a paired randomization test and bootstrap CI.

  Sec. `sec:delta-attr` representation identity: h_phys is bitwise identical
                       between the source model and Run B for every evaluation
                       stay, and differs for every stay under Run C.

Inputs   pubmed_rag_explanations.json   (written by script13Rag_BvsC.py, which
                                         stores the full per-case attribution
                                         vectors in attr_physio / attr_treat)
         two_stream_models.pt, temp_run_c_weights.pt
         <BASE_PATH>/test_final_enriched4.parquet
Outputs  attribution_similarity.json  +  a digest block for the paper
"""
import json, sys
from pathlib import Path
import numpy as np, polars as pl, torch
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, ".")
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
from utils.data_utils import load_enriched_split, normalize, ICUDataset
from models.architectures import TwoStreamModel

import os
BASE_PATH = Path(os.environ.get("DATA_DIR", "/home/tamanna/Downloads/paper"))
SAVE_PATH = Path(os.environ.get("OUT_DIR", "."))
SEED, N_PERM, N_BOOT, N_IDENT = 42, 10_000, 10_000, 2_000
rng = np.random.default_rng(SEED)

# ── 1. attribution similarity ────────────────────────────────────────────────
d = json.load(open(SAVE_PATH / "pubmed_rag_explanations.json"))
src, run_b, run_c = d["source"], d["run_b"], d["run_c"]
names = list(src[0]["attr_physio"].keys())
vec = lambda case: np.array([case["attr_physio"][n] for n in names])

M = {k: {"B": [], "C": []} for k in ("rho_signed", "rho_abs", "tau", "j5", "j10")}
for s_, b_, c_ in zip(src, run_b, run_c):
    vs = vec(s_)
    for tag, arm in (("B", b_), ("C", c_)):
        va = vec(arm)
        M["rho_signed"][tag].append(spearmanr(vs, va)[0])
        M["rho_abs"][tag].append(spearmanr(np.abs(vs), np.abs(va))[0])
        M["tau"][tag].append(kendalltau(np.abs(vs), np.abs(va))[0])
        for k, key in ((5, "j5"), (10, "j10")):
            ts = set(np.array(names)[np.argsort(-np.abs(vs))[:k]])
            ta = set(np.array(names)[np.argsort(-np.abs(va))[:k]])
            M[key][tag].append(len(ts & ta) / len(ts | ta))

def paired(B, C):
    """Paired randomization test (sign-flips) with percentile bootstrap CI."""
    d_ = np.asarray(B) - np.asarray(C); obs = d_.mean()
    perm = (rng.choice([-1, 1], size=(N_PERM, len(d_))) * d_).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (N_PERM + 1)
    bs = d_[rng.integers(0, len(d_), size=(N_BOOT, len(d_)))].mean(axis=1)
    return obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p

LABEL = {"rho_signed": "Spearman rho, signed", "rho_abs": "Spearman rho, |phi|",
         "tau": "Kendall tau", "j5": "Top-5 Jaccard", "j10": "Top-10 Jaccard"}
out = {}
print(f"\n  physiological attribution similarity to source (n={len(src)})")
print(f"  {'metric':24s} {'Run B':>7s} {'Run C':>7s} {'delta':>8s}  {'95% CI':>18s} {'p':>8s}")
for k in ("rho_signed", "rho_abs", "tau", "j5", "j10"):
    B, C = M[k]["B"], M[k]["C"]
    o, lo, hi, p = paired(B, C)
    out[k] = dict(mean_b=float(np.mean(B)), mean_c=float(np.mean(C)),
                  diff=float(o), ci_lo=float(lo), ci_hi=float(hi), p=float(p))
    ps = "<1e-4" if p < 1e-4 else f"{p:.4f}"
    print(f"  {LABEL[k]:24s} {np.mean(B):7.3f} {np.mean(C):7.3f} {o:+8.3f}"
          f"  [{lo:+.3f},{hi:+.3f}] {ps:>8s}")

# sign-flip rate among the features that drive a query
flips = {"B": [], "C": []}
for s_, b_, c_ in zip(src, run_b, run_c):
    vs = vec(s_); top = np.argsort(-np.abs(vs))[:5]
    for tag, arm in (("B", b_), ("C", c_)):
        flips[tag].append(np.mean(np.sign(vs[top]) != np.sign(vec(arm)[top])))
out["sign_flip_top5"] = {t: float(np.mean(v)) for t, v in flips.items()}
print(f"\n  sign flips among source top-5: B {100*np.mean(flips['B']):.1f}%  "
      f"C {100*np.mean(flips['C']):.1f}%")

# ── 2. representation identity ───────────────────────────────────────────────
ck = torch.load(SAVE_PATH / "two_stream_models.pt", map_location="cpu", weights_only=False)
def build(sd):
    m = TwoStreamModel(ck["seq_dim"], ck["treat_dim"], ck["n_targets"])
    m.load_state_dict(sd); m.eval(); return m
m_src, m_b = build(ck["source"]), build(ck["run_b"])
m_c = build(torch.load(SAVE_PATH / "temp_run_c_weights.pt", map_location="cpu", weights_only=False))

test = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
test = normalize(test, ck["train_stats"]).drop(
    [c for c in ("vasopressor_flag", "ventilation_flag") if c in test.columns])
# restrict to the post-drift evaluation partition written by script2 / regen_eval_split
eval_post = json.load(open(SAVE_PATH / "eval_split.json"))["eval_post_stays"]
test = test.filter(pl.col("stay_id").is_in(eval_post))
ds = ICUDataset(test, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, 6)
n = min(N_IDENT, len(ds))
xs = torch.stack([ds[i][0] for i in range(n)])
with torch.no_grad():
    h_src, h_b, h_c = m_src.physio(xs), m_b.physio(xs), m_c.physio(xs)

print(f"\n  h_phys identity over {n} evaluation stays")
for tag, h in (("Run B", h_b), ("Run C", h_c)):
    diff = (h - h_src).abs()
    changed = (diff.max(dim=1).values > 1e-6).float().mean().item()
    print(f"    source vs {tag}: bitwise identical={torch.equal(h, h_src)}  "
          f"max|delta|={diff.max().item():.6f}  stays changed={100*changed:.1f}%")
    print(f"    mean|delta|={diff.mean().item():.4f}")
    out[f"h_phys_{tag.split()[-1]}"] = dict(identical=bool(torch.equal(h, h_src)),
                                            max_abs=float(diff.max()),
                                            mean_abs=float(diff.mean()),
                                            frac_changed=float(changed),
                                            n_stays=n)

json.dump(out, open(SAVE_PATH / "attribution_similarity.json", "w"), indent=2)
print("\n  wrote attribution_similarity.json")
