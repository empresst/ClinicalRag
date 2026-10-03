"""
script19_regime_separability.py
═══════════════════════════════
Two readouts the manuscript reports from the saved attribution vectors, with no
model loading and no data access:

  1. Regime separability (Sec. "Ordering does carry some information"):
     each case's signed Spearman correlation between the adapted model's and the
     source model's 86-channel physiological attribution is used as a score for
     telling Run B cases from Run C cases.
       - AUC  = P(score_B > score_C) over all B x C pairs (ties count 1/2)
       - share of the 300 paired cases where Run B's agreement exceeds Run C's

  2. Treatment-stream attribution similarity: the same paired comparison as
     script18, on the 12-dimensional treatment attribution vector that
     script13Rag_BvsC.py stores alongside the physiological one (attr_treat).

Inputs   pubmed_rag_explanations.json   (written by script13Rag_BvsC.py)
Outputs  regime_separability.json
"""
import json, os
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

SAVE_PATH = Path(os.environ.get("OUT_DIR", "."))
SEED, N_PERM, N_BOOT = 42, 10_000, 10_000
rng = np.random.default_rng(SEED)

d = json.load(open(SAVE_PATH / "pubmed_rag_explanations.json"))
src, run_b, run_c = d["source"], d["run_b"], d["run_c"]


def vec(case, key):
    names = list(src[0][key].keys())
    return np.array([case[key][n] for n in names])


def paired(B, C):
    """Paired randomization test (sign-flips) with percentile bootstrap CI."""
    d_ = np.asarray(B) - np.asarray(C); obs = d_.mean()
    perm = (rng.choice([-1, 1], size=(N_PERM, len(d_))) * d_).mean(axis=1)
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (N_PERM + 1)
    bs = d_[rng.integers(0, len(d_), size=(N_BOOT, len(d_)))].mean(axis=1)
    return float(obs), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(p)


out = {"n_cases": len(src)}

# ── 1. regime separability on the physiological stream ───────────────────────
rho_b = np.array([spearmanr(vec(s, "attr_physio"), vec(b, "attr_physio"))[0]
                  for s, b in zip(src, run_b)])
rho_c = np.array([spearmanr(vec(s, "attr_physio"), vec(c, "attr_physio"))[0]
                  for s, c in zip(src, run_c)])
auc = float((rho_b[:, None] > rho_c[None, :]).mean()
            + 0.5 * (rho_b[:, None] == rho_c[None, :]).mean())
pct = float((rho_b > rho_c).mean())
out["separability"] = dict(auc=auc, pct_b_gt_c=pct,
                           mean_rho_b=float(rho_b.mean()), mean_rho_c=float(rho_c.mean()))
print(f"\n  regime separability (signed rho vs source, n={len(src)})")
print(f"    AUC = {auc:.3f}   Run B higher in {100*pct:.1f}% of paired cases")

# ── 2. treatment-stream attribution similarity ───────────────────────────────
M = {"rho_signed": {"B": [], "C": []}, "rho_abs": {"B": [], "C": []}}
for s_, b_, c_ in zip(src, run_b, run_c):
    vs = vec(s_, "attr_treat")
    for tag, arm in (("B", b_), ("C", c_)):
        va = vec(arm, "attr_treat")
        M["rho_signed"][tag].append(spearmanr(vs, va)[0])
        M["rho_abs"][tag].append(spearmanr(np.abs(vs), np.abs(va))[0])

print(f"\n  treatment attribution similarity to source (n={len(src)})")
out["treatment_attribution"] = {}
for k, M_ in M.items():
    B, C = np.nan_to_num(M_["B"]), np.nan_to_num(M_["C"])
    o, lo, hi, p = paired(B, C)
    out["treatment_attribution"][k] = dict(mean_b=float(B.mean()), mean_c=float(C.mean()),
                                           diff=o, ci_lo=lo, ci_hi=hi, p=p)
    ps = "<1e-4" if p < 1e-4 else f"{p:.4f}"
    print(f"    {k:12s} B {B.mean():.3f}  C {C.mean():.3f}  diff {o:+.3f} [{lo:+.3f},{hi:+.3f}] p {ps}")

# ── 3. headline contrasts without the septic shock slice (supplementary material) ─
keep = [c["label"] != "label_septic_shock" for c in src]
top5 = lambda v: set(np.argsort(-np.abs(v))[:5])
rb, rc, jb, jc = [], [], [], []
for s_, b_, c_, k in zip(src, run_b, run_c, keep):
    if not k:
        continue
    vs, vb, vc = vec(s_, "attr_physio"), vec(b_, "attr_physio"), vec(c_, "attr_physio")
    rb.append(spearmanr(vs, vb)[0]); rc.append(spearmanr(vs, vc)[0])
    jb.append(len(top5(vs) & top5(vb)) / len(top5(vs) | top5(vb)))
    jc.append(len(top5(vs) & top5(vc)) / len(top5(vs) | top5(vc)))
stab = json.load(open(SAVE_PATH / "pubmed_rag_stability.json"))
ret = [c["jacc_physio_b"] - c["jacc_physio_c"] for c in stab if c["label"] != "label_septic_shock"]
out["without_septic_shock"] = dict(n=len(rb), rho_signed_diff=float(np.mean(rb) - np.mean(rc)),
                                   top5_jaccard_diff=float(np.mean(jb) - np.mean(jc)),
                                   physio_retrieval_jaccard_diff=float(np.mean(ret)))
print(f"\n  without septic shock (n={len(rb)}): signed rho {np.mean(rb)-np.mean(rc):+.3f}  "
      f"top-5 Jaccard {np.mean(jb)-np.mean(jc):+.3f}  retrieval Jaccard {np.mean(ret):+.3f}")

json.dump(out, open(SAVE_PATH / "regime_separability.json", "w"), indent=2)
print("\n  wrote regime_separability.json")
