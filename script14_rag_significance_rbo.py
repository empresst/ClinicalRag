"""
script14_rag_significance_rbo.py
════════════════════════════════════════════════════════════════════
Post-hoc statistical hardening of the Run B vs Run C retrieval result.

Adds two things a reviewer looks for, WITHOUT retraining, WITHOUT touching
PubMed, and WITHOUT a clinician:

  1. RANK-BIASED OVERLAP (RBO)  — a top-weighted, non-conjoint ranked-list
     similarity measure (Webber, Moffat & Zobel, ACM TOIS 2010). It replaces
     the fragile Spearman rank-correlation row in Table 7, which was only
     defined on cases with >=3 shared PMIDs (241/300) and was therefore
     upward-biased (the most-divergent cases were dropped). RBO is defined
     for ALL 300 cases, rewards agreement at the top of the list, and needs
     no relevance judgments.

  2. PAIRED SIGNIFICANCE — the IR-recommended way. The information-retrieval
     methodology literature (Smucker, Allan & Carterette, CIKM 2007; and the
     SIGIR 2025 "Stop Using the Wilcoxon Test" paper) recommends the paired
     RANDOMIZATION (permutation) test and the BOOTSTRAP over the Wilcoxon /
     sign test, which have poor power and can produce false positives. The
     bootstrap here also matches the B=1000 bootstrap already used for the
     AUROC contrasts in the paper (Sec. 2.7), so the whole paper tests
     significance the same way.

INPUTS (already saved by script13_rag_pubmed_bvsc.py — nothing regenerated):
    pubmed_rag_stability.json     — per-case Jaccard / delta values (300 cases)
    pubmed_rag_explanations.json  — per-case RANKED hit lists (for RBO)

OUTPUT:
    pubmed_rag_significance.json  — every number below, machine-readable

This script is pure numpy/json: it runs in seconds on CPU.
"""

import json
from pathlib import Path
import numpy as np

SEED       = 42
import os
SAVE_PATH  = Path(os.environ.get("OUT_DIR", "/home/tamanna/Documents/try3"))
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]

# ── Analysis config ───────────────────────────────────────────────────────────
RBO_P      = 0.9      # persistence parameter; 0.9 ⇒ ~86% of weight in the top-10.
TOP_K_DOCS = 5        # list depth retrieved per query (must match script13)
N_PERM     = 10000    # paired randomization (sign-flip) resamples
N_BOOT     = 10000    # bootstrap resamples for the 95% CI
ALPHA      = 0.05

np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# RANK-BIASED OVERLAP  (extrapolated form, RBO_EXT)
#   Webber, Moffat & Zobel (2010), "A Similarity Measure for Indefinite Rankings"
#   RBO_EXT = (X_k / k) · p^k  +  ((1-p)/p) · Σ_{d=1..k} (X_d / d) · p^d
#   where X_d = |prefix_d(A) ∩ prefix_d(B)|. Ranges 0 (disjoint) → 1 (identical).
#   Top-weighted, handles non-conjoint lists, needs no relevance labels.
# ══════════════════════════════════════════════════════════════════════════════
def rbo_ext(list_a, list_b, p=RBO_P):
    if not list_a and not list_b:
        return 1.0
    if not list_a or not list_b:
        return 0.0
    k = max(len(list_a), len(list_b))
    agreement = []                       # A_d = X_d / d, for d = 1..k
    for d in range(1, k + 1):
        x_d = len(set(list_a[:d]) & set(list_b[:d]))
        agreement.append(x_d / d)
    sum_term = sum(a * (p ** d) for d, a in zip(range(1, k + 1), agreement))
    return agreement[-1] * (p ** k) + ((1.0 - p) / p) * sum_term


# ══════════════════════════════════════════════════════════════════════════════
# PAIRED SIGNIFICANCE MACHINERY
# ══════════════════════════════════════════════════════════════════════════════
def _clean_pairs(b_vals, c_vals):
    """Return the paired (B − C) differences, dropping any pair with a NaN."""
    b = np.asarray(b_vals, dtype=float)
    c = np.asarray(c_vals, dtype=float)
    mask = ~(np.isnan(b) | np.isnan(c))
    return b, c, mask, (b[mask] - c[mask])


def paired_randomization_test(diffs, n_perm=N_PERM, seed=SEED):
    """
    Two-sided paired randomization (permutation) test.
    Under H0 (B and C exchangeable within each patient) the sign of each
    per-patient difference is equally likely ±. We flip signs at random
    n_perm times and count how often |mean| meets or beats the observed one.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    obs = float(diffs.mean())
    rng = np.random.RandomState(seed)
    signs = rng.choice([1.0, -1.0], size=(n_perm, n))
    perm_means = (signs * diffs).mean(axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(obs) - 1e-12) + 1) / (n_perm + 1)
    return obs, float(p)


def bootstrap_ci(diffs, n_boot=N_BOOT, seed=SEED, alpha=ALPHA):
    """Percentile bootstrap CI for the mean paired difference."""
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed + 1)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(metric_name, b_vals, c_vals, higher_is_more_stable=True):
    """Full paired comparison of Run B vs Run C for one metric."""
    b, c, mask, diffs = _clean_pairs(b_vals, c_vals)
    mean_b   = float(np.nanmean(b))
    mean_c   = float(np.nanmean(c))
    obs, p   = paired_randomization_test(diffs)
    lo, hi   = bootstrap_ci(diffs)
    n_pairs  = int(mask.sum())
    pct_b_ge = float(100.0 * np.mean(diffs >= 0)) if n_pairs else float("nan")
    return {
        "metric":        metric_name,
        "mean_b":        mean_b,
        "mean_c":        mean_c,
        "mean_diff":     obs,               # B − C
        "ci95_low":      lo,
        "ci95_high":     hi,
        "perm_p":        p,
        "pct_b_ge_c":    pct_b_ge,
        "n_pairs":       n_pairs,
        "favours":       ("B" if obs > 0 else "C") if higher_is_more_stable
                         else ("B" if obs < 0 else "C"),
    }


def diff_in_diff(dd_vals):
    """
    Difference-in-differences test on a single vector (already B−C on one stream
    minus B−C on another stream). Tests whether the freeze's effect is larger on
    one stream than the other.
    """
    obs, p = paired_randomization_test(dd_vals)
    lo, hi = bootstrap_ci(dd_vals)
    return {"mean_dd": obs, "ci95_low": lo, "ci95_high": hi, "perm_p": p,
            "n": int(np.sum(~np.isnan(np.asarray(dd_vals, float))))}


def _fmt(s):
    p = s["perm_p"]
    p_str = "<1e-4" if p < 1e-4 else f"{p:.4f}"
    return (f"  {s['metric']:<26} B={s['mean_b']:.3f}  C={s['mean_c']:.3f}  "
            f"Δ(B−C)={s['mean_diff']:+.3f}  "
            f"95% CI[{s['ci95_low']:+.3f},{s['ci95_high']:+.3f}]  "
            f"p={p_str}  B≥C:{s['pct_b_ge_c']:.0f}%  (favours {s['favours']}, n={s['n_pairs']})")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD SAVED ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading saved RAG artifacts (no models / no PubMed / no retraining)")
print("=" * 70)

with open(SAVE_PATH / "pubmed_rag_stability.json") as f:
    all_cases = json.load(f)
with open(SAVE_PATH / "pubmed_rag_explanations.json") as f:
    explanations = json.load(f)

print(f"✅ stability cases: {len(all_cases)}")
print(f"✅ explanation sets: source={len(explanations['source'])} "
      f"run_b={len(explanations['run_b'])} run_c={len(explanations['run_c'])}")

# Index explanations by (stay_id, label) so we can pair source / B / C per case
def _index(store):
    return {(e["stay_id"], e["label"]): e for e in store}

exp_src = _index(explanations["source"])
exp_b   = _index(explanations["run_b"])
exp_c   = _index(explanations["run_c"])


def _pmids(entry, stream):
    """Ranked PMID list for a stream ('physio_hits' / 'treat_hits')."""
    if entry is None:
        return []
    return [h["pmid"] for h in entry.get(stream, [])]


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE PER-CASE RBO  (source-vs-B and source-vs-C, both streams)
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing Rank-Biased Overlap (RBO) per case...")
n_missing = 0
for c in all_cases:
    key = (c["stay_id"], c["label"])
    s_e, b_e, c_e = exp_src.get(key), exp_b.get(key), exp_c.get(key)
    if not (s_e and b_e and c_e):
        n_missing += 1
        c["rbo_physio_b"] = c["rbo_physio_c"] = float("nan")
        c["rbo_treat_b"]  = c["rbo_treat_c"]  = float("nan")
        continue
    src_ph, src_tr = _pmids(s_e, "physio_hits"), _pmids(s_e, "treat_hits")
    c["rbo_physio_b"] = rbo_ext(src_ph, _pmids(b_e, "physio_hits"))
    c["rbo_physio_c"] = rbo_ext(src_ph, _pmids(c_e, "physio_hits"))
    c["rbo_treat_b"]  = rbo_ext(src_tr, _pmids(b_e, "treat_hits"))
    c["rbo_treat_c"]  = rbo_ext(src_tr, _pmids(c_e, "treat_hits"))
if n_missing:
    print(f"  ⚠ {n_missing} cases had no matching explanation (RBO set to NaN)")
print(f"  ✅ RBO computed (p={RBO_P}, depth k={TOP_K_DOCS}) on "
      f"{len(all_cases) - n_missing}/{len(all_cases)} cases")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: pull a metric column for a subset of cases
# ══════════════════════════════════════════════════════════════════════════════
def col(cases, key):
    return [c.get(key, float("nan")) for c in cases]

def subset(label=None):
    return all_cases if label is None else [c for c in all_cases if c["label"] == label]


# ══════════════════════════════════════════════════════════════════════════════
# OVERALL RESULTS  (all 300 cases)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"PAIRED B-vs-C SIGNIFICANCE  (randomization test, {N_PERM} perms; "
      f"bootstrap {N_BOOT})")
print("=" * 70)
print("Positive Δ(B−C) and 'favours B' = Run B is MORE stable (closer to source).\n")

results = {"config": {"rbo_p": RBO_P, "depth_k": TOP_K_DOCS, "n_perm": N_PERM,
                      "n_boot": N_BOOT, "seed": SEED},
           "overall": {}, "per_label": {}}

overall_metrics = [
    ("Physio Jaccard",  "jacc_physio_b", "jacc_physio_c", True),
    ("Physio RBO",      "rbo_physio_b",  "rbo_physio_c",  True),
    ("Treat  Jaccard",  "jacc_treat_b",  "jacc_treat_c",  True),
    ("Treat  RBO",      "rbo_treat_b",   "rbo_treat_c",   True),
    # physio attribution delta: LOWER = more stable, so B expected < C
    ("Physio |Δattr|",  "physio_delta_b","physio_delta_c",False),
]

print("── Overall (all 300 cases) ─────────────────────────────────────────")
for name, kb, kc, hib in overall_metrics:
    s = summarize(name, col(all_cases, kb), col(all_cases, kc), hib)
    results["overall"][name] = s
    print(_fmt(s))


# ══════════════════════════════════════════════════════════════════════════════
# STREAM-LOCALISATION TEST (difference-in-differences)
# Is the freeze's stabilising effect significantly LARGER on the physiology
# stream than on the treatment stream? If yes, the effect localises to the
# frozen channel exactly as the mechanism predicts.
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Stream localisation (physio advantage − treat advantage) ────────")
dd_jacc = [ (c["jacc_physio_b"] - c["jacc_physio_c"])
          - (c["jacc_treat_b"]  - c["jacc_treat_c"]) for c in all_cases ]
dd_rbo  = [ (c["rbo_physio_b"]  - c["rbo_physio_c"])
          - (c["rbo_treat_b"]   - c["rbo_treat_c"])  for c in all_cases ]
loc_jacc = diff_in_diff(dd_jacc)
loc_rbo  = diff_in_diff(dd_rbo)
results["localisation"] = {"jaccard": loc_jacc, "rbo": loc_rbo}
for nm, d in [("Jaccard", loc_jacc), ("RBO", loc_rbo)]:
    pj = "<1e-4" if d["perm_p"] < 1e-4 else f"{d['perm_p']:.4f}"
    print(f"  {nm:<10} mean Δ-of-Δ={d['mean_dd']:+.3f}  "
          f"95% CI[{d['ci95_low']:+.3f},{d['ci95_high']:+.3f}]  p={pj}  "
          f"(>0 ⇒ freeze stabilises physio more than treatment)")


# ══════════════════════════════════════════════════════════════════════════════
# PER-LABEL BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Per-label (physiology stream) ───────────────────────────────────")
for ln in LABEL_COLS:
    sub = subset(ln)
    results["per_label"][ln] = {}
    print(f"\n  {ln}  (n={len(sub)})")
    for name, kb, kc in [("Physio Jaccard", "jacc_physio_b", "jacc_physio_c"),
                         ("Physio RBO",     "rbo_physio_b",  "rbo_physio_c")]:
        s = summarize(name, col(sub, kb), col(sub, kc), True)
        results["per_label"][ln][name] = s
        print(_fmt(s))


# ══════════════════════════════════════════════════════════════════════════════
# SAVE + PASTE-READY LINES
# ══════════════════════════════════════════════════════════════════════════════
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

with open(SAVE_PATH / "pubmed_rag_significance.json", "w") as f:
    json.dump(results, f, indent=2, cls=_NpEncoder)
print(f"\n✅ Saved → {SAVE_PATH / 'pubmed_rag_significance.json'}")

pj = results["overall"]["Physio Jaccard"]
pr = results["overall"]["Physio RBO"]
def _p(s): return "<0.0001" if s["perm_p"] < 1e-4 else f"{s['perm_p']:.4f}"
print("\n" + "=" * 70)
print("PASTE-READY (Table 7 RBO row + significance sentence)")
print("=" * 70)
print(f"  RBO row  →  Run B {pr['mean_b']:.3f}   Run C {pr['mean_c']:.3f}")
print(f"\n  \"Run B's physiology retrieval is more stable than Run C's on both")
print(f"   Jaccard (ΔJaccard = {pj['mean_diff']:+.3f}, 95% CI "
      f"[{pj['ci95_low']:+.3f}, {pj['ci95_high']:+.3f}], randomization p = {_p(pj)}) and")
print(f"   rank-biased overlap (ΔRBO = {pr['mean_diff']:+.3f}, 95% CI "
      f"[{pr['ci95_low']:+.3f}, {pr['ci95_high']:+.3f}], p = {_p(pr)}), computed on all")
print(f"   {pj['n_pairs']} post-drift cases with no relevance judgments; RBO is defined")
print(f"   for every case, unlike the Spearman row it replaces.\"")
print("\n✅ Done.")
