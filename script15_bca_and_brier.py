"""
Recomputes, from SAVED PREDICTIONS ONLY (no retraining, no MIMIC access):

  1. True BCa 95% CIs for the paired AUROC deltas  (B-A, B-C, C-A, D-A)
  2. Sign-reversal counts, so "0 of 1,000 resamples reversing sign" is reproducible
  3. Brier scores for every run x every target, filling the vasopressor gap
     for Runs C and D

Inputs (written by script2 / script3):
  post_drift_predictions.npz    -> labels, probs_a, probs_b, probs_c
  post_drift_predictions_d.npz  -> labels, probs_d       (optional)

Usage:  python script15_bca_and_brier.py [DIR]
"""
import sys
import math
import numpy as np
from pathlib import Path

# normal CDF / quantile, so the script needs numpy only
def _cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ppf(p):
    """Acklam's inverse normal CDF, refined by one Halley step (~1e-15 abs)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= ph:
        q = p - 0.5; r = q * q
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    e = _cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


LABELS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
B_BOOT = 1000
SEED   = 2

DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


# ── fast AUC + O(n log n) leave-one-out AUCs ─────────────────────────────────
def _auc_parts(y, s):
    """Return (U, n_pos, n_neg, c_pos, d_neg) for the Mann-Whitney form."""
    pos, neg = s[y == 1], s[y == 0]
    npos, nneg = len(pos), len(neg)
    sneg = np.sort(neg); spos = np.sort(pos)
    # c_i: negatives a positive beats, ties at 0.5
    lt = np.searchsorted(sneg, pos, side="left")
    le = np.searchsorted(sneg, pos, side="right")
    c_pos = lt + 0.5 * (le - lt)
    # d_j: positives beating a negative, ties at 0.5
    gt = npos - np.searchsorted(spos, neg, side="right")
    ge = npos - np.searchsorted(spos, neg, side="left")
    d_neg = gt + 0.5 * (ge - gt)
    return c_pos.sum(), npos, nneg, c_pos, d_neg


def auc(y, s):
    U, npos, nneg, _, _ = _auc_parts(y, s)
    return U / (npos * nneg)


def loo_auc(y, s):
    """AUC with each observation deleted in turn, in original index order."""
    U, npos, nneg, c_pos, d_neg = _auc_parts(y, s)
    out = np.empty(len(y))
    ip = np.where(y == 1)[0]
    ineg = np.where(y == 0)[0]
    out[ip]   = (U - c_pos) / ((npos - 1) * nneg)
    out[ineg] = (U - d_neg) / (npos * (nneg - 1))
    return out


# ── BCa for a paired AUROC difference ────────────────────────────────────────
def bca_paired(y, sx, sy, n_boot=B_BOOT, seed=SEED, alpha=0.05):
    theta = auc(y, sx) - auc(y, sy)

    rng = np.random.default_rng(seed)
    n = len(y)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == n:
            continue
        boot.append(auc(yb, sx[idx]) - auc(yb, sy[idx]))
    boot = np.asarray(boot)

    # bias correction
    prop = np.mean(boot < theta)
    prop = min(max(prop, 1.0 / (len(boot) + 1)), 1 - 1.0 / (len(boot) + 1))
    z0 = _ppf(prop)

    # acceleration from the exact jackknife
    jk = loo_auc(y, sx) - loo_auc(y, sy)
    dev = jk.mean() - jk
    denom = 6.0 * (np.sum(dev ** 2) ** 1.5)
    a = np.sum(dev ** 3) / denom if denom > 0 else 0.0

    def endpoint(z):
        zz = z0 + (z0 + z) / (1 - a * (z0 + z))
        return float(np.clip(_cdf(zz), 1e-6, 1 - 1e-6))

    lo = np.percentile(boot, 100 * endpoint(_ppf(alpha / 2)))
    hi = np.percentile(boot, 100 * endpoint(_ppf(1 - alpha / 2)))
    reversals = int(np.sum(np.sign(boot) != np.sign(theta)))
    return theta, lo, hi, reversals, len(boot), z0, a


def brier(y, p):
    return float(np.mean((p - y) ** 2))


# ── run ──────────────────────────────────────────────────────────────────────
d = np.load(DIR / "post_drift_predictions.npz")
labels = d["labels"]
P = {"A": d["probs_a"], "B": d["probs_b"], "C": d["probs_c"]}

dpath = DIR / "post_drift_predictions_d.npz"
if dpath.exists():
    dd = np.load(dpath)
    if np.array_equal(dd["labels"], labels):
        P["D"] = dd["probs_d"]
    else:
        print("WARNING: Run D labels differ from A/B/C — skipping Run D\n")

print(f"n = {len(labels)} stays | runs: {', '.join(sorted(P))}\n")

print("=" * 78)
print("BCa 95% CIs for paired AUROC differences")
print("=" * 78)
print(f"  {'contrast':<9}{'label':<20}{'delta':>9}  {'BCa 95% CI':<24}{'rev/B':>10}")
for x, y_ in [("B", "A"), ("B", "C"), ("C", "A"), ("D", "A")]:
    if x not in P or y_ not in P:
        continue
    for i, lbl in enumerate(LABELS):
        yt = labels[:, i].astype(int)
        if yt.sum() in (0, len(yt)):
            continue
        th, lo, hi, rev, nb, z0, a = bca_paired(yt, P[x][:, i], P[y_][:, i])
        print(f"  {x+'-'+y_:<9}{lbl:<20}{th:>+9.4f}  [{lo:+.4f}, {hi:+.4f}]      "
              f"{rev:>4d}/{nb:<5d}   (z0={z0:+.3f}, a={a:+.4f})")

print("\n" + "=" * 78)
print("Brier scores (lower is better); reference is the base rate pi(1-pi)")
print("=" * 78)
hdr = "  " + f"{'label':<20}" + "".join(f"{r:>10}" for r in sorted(P)) + f"{'base rate':>12}"
print(hdr)
for i, lbl in enumerate(LABELS):
    yt = labels[:, i].astype(int)
    pi = yt.mean()
    row = "  " + f"{lbl:<20}" + "".join(f"{brier(yt, P[r][:, i]):>10.4f}" for r in sorted(P))
    print(row + f"{pi * (1 - pi):>12.4f}")
print("\n  A Brier ABOVE the base-rate column means the run is worse than a")
print("  constant predictor at the observed prevalence.")
