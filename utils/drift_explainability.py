%%writefile utils/drift_explainability.py
"""
drift_explainability.py
═══════════════════════
Drop-in explainability module for script2_two_stream_model_v4.py

Answers the three clinical questions:
  1. WHICH features are drifting, and by how much? (PSI + KS + rate-shift)
  2. WHY is the model updating? (feature-importance × drift-magnitude attribution)
  3. HOW MUCH did adaptation help? (per-label, per-run metric delta with effect size)

USAGE — add these three call-sites in your existing script:
─────────────────────────────────────────────────────────────
  # ① After drift detection (end of Phase 2 / check_drift calls):
  from drift_explainability import DriftExplainer
  explainer = DriftExplainer(
      train_df=train_df,
      test_pre_df=test_pre,
      test_post_df=test_post,
      treat_features=TREATMENT_FEATURES,
      binary_cols=BINARY_COLS,
      label_cols=LABEL_COLS,
      psi_thresh=PSI_THRESH,
      save_path=SAVE_PATH,
  )
  drift_report = explainer.explain_drift()   # saves drift_report.json

  # ② After adaptation (end of Phase 2b, before Phase 3):
  explainer.explain_adaptation_update(
      model_before=model_B_pre,          # source weights
      model_after=model_B,               # adapted Run B
      adapt_train_ds=adapt_train_ds,
      device=device,
      run_tag="RunB",
  )
  explainer.explain_adaptation_update(
      model_before=model_C_pre,
      model_after=model_C,
      adapt_train_ds=adapt_train_ds,
      device=device,
      run_tag="RunC",
  )

  # ③ After Phase 3 (after results dict is built):
  explainer.explain_metric_gains(results, LABEL_COLS)  # appends to drift_report.json
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import json, copy, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
import datetime

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v) -> float:
    """Convert numpy scalars / NaN to JSON-safe Python float."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f) else round(f, 6)   # NaN → None
    except Exception:
        return None


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI using quantile binning — matches script7 psi_continuous exactly."""
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, bins + 1)
    breaks = np.unique(np.percentile(ref, quantiles))

    if len(breaks) < 2:
        return 0.0

    breaks[0] = -np.inf
    breaks[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=breaks)
    cur_counts, _ = np.histogram(cur, bins=breaks)

    p_ref = np.clip(ref_counts / len(ref), 1e-6, 1.0 - 1e-6)
    p_cur = np.clip(cur_counts / len(cur), 1e-6, 1.0 - 1e-6)

    psi_val = np.sum((p_ref - p_cur) * np.log(p_ref / p_cur))
    return float(psi_val) if np.isfinite(psi_val) else 0.0

def _psi_binary(ref: np.ndarray, cur: np.ndarray) -> float:
    """PSI for binary features — matches script7 psi_binary exactly."""
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    p_ref = np.clip(
        np.array([np.mean(ref == 0), np.mean(ref == 1)]), 1e-6, 1.0 - 1e-6)
    p_cur = np.clip(
        np.array([np.mean(cur == 0), np.mean(cur == 1)]), 1e-6, 1.0 - 1e-6)
    psi_val = np.sum((p_ref - p_cur) * np.log(p_ref / p_cur))
    return float(psi_val) if np.isfinite(psi_val) else 0.0

def _ks(ref: np.ndarray, cur: np.ndarray) -> Tuple[float, float]:
    """Two-sample KS test: returns (statistic, p-value)."""
    if len(ref) < 5 or len(cur) < 5:
        return 0.0, 1.0
    stat, pval = scipy_stats.ks_2samp(ref, cur)
    return float(stat), float(pval)


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_sd = np.sqrt(((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    if pooled_sd < 1e-9:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def _drift_severity(psi: float) -> str:
    if psi < 0.10:  return "stable"
    if psi < 0.20:  return "minor"
    if psi < 0.35:  return "moderate"
    return "severe"


def _get_first_row_array(df: pl.DataFrame, col: str) -> np.ndarray:
    """Return numpy array of feature values at hrs_from_admit == 0."""
    if col not in df.columns:
        return np.array([])
    filtered = df.filter(pl.col("hrs_from_admit") == 0)[col]
    return filtered.to_numpy().astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# GRADIENT-BASED TREATMENT FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _treatment_importance_from_model(
    model: nn.Module,
    dataset,
    device: torch.device,
    n_samples: int = 512,
) -> Dict[str, float]:
    """
    Permutation importance: for each treatment feature, zero it out and
    measure the average absolute change in logits across all labels.
    This tells us how much each treatment feature matters TO THE MODEL.
    Returns {feature_name: importance_score}.
    """
    from torch.utils.data import DataLoader, Subset
    import random

    model.eval()
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    idx = idx[:n_samples]
    sub = Subset(dataset, idx)
    loader = DataLoader(sub, batch_size=64, shuffle=False)

    # Collect baseline logits
    base_logits_list = []
    all_x_treats = []
    all_x_seqs   = []
    for x_seq, x_treat, _ in loader:
        x_seq, x_treat = x_seq.to(device), x_treat.to(device)
        base_logits_list.append(model(x_seq, x_treat).cpu())
        all_x_treats.append(x_treat.cpu())
        all_x_seqs.append(x_seq.cpu())

    base_logits = torch.cat(base_logits_list)   # (N, n_labels)
    all_x_t     = torch.cat(all_x_treats)
    all_x_s     = torch.cat(all_x_seqs)

    treat_cols = dataset.treat_cols
    importance = {}
    for fi, col in enumerate(treat_cols):
        perturbed = all_x_t.clone()
        feat_mean = all_x_t[:, fi].mean().item()
        perturbed[:, fi] = feat_mean # mean ablation

        perm_logits_list = []
        for i in range(0, len(all_x_s), 64):
            xs = all_x_s[i:i+64].to(device)
            xt = perturbed[i:i+64].to(device)
            perm_logits_list.append(model(xs, xt).cpu())
        perm_logits = torch.cat(perm_logits_list)

        delta = (base_logits - perm_logits).abs().mean().item()
        importance[col] = round(delta, 6)

    # Normalise to [0, 1]
    max_imp = max(importance.values()) if importance else 1.0
    if max_imp > 0:
        importance = {k: round(v / max_imp, 6) for k, v in importance.items()}
    return importance


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT-CHANGE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _weight_delta_report(
    before_state: dict,
    after_state: dict,
) -> Dict[str, dict]:
    """
    For each layer compare L2 norm of weight change and cosine similarity.
    Returns per-layer dict.
    """
    report = {}
    for key in before_state:
        if key not in after_state:
            continue
        wb = before_state[key].float()
        wa = after_state[key].float()
        delta = (wa - wb).norm().item()
        base_norm = wb.norm().item()
        cos_sim = float(
            torch.nn.functional.cosine_similarity(
                wb.reshape(1, -1), wa.reshape(1, -1)
            ).item()
        )
        rel_change = delta / (base_norm + 1e-9)
        def _get_group(key):
            if key.startswith("physio"): return "physio"
            if key.startswith("treat"):  return "treat"
            if key.startswith("fusion"): return "fusion"
            return "other"

        report[key] = {
            "l2_change":   round(delta, 6),
            "rel_change":  round(rel_change, 6),
            "cos_sim":     round(cos_sim, 6),
            "layer_group": _get_group(key)  # physio / treat / fusion
        }
    return report


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPLAINER CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DriftExplainer:
    """
    Central explainability object.  Collects drift statistics, model-update
    attribution, and metric deltas, then serialises everything to JSON for
    the dashboard.
    """

    def __init__(
        self,
        train_df: pl.DataFrame,
        test_pre_df: pl.DataFrame,
        test_post_df: pl.DataFrame,
        treat_features: List[str],
        binary_cols: set,
        label_cols: List[str],
        psi_thresh: float = 0.20,
        save_path: Path = Path("."),
    ):
        self.train_df      = train_df
        self.pre_df        = test_pre_df
        self.post_df       = test_post_df
        self.treat_feats   = treat_features
        self.binary_cols   = binary_cols
        self.label_cols    = label_cols
        self.psi_thresh    = psi_thresh
        self.save_path     = Path(save_path)
        self._report: dict = {
            "drift":       {},
            "attribution": {},
            "adaptation":  {},
            "metric_gains": {},
        }

    # ── 1. DRIFT EXPLANATION ──────────────────────────────────────────────────

    def explain_drift(self) -> dict:
        """
        Compute per-feature drift statistics comparing:
          train  →  pre-drift (sanity check — should be small)
          train  →  post-drift (the real drift)
          pre    →  post (temporal shift within test)

        Returns and saves drift_report.json.
        """
        print("\n" + "═"*60)
        print("DRIFT EXPLAINABILITY — Feature-level analysis")
        print("═"*60)

        continuous_feats = [
            c for c in self.treat_feats
            if c not in self.binary_cols and c != "age"
            and c in self.train_df.columns
        ]
        binary_feats = [
            c for c in self.treat_feats
            if c in self.binary_cols and c in self.train_df.columns
        ]

        feature_stats = {}

        # ── Continuous features ──────────────────────────────────────────────
        for col in continuous_feats:
            ref  = _get_first_row_array(self.train_df, col)
            pre  = _get_first_row_array(self.pre_df,   col)
            post = _get_first_row_array(self.post_df,  col)

            if len(ref) < 5 or len(post) < 5:
                continue

            psi_pre  = _psi(ref, pre)
            psi_post = _psi(ref, post)
            psi_temporal = _psi(pre, post) if len(pre) > 5 else None

            ks_stat, ks_p = _ks(ref, post)
            d = _cohen_d(ref, post)

            feature_stats[col] = {
                "type":         "continuous",
                "psi_train_pre":     _safe(psi_pre),
                "psi_train_post":    _safe(psi_post),
                "psi_pre_post":      _safe(psi_temporal),
                "ks_statistic":      _safe(ks_stat),
                "ks_pvalue":         _safe(ks_p),
                "cohen_d":           _safe(d),
                "mean_train":        _safe(ref.mean()),
                "mean_pre":          _safe(pre.mean() if len(pre) > 0 else None),
                "mean_post":         _safe(post.mean()),
                "std_train":         _safe(ref.std()),
                "std_post":          _safe(post.std()),
                "drift_severity":    _drift_severity(psi_post),
                "drifted":           psi_post > self.psi_thresh,
                "significant":       ks_p < 0.05,
            }

        # ── Binary features ──────────────────────────────────────────────────
        for col in binary_feats:
            ref  = _get_first_row_array(self.train_df, col)
            pre  = _get_first_row_array(self.pre_df,   col)
            post = _get_first_row_array(self.post_df,  col)

            if len(ref) < 5 or len(post) < 5:
                continue

            rate_train = float(ref.mean())
            rate_pre   = float(pre.mean()) if len(pre) > 0 else None
            rate_post  = float(post.mean())
            abs_shift  = abs(rate_post - rate_train)
            rel_shift  = abs_shift / (rate_train + 1e-6)

            psi_bin = _psi_binary(ref, post)

            # Chi-squared test for rate shift
            try:
                n_ref, n_post = len(ref), len(post)
                contingency = np.array([
                    [int(ref.sum()),  n_ref  - int(ref.sum())],
                    [int(post.sum()), n_post - int(post.sum())],
                ])
                chi2, chi_p, _, _ = scipy_stats.chi2_contingency(contingency)
            except Exception:
                chi2, chi_p = 0.0, 1.0

            feature_stats[col] = {
                "type":         "binary",
                "psi":         _safe(psi_bin),   
                "rate_train":   _safe(rate_train),
                "rate_pre":     _safe(rate_pre),
                "rate_post":    _safe(rate_post),
                "abs_shift":    _safe(abs_shift),
                "rel_shift":    _safe(rel_shift),
                "chi2":         _safe(chi2),
                "chi2_pvalue":  _safe(chi_p),
                "drifted": abs_shift > 0.05 or rel_shift > 0.20,
                "drift_severity": (
                    "severe"   if rel_shift > 0.50 or abs_shift > 0.15 else
                    "moderate" if rel_shift > 0.25 or abs_shift > 0.08 else
                    "minor"    if rel_shift > 0.10 or abs_shift > 0.03 else
                    "stable"
                ),
                "significant":  chi_p < 0.05,
            }

        # After both feature loops, before summary calculation
        n_tests = len(feature_stats)
        for col, stats in feature_stats.items():
            if stats["type"] == "continuous":
                stats["significant_corrected"] = stats["ks_pvalue"] < (0.05 / n_tests)
            else:
                stats["significant_corrected"] = stats["chi2_pvalue"] < (0.05 / n_tests)
        # ── Summary ─────────────────────────────────────────────────────────
        n_drifted_cont = sum(
            1 for k, v in feature_stats.items()
            if v["type"] == "continuous" and v["drifted"]
        )
        n_drifted_bin = sum(
            1 for k, v in feature_stats.items()
            if v["type"] == "binary" and v["drifted"]
        )
        n_significant = sum(1 for v in feature_stats.values() if v["significant"])

        psi_values = [
            v["psi_train_post"]
            for v in feature_stats.values()
            if v["type"] == "continuous" and v.get("psi_train_post") is not None
        ]
        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0

        # Top drifted by PSI / abs_shift
        top_drifted = sorted(
            feature_stats.items(),
            key=lambda x: (
                x[1].get("psi_train_post") or x[1].get("abs_shift") or 0
            ),
            reverse=True,
        )[:15]

        summary = {
            "n_features_analysed":   len(feature_stats),
            "n_drifted_continuous":  n_drifted_cont,
            "n_drifted_binary":      n_drifted_bin,
            "n_statistically_significant": n_significant,
            "mean_psi":              _safe(mean_psi),
            "overall_drift_level":   _drift_severity(mean_psi),
            "top_drifted_features":  [k for k, _ in top_drifted],
        }

        self._report["drift"] = {
            "summary":  summary,
            "features": feature_stats,
        }

        # Console summary
        # print(f"\n  Features analysed : {len(feature_stats)}")
        # print(f"  Continuous drifted: {n_drifted_cont}")
        # print(f"  Binary drifted    : {n_drifted_bin}")
        # print(f"  Statistically sig : {n_significant}")
        # print(f"  Mean PSI          : {mean_psi:.4f}  → {_drift_severity(mean_psi).upper()}")
        # print(f"\n  Top drifted features (train → post-drift):")
        # for col, st in top_drifted[:10]:
        #     if st["type"] == "continuous":
        #         print(f"    {col:<35} PSI={st['psi_train_post']:.4f}  "
        #               f"d={st['cohen_d']:.3f}  [{st['drift_severity']}]")
        #     else:
        #         print(f"    {col:<35} Δrate={st['abs_shift']:.4f}  "
        #               f"χ²p={st['chi2_pvalue']:.4f}  [{st['drift_severity']}]")
        print("\n" + "="*70)
        print("DETAILED DRIFT ANALYSIS & EXPLANATION")
        print("="*70)
        print(self.generate_human_summary())

        self._save_report()
        return self._report["drift"]

    # ── 2. ATTRIBUTION: which drifted features matter to the model ────────────

    def explain_adaptation_update(
        self,
        model_before: nn.Module,
        model_after:  nn.Module,
        adapt_train_ds,
        device: torch.device,
        run_tag: str = "RunB",
    ) -> dict:
        """
        After adaptation, quantify:
          a) Which layers changed the most (weight-delta report)
          b) Which treatment features matter to the adapted model
          c) Combined attribution: drift × importance
        """
        print(f"\n{'═'*60}")
        print(f"ADAPTATION EXPLANATION — {run_tag}")
        print(f"{'═'*60}")

        # a) Weight changes
        print("  Computing weight deltas...")
        wd = _weight_delta_report(
            model_before.state_dict(),
            model_after.state_dict(),
        )

        # Group by stream
        stream_deltas: Dict[str, List[float]] = {
            "physio": [], "treat": [], "fusion": []
        }
        for key, info in wd.items():
            g = info["layer_group"]
            if g in stream_deltas:
                stream_deltas[g].append(info["rel_change"])

        stream_summary = {
            g: {
                "mean_rel_change": round(float(np.mean(v)), 6) if v else 0.0,
                "max_rel_change":  round(float(np.max(v)),  6) if v else 0.0,
                "n_layers": len(v),
            }
            for g, v in stream_deltas.items()
        }

        print(f"  Layer-stream weight changes:")
        for g, info in stream_summary.items():
            print(f"    {g:<10} mean_rel_Δ={info['mean_rel_change']:.4f}  "
                  f"max_rel_Δ={info['max_rel_change']:.4f}")

        # b) Treatment feature importance from adapted model
        print("  Computing treatment feature importance (permutation)...")
        importance = _treatment_importance_from_model(
            model_after, adapt_train_ds, device, n_samples=512
        )

        # c) Combined attribution: drift_score × model_importance
        feature_stats = self._report.get("drift", {}).get("features", {})
        attribution = {}
        for col, imp in importance.items():
            fs = feature_stats.get(col, {})
            if fs.get("type") == "continuous":
                drift_score = fs.get("psi_train_post") or 0.0
            else:
                drift_score = fs.get("abs_shift") or 0.0

            attribution[col] = {
                "model_importance":   round(imp, 6),
                "drift_score":        round(float(drift_score), 6),
                "combined_score":     round(float(imp) * float(drift_score), 8),
                "drift_severity":     fs.get("drift_severity", "unknown"),
                "feature_type":       fs.get("type", "unknown"),
            }

        # Top attributions
        top_attr = sorted(
            attribution.items(),
            key=lambda x: x[1]["combined_score"],
            reverse=True,
        )[:15]

        print(f"\n  Top treatment features driving adaptation ({run_tag}):")
        print(f"  {'Feature':<35} {'Importance':>10} {'Drift':>8} {'Combined':>10}")
        for col, a in top_attr[:10]:
            print(f"  {col:<35} {a['model_importance']:>10.4f} "
                  f"{a['drift_score']:>8.4f} {a['combined_score']:>10.6f}")

        self._report["adaptation"][run_tag] = {
            "weight_deltas":   wd,
            "stream_summary":  stream_summary,
            "feature_importance": importance,
            "attribution":     attribution,
            "top_drivers":     [k for k, _ in top_attr],
        }
        self._save_report()
        return self._report["adaptation"][run_tag]

    # ── 3. METRIC GAINS ───────────────────────────────────────────────────────
    def explain_metric_gains(
        self,
        results: dict,
        label_cols: List[str],
    ) -> dict:
        """
        Summarise how much each run gained vs Run A on each split,
        with Cohen's d effect size and a plain-English interpretation.
        """
        print(f"\n{'═'*60}")
        print("METRIC GAIN EXPLANATION — Run A vs B vs C")
        print(f"{'═'*60}")

        gains = {}
        for split, r in results.items():
            gains[split] = {}
            for lbl in label_cols:
                a_roc = r["A"].get(f"{lbl}_auroc")
                b_roc = r["B"].get(f"{lbl}_auroc")
                c_roc = r["C"].get(f"{lbl}_auroc")

                def delta_interp(d):
                    if d is None: return "n/a"
                    if abs(d) < 0.005: return "negligible"
                    if abs(d) < 0.015: return "small"
                    if abs(d) < 0.03:  return "moderate"
                    return "large"

                d_b = (_safe(b_roc - a_roc)
                       if b_roc is not None and a_roc is not None else None)
                d_c = (_safe(c_roc - a_roc)
                       if c_roc is not None and a_roc is not None else None)

                gains[split][lbl] = {
                    "A_auroc": _safe(a_roc),
                    "B_auroc": _safe(b_roc),
                    "C_auroc": _safe(c_roc),
                    "B_minus_A": d_b,
                    "C_minus_A": d_c,
                    "B_minus_A_interp": delta_interp(d_b),
                    "C_minus_A_interp": delta_interp(d_c),
                    "winner": (
                        "A" if (b_roc or 0) <= (a_roc or 0) and (c_roc or 0) <= (a_roc or 0)
                        else "B" if (b_roc or 0) >= (c_roc or 0) else "C"
                    ),
                }

        # Post-drift macro summary
        post_key = next((k for k in results if "Post" in k and "Pre" not in k), None)
        macro = {}
        if post_key and post_key in gains:
            for run in ["B", "C"]:
                deltas = [
                    gains[post_key][lbl][f"{run}_minus_A"]
                    for lbl in label_cols
                    if gains[post_key][lbl][f"{run}_minus_A"] is not None
                ]
                macro[run] = {
                    "mean_auroc_delta": _safe(np.mean(deltas)) if deltas else None,
                    "max_auroc_delta":  _safe(np.max(deltas))  if deltas else None,
                    "labels_improved":  sum(1 for d in deltas if d > 0),
                    "n_labels":         len(deltas),
                }

        print(f"\n  Post-drift AUROC gains vs frozen Run A:")
        if post_key and post_key in gains:
            for lbl in label_cols:
                g = gains[post_key][lbl]
                print(f"  {lbl:<28} "
                      f"B Δ={g['B_minus_A']:+.4f} ({g['B_minus_A_interp']})  "
                      f"C Δ={g['C_minus_A']:+.4f} ({g['C_minus_A_interp']})"
                      f"  → winner: {g['winner']}")

        self._report["metric_gains"] = {
            "per_split":   gains,
            "macro":       macro,
            "post_split_key": post_key,
        }
        self._save_report()
        return self._report["metric_gains"]

    # ── SAVE ──────────────────────────────────────────────────────────────────

    def get_report(self) -> dict:
        return self._report

    def generate_human_summary(self) -> str:
        """Generates a dynamic natural language clinical summary of the data drift and adaptation."""
        drift_data = self._report.get("drift", {})
        if not drift_data or "features" not in drift_data:
            return "No drift analysis available."

        feature_stats = drift_data["features"]
        
        # Sort features by severity and grab the top 5
        top_features = sorted(
            feature_stats.items(),
            key=lambda x: x[1].get("psi_train_post") or x[1].get("abs_shift") or 0.0,
            reverse=True
        )[:5]

        is_drifting = drift_data["summary"].get("n_drifted_continuous", 0) > 0 or drift_data["summary"].get("n_drifted_binary", 0) > 0
        
        if not is_drifting:
            return "✅ Clinical practices appear stable between the two time periods. No model adaptation is required."

        lines = [
            "🚨 CLINICAL PRACTICE SHIFT DETECTED",
            "Between the training period (2014-2016) and the new data (2020-2022), patient treatment patterns have changed significantly."
        ]

        # ── DYNAMIC EXPLANATION OF "HOW" AND "WHY" ──
        adapt_data = self._report.get("adaptation", {})
        
        # If adaptation has happened (Audit Log phase), explain the mechanics dynamically
        if "RunB" in adapt_data:
            run_b = adapt_data["RunB"]
            
            # DYNAMIC Q5: How is it updating? (Reads actual layer weight deltas)
            streams = run_b.get("stream_summary", {})
            physio_change = streams.get("physio", {}).get("mean_rel_change", 0)
            treat_change = streams.get("treat", {}).get("mean_rel_change", 0)
            
            if physio_change < 1e-4 and treat_change > 0:
                how_text = "To maintain accuracy safely, the model locked its physiological representation and exclusively updated its treatment and fusion layers."
            elif physio_change > 0 and treat_change > 0:
                how_text = "To maintain accuracy, the model globally updated all layers (full fine-tuning) to adjust to the new data."
            else:
                how_text = "The model attempted adaptation but detected minimal necessary weight updates."
            
            # DYNAMIC Q6: Why will this result in better accuracy? (Reads attribution scores)
            top_drivers = run_b.get("top_drivers", [])
            if len(top_drivers) >= 2:
                d1 = top_drivers[0].replace('has_', '').replace('_obs', '').replace('_', ' ').title()
                d2 = top_drivers[1].replace('has_', '').replace('_obs', '').replace('_', ' ').title()
                why_text = f"This restores accuracy by forcing the network to relearn the specific features that carry high predictive importance but suffered from severe real-world drift (most notably {d1} and {d2})."
            else:
                why_text = "This restores accuracy by realigning the model's internal weights with the new clinical reality."
                
            lines.append(f"{how_text} {why_text}\n")
        else:
            # Pre-adaptation fallback (for Phase 2 console print)
            lines.append("The model will now trigger an adaptation phase to account for these shifts.\n")

        lines.append("Key clinical changes driving this update:")

        for feat, stats in top_features:
            clean_name = feat.replace('has_', '').replace('_obs', '').replace('_', ' ').title()
            
            if stats["type"] == "continuous":
                m_train = stats.get("mean_train") or 0.0
                m_post = stats.get("mean_post") or 0.0
                psi_val = stats.get("psi_train_post") or 0.0
                
                delta = m_post - m_train
                direction = "increased" if delta > 0 else "decreased"
                lines.append(f"• {clean_name}: Average value {direction} from {m_train:.2f} to {m_post:.2f} (Severe Drift PSI: {psi_val:.2f})")
            else:
                r_train = stats.get("rate_train") or 0.0
                r_post = stats.get("rate_post") or 0.0
                
                shift = (r_post - r_train) * 100
                direction = "increased" if shift > 0 else "dropped"
                lines.append(f"• {clean_name}: Usage {direction} by {abs(shift):.1f}% in the newer data.")

        return "\n".join(lines)

    

    def _save_report(self):
        out = self.save_path / "drift_report.json"
        self._report["provenance"] = {
            "generated_at": datetime.datetime.utcnow().isoformat(),
            "psi_threshold": self.psi_thresh,
            "n_label_cols": len(self.label_cols),
            "n_treat_features": len(self.treat_feats),
        }
        with open(out, "w") as f:
            json.dump(self._report, f, indent=2, default=str)