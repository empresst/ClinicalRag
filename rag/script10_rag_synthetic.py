"""
script10_rag_synthetic.py
═════════════════════════
RAG evaluation on expanded synthetic corpus.
Proof-of-concept retrieval before real PubMed evaluation.

Requires:
  - train/test parquet files from script1
  - two_stream_models.pt from script2
  - eval_post_stays.json from script3
"""

import numpy as np
import polars as pl
import json
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, SingleStreamDataset, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import PhysiologyStream, TreatmentStream, FusionHead, SingleStreamModel, TwoStreamModel, SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM, BATCH_SIZE, LSTM_LAYERS, DROPOUT, LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 42, 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")
s2 = Path("/kaggle/input/datasets/fatematamanna/ptfiles")
SAVE_PATH  = Path("/kaggle/working")
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
TOP_K_DOCS = 5
N_CASES_PER_CONDITION = 5

# ── DATA LOADING ───────────────────────────────────────────────────────────────

print("Loading saved models...")
ckpt = torch.load(SAVE_PATH / "two_stream_models.pt", map_location=device, weights_only=False)
seq_dim, treat_dim, n_targets = ckpt["seq_dim"], ckpt["treat_dim"], ckpt["n_targets"]
train_stats = ckpt["train_stats"]

model_A = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_A.load_state_dict(ckpt["run_a"]); model_A.eval()
model_B = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_B.load_state_dict(ckpt["run_b"]); model_B.eval()
print(f"Models loaded: seq={seq_dim}, treat={treat_dim}, targets={n_targets}")


train_df = normalize(load_enriched_split(BASE_PATH, "train", SEQ_FEATURES, TREATMENT_FEATURES),train_stats)
val_df = normalize(load_enriched_split(BASE_PATH, "val", SEQ_FEATURES, TREATMENT_FEATURES),train_stats)
test_df = normalize(load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES),train_stats)

test_pre  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

# Reproduce the held-out split from script2
post_stays = test_post.filter(pl.col("hrs_from_admit") == 0).sort("intime")["stay_id"].to_list()
n_total_post = len(post_stays)
n_adapt_train = int(n_total_post * 0.30)
n_adapt_val   = int(n_total_post * 0.10)
eval_post_stays = post_stays[n_adapt_train + n_adapt_val:]
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))

print(f"Test-Pre: {test_pre['stay_id'].n_unique()} stays")
print(f"Test-Post held-out: {len(eval_post_stays)} stays")


# ══════════════════════════════════════════════════════════════════════════════
# ADDITION 3: EXPANDED RAG CORPUS (~200+ passages)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ADDITION 3: Expanded RAG Corpus")
print("="*70)
CORPUS_PATH = Path(SAVE_PATH / "data/expanded_corpus.json") # Update path as needed

# Load the data
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    EXPANDED_CORPUS = json.load(f)

print(f"Expanded corpus: {len(EXPANDED_CORPUS)} passages")
n_relevant_exp = sum(1 for g in EXPANDED_CORPUS if g["id"].startswith("ICU_"))
n_noise_exp = len(EXPANDED_CORPUS) - n_relevant_exp
print(f"  Relevant ICU: {n_relevant_exp}, Noise: {n_noise_exp}")

# ── Embed expanded corpus ──
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))

corpus_texts_exp = [g["topic"].replace("_", " ") + ". " + g["text"] for g in EXPANDED_CORPUS]
print("Embedding expanded corpus...")
corpus_embeddings_exp = embedder.encode(corpus_texts_exp, convert_to_numpy=True,
                                         show_progress_bar=True, batch_size=32,
                                         normalize_embeddings=True)
print(f"  Shape: {corpus_embeddings_exp.shape}")

def retrieve_guidelines_expanded(query_terms, era_filter=None, k=TOP_K_DOCS):
    query = " ".join(query_terms) if isinstance(query_terms, list) else query_terms
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = (corpus_embeddings_exp @ q_emb.T).ravel()
    candidates = sorted(zip(range(len(EXPANDED_CORPUS)), sims), key=lambda x: x[1], reverse=True)
    results = []
    for idx, sim in candidates:
        g = EXPANDED_CORPUS[idx]
        if era_filter and g["era"] not in (era_filter, "all"):
            continue
        results.append({
            "guideline_id": g["id"], "topic": g["topic"], "era": g["era"],
            "text": g["text"], "source": g["source"], "relevance_score": float(sim),
        })
        if len(results) >= k:
            break
    return results

# ── Updated ground truth (same IDs, expanded relevant set) ──
GROUND_TRUTH_EXPANDED = {
    "label_vasopressor": {
        "post_covid": {"ICU_ssc_2021_vaso_first", "ICU_vanish_trial", "ICU_vasopressor_choice",
                       "ICU_sepsis3_definition", "ICU_abx_timing", "ICU_andromeda_shock",
                       "ICU_hour1_bundle", "ICU_ssc_corticosteroids", "ICU_map_target_65",
                       "ICU_balanced_crystalloid"},
        "pre_covid":  {"ICU_ssc_2016_fluid", "ICU_vasopressor_choice", "ICU_sepsis3_definition",
                       "ICU_qsofa", "ICU_vanish_trial", "ICU_early_goal_directed",
                       "ICU_map_target_65"},
    },
    "label_intubation": {
        "post_covid": {"ICU_covid_permissive_hypoxemia", "ICU_recovery_dexamethasone",
                       "ICU_hfnc_first", "ICU_lung_protective", "ICU_rsi_etomidate",
                       "ICU_prone_positioning", "ICU_driving_pressure", "ICU_nmba_ards",
                       "ICU_covid_awake_prone", "ICU_sbt_protocol", "ICU_peep_titration"},
        "pre_covid":  {"ICU_lung_protective", "ICU_rsi_etomidate", "ICU_prone_positioning",
                       "ICU_driving_pressure", "ICU_nmba_ards", "ICU_sbt_protocol",
                       "ICU_peep_titration", "ICU_vili_prevention"},
    },
    "label_septic_shock": {
        "post_covid": {"ICU_ssc_2021_vaso_first", "ICU_sepsis3_definition", "ICU_abx_timing",
                       "ICU_qsofa", "ICU_andromeda_shock", "ICU_vasopressor_choice",
                       "ICU_hour1_bundle", "ICU_ssc_corticosteroids", "ICU_procalcitonin_abx",
                       "ICU_fluid_conservative"},
        "pre_covid":  {"ICU_ssc_2016_fluid", "ICU_sepsis3_definition", "ICU_qsofa",
                       "ICU_abx_timing", "ICU_vasopressor_choice", "ICU_early_goal_directed",
                       "ICU_albumin_sepsis"},
    },
}

def compute_retrieval_metrics(retrieved, relevant_set, k_values=(1, 3, 5)):
    ids = [r["guideline_id"] for r in retrieved]
    metrics = {}
    for k in k_values:
        top_k = set(ids[:k])
        metrics[f"hit@{k}"] = 1.0 if top_k & relevant_set else 0.0
        metrics[f"recall@{k}"] = len(top_k & relevant_set) / max(len(relevant_set), 1)
    rr = 0.0
    for i, gid in enumerate(ids):
        if gid in relevant_set:
            rr = 1.0 / (i + 1)
            break
    metrics["reciprocal_rank"] = rr
    # Precision@k
    for k in k_values:
        top_k = set(ids[:k])
        metrics[f"precision@{k}"] = len(top_k & relevant_set) / k
    return metrics

# ── Feature-to-query text (reuse from script3) ──
def features_to_query_text(top_features, label_name):
    label_text = {
        "label_vasopressor": "vasopressor need septic shock hemodynamic support",
        "label_intubation":  "mechanical ventilation intubation respiratory failure",
        "label_septic_shock": "septic shock sepsis organ dysfunction lactate",
    }
    feat_text = {
        "lactate": "elevated lactate perfusion",
        "map_invasive": "low mean arterial pressure hypotension",
        "spo2": "hypoxemia oxygen saturation",
        "resp_rate": "tachypnea respiratory rate",
        "creatinine": "acute kidney injury creatinine",
        "wbc": "leukocytosis infection white blood cells",
        "has_norepinephrine_obs": "norepinephrine vasopressor",
        "has_vasopressin_obs": "vasopressin refractory shock",
        "total_crystalloid_ml": "fluid resuscitation crystalloid volume",
        "max_fio2_obs": "high FiO2 oxygen requirement",
        "high_fio2_flag": "severe hypoxemia high oxygen",
        "max_peep_obs": "PEEP ventilator settings",
        "early_steroid": "corticosteroid dexamethasone",
        "early_antibiotic": "empiric antibiotic sepsis",
        "has_propofol_midaz_obs": "sedation intubation preparation",
        "time_to_first_vaso_hrs": "early vasopressor timing",
    }
    terms = [label_text.get(label_name, label_name.replace("label_", "").replace("_", " "))]
    for f in top_features:
        clean = f["feature"].replace("physio:", "").replace("treat:", "").replace("_mask", "")
        if clean in feat_text:
            terms.append(feat_text[clean])
    return ". ".join(terms)

# ── Run expanded RAG evaluation ──
print("\nEvaluating expanded RAG retrieval...")

# Use same integrated-gradients explain from script3 to get feature-driven queries
def integrated_gradients(model, x_seq, x_treat, target_idx, steps=20):
    model.eval()
    x_seq, x_treat = x_seq.to(device), x_treat.to(device)
    baseline_seq = torch.zeros_like(x_seq)
    baseline_treat = torch.zeros_like(x_treat)
    seq_grads_sum = torch.zeros_like(x_seq)
    treat_grads_sum = torch.zeros_like(x_treat)
    for alpha in np.linspace(0, 1, steps):
        interp_seq = (baseline_seq + alpha * (x_seq - baseline_seq)).requires_grad_(True)
        interp_treat = (baseline_treat + alpha * (x_treat - baseline_treat)).requires_grad_(True)
        logits = model(interp_seq, interp_treat)
        target = logits[:, target_idx].sum()
        seq_grad, treat_grad = torch.autograd.grad(target, [interp_seq, interp_treat])
        seq_grads_sum += seq_grad; treat_grads_sum += treat_grad
    avg_seq = seq_grads_sum / steps; avg_treat = treat_grads_sum / steps
    seq_attr = ((x_seq - baseline_seq) * avg_seq).cpu().numpy()
    treat_attr = ((x_treat - baseline_treat) * avg_treat).cpu().numpy()
    return seq_attr, treat_attr

TOP_K_FEATURES = 8

def explain_prediction(model, seq_input, treat_input, label_idx, seq_cols, treat_cols):
    model.eval()
    with torch.no_grad():
        logits = model(seq_input, treat_input)
        prob = torch.sigmoid(logits)[0, label_idx].item()
    seq_attr, treat_attr = integrated_gradients(model, seq_input, treat_input, label_idx)
    seq_attr_per_feature = seq_attr[0].sum(axis=0)
    treat_attr_per_feature = treat_attr[0]
    all_importances = ([("physio:" + n, v) for n, v in zip(seq_cols, seq_attr_per_feature)] +
                       [("treat:" + n, v) for n, v in zip(treat_cols, treat_attr_per_feature)])
    all_importances.sort(key=lambda x: abs(x[1]), reverse=True)
    top = all_importances[:TOP_K_FEATURES]
    return {
        "predicted_probability": prob,
        "top_features": [{"feature": n, "contribution": float(v),
                          "direction": "increases" if v > 0 else "decreases"} for n, v in top],
    }

# Build datasets
ds_pre  = ICUDataset(test_pre,  SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
ds_post = ICUDataset(eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)

def select_cases_by_label(dataset, label_idx, n_positive=N_CASES_PER_CONDITION):
    pos_indices = [i for i in range(len(dataset)) if dataset.labels[i, label_idx] == 1]
    if len(pos_indices) > n_positive:
        rng = np.random.RandomState(SEED)
        pos_indices = rng.choice(pos_indices, n_positive, replace=False).tolist()
    return pos_indices

# Post-drift with expanded corpus
expanded_post_metrics = {lbl: [] for lbl in LABEL_COLS}
for label_idx, label_name in enumerate(LABEL_COLS):
    cases = select_cases_by_label(ds_post, label_idx, N_CASES_PER_CONDITION)
    for case_idx in cases:
        seq, treat, lbl = ds_post[case_idx]
        seq_b = seq.unsqueeze(0).to(device)
        treat_b = treat.unsqueeze(0).to(device)
        exp = explain_prediction(model_B, seq_b, treat_b, label_idx, ds_post.seq_cols, ds_post.treat_cols)
        query = features_to_query_text(exp["top_features"], label_name)
        retrieved = retrieve_guidelines_expanded(query, era_filter="post_covid", k=TOP_K_DOCS)
        relevant = GROUND_TRUTH_EXPANDED[label_name]["post_covid"]
        m = compute_retrieval_metrics(retrieved, relevant)
        expanded_post_metrics[label_name].append(m)

# Pre-drift with expanded corpus
expanded_pre_metrics = {lbl: [] for lbl in LABEL_COLS}
for label_idx, label_name in enumerate(LABEL_COLS):
    cases = select_cases_by_label(ds_pre, label_idx, N_CASES_PER_CONDITION)
    for case_idx in cases:
        seq, treat, lbl = ds_pre[case_idx]
        seq_b = seq.unsqueeze(0).to(device)
        treat_b = treat.unsqueeze(0).to(device)
        exp = explain_prediction(model_A, seq_b, treat_b, label_idx, ds_pre.seq_cols, ds_pre.treat_cols)
        query = features_to_query_text(exp["top_features"], label_name)
        retrieved = retrieve_guidelines_expanded(query, era_filter="pre_covid", k=TOP_K_DOCS)
        relevant = GROUND_TRUTH_EXPANDED[label_name]["pre_covid"]
        m = compute_retrieval_metrics(retrieved, relevant)
        expanded_pre_metrics[label_name].append(m)

# Aggregate
def aggregate_metrics(metrics_dict):
    agg = {}
    for label, case_list in metrics_dict.items():
        if not case_list: continue
        agg[label] = {}
        for key in ["hit@1", "hit@3", "hit@5", "recall@3", "recall@5",
                     "precision@1", "precision@3", "precision@5", "reciprocal_rank"]:
            agg[label][key] = float(np.mean([m[key] for m in case_list]))
    return agg

exp_post_agg = aggregate_metrics(expanded_post_metrics)
exp_pre_agg  = aggregate_metrics(expanded_pre_metrics)

print("\n" + "="*70)
print("EXPANDED RAG RETRIEVAL RESULTS")
print(f"Corpus: {len(EXPANDED_CORPUS)} ({n_relevant_exp} relevant + {n_noise_exp} noise)")
print("="*70)

def print_rag_table(title, agg, metrics_dict):
    print(f"\n{title}")
    print(f"  {'Label':<22} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'P@3':>8} {'R@3':>8} {'MRR':>8}")
    print("  " + "-"*64)
    for lbl in LABEL_COLS:
        if lbl in agg:
            m = agg[lbl]
            print(f"  {lbl:<22} {m['hit@1']:>8.3f} {m['hit@3']:>8.3f} {m['hit@5']:>8.3f} "
                  f"{m['precision@3']:>8.3f} {m['recall@3']:>8.3f} {m['reciprocal_rank']:>8.3f}")
    all_cases = []
    for cases in metrics_dict.values():
        all_cases.extend(cases)
    if all_cases:
        overall = {k: float(np.mean([m[k] for m in all_cases]))
                   for k in ["hit@1","hit@3","hit@5","precision@3","recall@3","reciprocal_rank"]}
        print(f"  {'OVERALL':<22} {overall['hit@1']:>8.3f} {overall['hit@3']:>8.3f} "
              f"{overall['hit@5']:>8.3f} {overall['precision@3']:>8.3f} {overall['recall@3']:>8.3f} "
              f"{overall['reciprocal_rank']:>8.3f}")
        return overall
    return {}

exp_post_overall = print_rag_table("Post-drift (expanded corpus)", exp_post_agg, expanded_post_metrics)
exp_pre_overall  = print_rag_table("Pre-drift (expanded corpus)", exp_pre_agg, expanded_pre_metrics)

# Save expanded RAG results
expanded_rag_results = {
    "corpus_size": len(EXPANDED_CORPUS),
    "n_relevant": n_relevant_exp, "n_noise": n_noise_exp,
    "embedding_model": EMBEDDING_MODEL,
    "post_drift_per_label": exp_post_agg, "post_drift_overall": exp_post_overall,
    "pre_drift_per_label": exp_pre_agg,   "pre_drift_overall": exp_pre_overall,
}
with open(SAVE_PATH / "expanded_rag_metrics.json", "w") as f:
    json.dump(expanded_rag_results, f, indent=2)
print(f"\nSaved → {SAVE_PATH / 'expanded_rag_metrics.json'}")

# ── LOAD METRICS FROM SCRIPT 5 ──
import json

# Adjust this path to wherever your script5 saved the JSON file
RESULTS_PATH = Path("/kaggle/working/full_comparison_results.json") 

print("Loading baseline and bootstrap metrics from script 5...")
with open(RESULTS_PATH, "r") as f:
    comp_results = json.load(f)

# Extract exactly the variables your plotting code is looking for:
auroc_boot = comp_results["bootstrap_auroc"]
auprc_boot = comp_results["bootstrap_auprc"]
xgb_results = comp_results["xgboost_source"]
xgb_adapted_results = comp_results["xgboost_adapted"]

# Extract the Two-Stream Run A and Run B post-drift metrics
ts_a_post = comp_results["two_stream_run_a"]["test_post"]
ts_b_post = comp_results["two_stream_run_b"]["test_post"]
# ══════════════════════════════════════════════════════════════════════════════
# COMBINED FIGURE: Forest plot of bootstrap CIs
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating forest plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# AUROC forest plot
labels_short = [l.replace("label_", "").replace("_", " ").title() for l in LABEL_COLS]
y_pos = np.arange(len(LABEL_COLS))
deltas = [auroc_boot[l]["delta"] for l in LABEL_COLS]
ci_lows = [auroc_boot[l]["ci_low"] for l in LABEL_COLS]
ci_highs = [auroc_boot[l]["ci_high"] for l in LABEL_COLS]
p_vals = [auroc_boot[l]["p_value"] for l in LABEL_COLS]

colors = ["#2ca02c" if ci_lows[i] > 0 else "#6b7280" for i in range(len(LABEL_COLS))]
ax1.barh(y_pos, deltas, xerr=[[d - cl for d, cl in zip(deltas, ci_lows)],
                                [ch - d for d, ch in zip(deltas, ci_highs)]],
         color=colors, alpha=0.7, capsize=5, height=0.5, edgecolor="black", linewidth=0.5)
ax1.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=1)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels_short)
ax1.set_xlabel("ΔAUROC (Run B − Run A)")
ax1.set_title("AUROC Improvement with 95% CI\n(Post-drift, N=1000 bootstrap)")
for i, (d, p) in enumerate(zip(deltas, p_vals)):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax1.text(max(deltas) * 1.3, i, f"Δ={d:+.4f} (p={p:.3f}) {sig}", va="center", fontsize=9)
ax1.grid(True, alpha=0.3, axis="x")

# AUPRC forest plot
deltas_pr = [auprc_boot[l]["delta"] for l in LABEL_COLS]
ci_lows_pr = [auprc_boot[l]["ci_low"] for l in LABEL_COLS]
ci_highs_pr = [auprc_boot[l]["ci_high"] for l in LABEL_COLS]
p_vals_pr = [auprc_boot[l]["p_value"] for l in LABEL_COLS]

colors_pr = ["#2ca02c" if ci_lows_pr[i] > 0 else "#6b7280" for i in range(len(LABEL_COLS))]
ax2.barh(y_pos, deltas_pr, xerr=[[d - cl for d, cl in zip(deltas_pr, ci_lows_pr)],
                                   [ch - d for d, ch in zip(deltas_pr, ci_highs_pr)]],
         color=colors_pr, alpha=0.7, capsize=5, height=0.5, edgecolor="black", linewidth=0.5)
ax2.axvline(x=0, color="red", linestyle="--", alpha=0.7, linewidth=1)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels_short)
ax2.set_xlabel("ΔAUPRC (Run B − Run A)")
ax2.set_title("AUPRC Improvement with 95% CI\n(Post-drift, N=1000 bootstrap)")
for i, (d, p) in enumerate(zip(deltas_pr, p_vals_pr)):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax2.text(max(deltas_pr) * 1.3, i, f"Δ={d:+.4f} (p={p:.3f}) {sig}", va="center", fontsize=9)
ax2.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(SAVE_PATH / "bootstrap_forest_plot.png", dpi=150, bbox_inches="tight")
print(f"Saved → {SAVE_PATH / 'bootstrap_forest_plot.png'}")

# ── Model comparison bar chart ──
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
for j, lbl in enumerate(LABEL_COLS):
    ax = axes[j]
    models = ["XGBoost\n(source)", "XGBoost\n(adapted)", "Two-Stream\nRun A", "Two-Stream\nRun B"]
    aurocs = [
        xgb_results["test_post"][lbl]["auroc"],
        xgb_adapted_results["test_post"][lbl]["auroc"],
        ts_a_post[lbl]["auroc"],
        ts_b_post[lbl]["auroc"],
    ]
    auprcs = [
        xgb_results["test_post"][lbl]["auprc"],
        xgb_adapted_results["test_post"][lbl]["auprc"],
        ts_a_post[lbl]["auprc"],
        ts_b_post[lbl]["auprc"],
    ]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, aurocs, w, label="AUROC", color="#2563eb", alpha=0.8)
    ax.bar(x + w/2, auprcs, w, label="AUPRC", color="#16a34a", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(lbl.replace("label_", "").replace("_", " ").title(), fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

plt.suptitle("Model Comparison on Post-Drift Test Set (2020-2022)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(SAVE_PATH / "model_comparison_bar.png", dpi=150, bbox_inches="tight")
print(f"Saved → {SAVE_PATH / 'model_comparison_bar.png'}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL SUMMARY — ALL ADDITIONS")
print("="*70)

print("\n1. BOOTSTRAP CONFIDENCE INTERVALS (Post-Drift)")
for lbl in LABEL_COLS:
    r = auroc_boot[lbl]
    sig = "✅ significant" if r["p_value"] < 0.05 else "⚠ not significant"
    print(f"   {lbl}: ΔAUROC={r['delta']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] p={r['p_value']:.4f} → {sig}")

print(f"\n2. XGBOOST BASELINE (Post-Drift)")
print(f"   {'Model':<25} {'mAUROC':>8}")
xgb_mean = np.mean([xgb_results['test_post'][l]['auroc'] for l in LABEL_COLS])
xgba_mean = np.mean([xgb_adapted_results['test_post'][l]['auroc'] for l in LABEL_COLS])
tsa_mean = np.mean([ts_a_post[l]['auroc'] for l in LABEL_COLS])
tsb_mean = np.mean([ts_b_post[l]['auroc'] for l in LABEL_COLS])
print(f"   {'XGBoost (source)':<25} {xgb_mean:.4f}")
print(f"   {'XGBoost (adapted)':<25} {xgba_mean:.4f}")
print(f"   {'Two-Stream Run A':<25} {tsa_mean:.4f}")
print(f"   {'Two-Stream Run B':<25} {tsb_mean:.4f}")

print(f"\n3. EXPANDED RAG (corpus: {len(EXPANDED_CORPUS)} = {n_relevant_exp}+{n_noise_exp})")
if exp_post_overall:
    print(f"   Post-drift: Hit@1={exp_post_overall['hit@1']:.3f} Hit@3={exp_post_overall['hit@3']:.3f} MRR={exp_post_overall['reciprocal_rank']:.3f}")
if exp_pre_overall:
    print(f"   Pre-drift:  Hit@1={exp_pre_overall['hit@1']:.3f} Hit@3={exp_pre_overall['hit@3']:.3f} MRR={exp_pre_overall['reciprocal_rank']:.3f}")

print("\n✅ All additions complete. Output files:")
print(f"   {SAVE_PATH / 'bootstrap_ci_results.json'}")
print(f"   {SAVE_PATH / 'full_comparison_results.json'}")
print(f"   {SAVE_PATH / 'expanded_rag_metrics.json'}")
print(f"   {SAVE_PATH / 'bootstrap_forest_plot.png'}")
print(f"   {SAVE_PATH / 'model_comparison_bar.png'}")