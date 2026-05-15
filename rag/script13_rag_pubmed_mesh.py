"""
script5b_pubmed_rag_hard.py
═══════════════════════════
Real-world PubMed RAG — HARD evaluation version.

Fixes from script5:
  1. Signal:noise ratio → ~1:5 (was 1:1). Reduces relevant query volume,
     increases noise volume. Real PubMed would be even sparser.
  2. Hard negatives: adds "confusable" domains that share vocabulary with
     ICU sepsis/ventilation but aren't directly applicable:
       - Cardiogenic shock (shares "shock", "vasopressor", "MAP")
       - Pediatric sepsis (shares "sepsis", "antibiotics" but different management)
       - Non-ICU respiratory (COPD exacerbation, asthma, outpatient pneumonia)
       - Surgical/trauma bleeding (shares "hypotension", "fluid resuscitation")
       - Palliative care / end-of-life ICU (shares "ventilation", "ICU")
       - Anesthesia / perioperative (shares "intubation", "sedation", "propofol")
  3. Domain-relevance evaluation uses 3-tier labeling:
       - "relevant": directly applicable ICU sepsis/ventilation evidence
       - "hard_negative": related but not directly applicable
       - "noise": completely unrelated

Run AFTER script2 (needs two_stream_models.pt + parquet files).
Dependencies: pip install sentence-transformers biopython
"""

import json, time, warnings, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize, SingleStreamDataset, ICUDataset
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from models.architectures import PhysiologyStream, TreatmentStream, FusionHead, SingleStreamModel, TwoStreamModel, SEED, SEQ_LEN, HIDDEN_DIM, TREAT_DIM, BATCH_SIZE, LSTM_LAYERS, DROPOUT, LR_ADAPT, ADAPT_EPOCHS, ADAPT_PATIENCE

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── CONFIG ─────────────────────────────────────────────────────────────────────
SEED = 42
SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH  = Path("/kaggle/working")
TOP_K_FEATURES = 8
TOP_K_DOCS = 5
N_CASES_PER_CONDITION = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

torch.manual_seed(SEED); np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# PUBMED FETCHER (same as script5)
# ══════════════════════════════════════════════════════════════════════════════
from Bio import Entrez
from xml.etree import ElementTree as ET

NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY
    print("✅ NCBI API Key found: Running at 10 requests/sec")
else:
    print("⚠️ No API Key found in environment: Limited to 3 requests/sec")


def fetch_pubmed_abstracts(query, max_results=50, min_date="2015/01/01", max_date="2024/12/31"):
    """Search PubMed using MeSH-enabled queries and return articles with abstracts."""
    try:
        handle = Entrez.esearch(
            db="pubmed", term=query, retmax=max_results,
            mindate=min_date, maxdate=max_date,
            sort="relevance", retmode="json"
        )
        search_results = json.loads(handle.read())
        handle.close()
        pmids = search_results.get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []
        time.sleep(DELAY)

        articles = []
        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="xml", retmode="xml")
            xml_data = handle.read()
            handle.close()
            root = ET.fromstring(xml_data)
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    medline = article_elem.find(".//MedlineCitation")
                    
                    # 1. MeSH Extraction
                    mesh_terms = []
                    mesh_list = medline.find(".//MeshHeadingList")
                    if mesh_list is not None:
                        for mesh in mesh_list.findall(".//MeshHeading/DescriptorName"):
                            mesh_terms.append(mesh.text)

                    article = medline.find(".//Article")
                    title_elem = article.find(".//ArticleTitle")
                    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
                    
                    abstract_elem = article.find(".//Abstract")
                    if abstract_elem is None: continue
                    
                    abstract_texts = ["".join(at.itertext()).strip() for at in abstract_elem.findall(".//AbstractText")]
                    abstract = " ".join(abstract_texts).strip()
                    if len(abstract) < 100: continue
                    
                    pmid_elem = medline.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""
                    
                    year_elem = article.find(".//Journal/JournalIssue/PubDate/Year")
                    year = year_elem.text if year_elem is not None else "2020"

                    # --- FIX: Re-adding Author Extraction ---
                    author_list = article.find(".//AuthorList")
                    first_author = "Unknown"
                    if author_list is not None:
                        first = author_list.find(".//Author")
                        if first is not None:
                            ln = first.find("LastName")
                            ini = first.find("Initials")
                            if ln is not None:
                                first_author = ln.text + (f" {ini.text}" if ini is not None else "")
                    # ----------------------------------------

                    articles.append({
                        "pmid": pmid, 
                        "title": title, 
                        "abstract": abstract,
                        "year": year, 
                        "mesh_terms": mesh_terms,
                        "first_author": first_author, # Key is back!
                        "journal": (article.find(".//Journal/Title").text if article.find(".//Journal/Title") is not None else "Unknown Journal")
                    })
                except: continue
            time.sleep(DELAY)
        return articles
    except Exception as e:
        print(f"  Error: {e}")
        return []

# ══════════════════════════════════════════════════════════════════════════════
# QUERY DEFINITIONS — balanced for hard evaluation
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("Fetching PubMed abstracts (hard evaluation corpus)")
print("="*70)

# RELEVANT: ~155 target-relevant abstracts
RELEVANT_QUERIES = [
    ('("Shock, Septic"[MeSH] OR "Sepsis"[MeSH]) AND "Vasoconstrictor Agents"[MeSH]', 40, "relevant"),
    ('("Respiration, Artificial"[MeSH] OR "Positive-Pressure Respiration"[MeSH]) AND "Respiratory Distress Syndrome"[MeSH]', 40, "relevant"),
    ('"Shock, Septic"[MeSH] AND "Fluid Therapy"[MeSH]', 40, "relevant"),
    ('("Sepsis"[MeSH] OR "Shock, Septic"[MeSH]) AND "Anti-Bacterial Agents"[MeSH]', 35, "relevant")
]

# HARD NEGATIVES: ~275 semantically proximate hard negatives
HARD_NEGATIVE_QUERIES = [
    ('"Shock, Cardiogenic"[MeSH]', 55, "hard_negative"),
    ('"Sepsis"[MeSH] AND "Pediatrics"[MeSH]', 55, "hard_negative"),
    ('"Hemorrhage"[MeSH] AND "Fluid Therapy"[MeSH] AND "Wounds and Injuries"[MeSH]', 55, "hard_negative"),
    ('"Pulmonary Disease, Chronic Obstructive"[MeSH] AND "Noninvasive Ventilation"[MeSH]', 55, "hard_negative"),
    ('"Shock"[MeSH] AND "Pulmonary Embolism"[MeSH]', 55, "hard_negative")
]

# EASY NOISE: ~491 general noise abstracts
NOISE_QUERIES = [
    ('"Diabetes Mellitus, Type 2"[MeSH] AND "Hypoglycemic Agents"[MeSH]', 82, "noise"),
    ('"Neoplasms"[MeSH] AND "Immunotherapy"[MeSH]', 82, "noise"),
    ('"Osteoarthritis, Knee"[MeSH] AND "Arthroplasty, Replacement, Knee"[MeSH]', 82, "noise"),
    ('"Depressive Disorder, Major"[MeSH] AND "Serotonin Uptake Inhibitors"[MeSH]', 82, "noise"),
    ('"Stroke"[MeSH] AND "Thrombolytic Therapy"[MeSH]', 82, "noise"),
    ('"Atrial Fibrillation"[MeSH] AND "Anticoagulants"[MeSH]', 81, "noise")
]

# ── Fetch all ──────────────────────────────────────────────────────────────────
all_queries = (
    [(q, n, "relevant", t) for q, n, t in RELEVANT_QUERIES] +
    [(q, n, "hard_negative", t) for q, n, t in HARD_NEGATIVE_QUERIES] +
    [(q, n, "noise", t) for q, n, t in NOISE_QUERIES]
)

corpus_raw = []
for query, max_n, domain, _ in all_queries:
    articles = fetch_pubmed_abstracts(query, max_results=max_n)
    for a in articles:
        a["domain"] = domain
    corpus_raw.extend(articles)
    domain_icon = {"relevant": "🟢", "hard_negative": "🟡", "noise": "⚪"}.get(domain, "?")
    print(f"  {domain_icon} [{domain:14s}] '{query[:55]}...' → {len(articles)}")

# Deduplicate
seen = set()
deduped = []
for a in corpus_raw:
    if a["pmid"] not in seen:
        seen.add(a["pmid"])
        deduped.append(a)

counts = defaultdict(int)
for a in deduped:
    counts[a["domain"]] += 1

print(f"\n✅ Corpus: {len(deduped)} unique abstracts")
print(f"  Relevant (ICU sepsis/vent): {counts['relevant']}")
print(f"  Hard negatives (confusable): {counts['hard_negative']}")
print(f"  Easy noise (unrelated): {counts['noise']}")
print(f"  Signal:noise ratio = 1:{(counts['hard_negative']+counts['noise'])/max(counts['relevant'],1):.1f}")

with open(SAVE_PATH / "pubmed_corpus_hard.json", "w") as f:
    json.dump(deduped, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# BUILD INDEX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Building semantic index")
print("="*70)

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))

corpus_entries = []
for a in deduped:
    text = f"{a['title']}. {a['abstract']}"
    if len(text) > 2000:
        text = text[:2000]
    year = int(a["year"]) if a["year"].isdigit() else 2020

    # Era classification
    if year >= 2020:
        era = "post_covid"
    elif year < 2020:
        era = "pre_covid"
    else:
        era = "all"

    corpus_entries.append({
        "id": f"PMID_{a['pmid']}",
        "pmid": a["pmid"],
        "title": a["title"],
        "text": a["abstract"],
        "source": f"{a['first_author']} et al. {a['journal']} {a['year']}",
        "domain": a["domain"],
        "era": era,
        "year": year,
        "text_for_embedding": text,
    })

print(f"Embedding {len(corpus_entries)} passages...")
corpus_embeddings = embedder.encode(
    [e["text_for_embedding"] for e in corpus_entries],
    convert_to_numpy=True, show_progress_bar=True,
    batch_size=32, normalize_embeddings=True
)
print(f"Shape: {corpus_embeddings.shape}")

era_dist = defaultdict(int)
for e in corpus_entries:
    era_dist[e["era"]] += 1
print(f"Era distribution: {dict(era_dist)}")


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL + METRICS
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_pubmed(query, era_filter=None, k=TOP_K_DOCS):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).ravel()
    candidates = sorted(zip(range(len(corpus_entries)), sims), key=lambda x: x[1], reverse=True)
    results = []
    for idx, sim in candidates:
        e = corpus_entries[idx]
        if era_filter and e["era"] not in (era_filter, "all"):
            continue
        results.append({
            "id": e["id"], "pmid": e["pmid"], "title": e["title"],
            "text": e["text"][:500], "source": e["source"],
            "domain": e["domain"], "era": e["era"],
            "relevance_score": float(sim),
        })
        if len(results) >= k:
            break
    return results

def compute_metrics(retrieved, k_values=(1, 3, 5)):
    """
    Three-tier evaluation:
      - hit@k: is there at least one relevant article?
      - precision@k: fraction of relevant in top-k
      - hard_negative_rate@k: fraction that are hard negatives (not useful but not random)
      - noise_intrusion@k: fraction that are completely unrelated noise
      - MRR: reciprocal rank of first relevant
    """
    m = {}
    for k in k_values:
        top = retrieved[:k]
        n_rel = sum(1 for r in top if r["domain"] == "relevant")
        n_hard = sum(1 for r in top if r["domain"] == "hard_negative")
        n_noise = sum(1 for r in top if r["domain"] == "noise")
        m[f"hit@{k}"] = 1.0 if n_rel > 0 else 0.0
        m[f"precision@{k}"] = n_rel / k
        m[f"hard_neg_rate@{k}"] = n_hard / k
        m[f"noise_intrusion@{k}"] = n_noise / k
    rr = 0.0
    for i, r in enumerate(retrieved):
        if r["domain"] == "relevant":
            rr = 1.0 / (i + 1)
            break
    m["reciprocal_rank"] = rr
    return m


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS + RUN EVALUATION (same architecture as scripts 2-4)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Loading models and running patient-level evaluation")
print("="*70)

ckpt = torch.load(SAVE_PATH / "two_stream_models.pt", map_location=device, weights_only=False)
seq_dim, treat_dim, n_targets = ckpt["seq_dim"], ckpt["treat_dim"], ckpt["n_targets"]
train_stats = ckpt["train_stats"]

model_A = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_A.load_state_dict(ckpt["run_a"]); model_A.eval()
model_B = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_B.load_state_dict(ckpt["run_b"]); model_B.eval()

test_df = normalize(load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES),train_stats)
test_pre  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")

post_stays = test_post.filter(pl.col("hrs_from_admit")==0).sort("intime")["stay_id"].to_list()
n_at = int(len(post_stays)*0.30); n_av = int(len(post_stays)*0.10)
eval_post_df = test_post.filter(pl.col("stay_id").is_in(post_stays[n_at+n_av:]))

ds_pre = ICUDataset(test_pre, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
ds_post = ICUDataset(eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
print(f"Pre: {len(ds_pre)} | Post: {len(ds_post)} patients")

def integrated_gradients(model, xs, xt, ti, steps=20):
    model.eval(); xs,xt = xs.to(device),xt.to(device)
    bs,bt = torch.zeros_like(xs),torch.zeros_like(xt)
    sg,tg = torch.zeros_like(xs),torch.zeros_like(xt)
    for a in np.linspace(0,1,steps):
        is_ = (bs+a*(xs-bs)).requires_grad_(True)
        it_ = (bt+a*(xt-bt)).requires_grad_(True)
        out = model(is_,it_)[:,ti].sum()
        g1,g2 = torch.autograd.grad(out,[is_,it_])
        sg+=g1; tg+=g2
    return ((xs-bs)*sg/steps).cpu().numpy(), ((xt-bt)*tg/steps).cpu().numpy()

def explain(model, xs, xt, li, sc, tc):
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(xs,xt))[0,li].item()
    sa,ta = integrated_gradients(model,xs,xt,li)
    saf = sa[0].sum(axis=0); taf = ta[0]
    imps = ([("physio:"+n,v) for n,v in zip(sc,saf)]+[("treat:"+n,v) for n,v in zip(tc,taf)])
    imps.sort(key=lambda x:abs(x[1]),reverse=True)
    return {"predicted_probability":prob,
            "top_features":[{"feature":n,"contribution":float(v),
                             "direction":"increases" if v>0 else "decreases"} for n,v in imps[:TOP_K_FEATURES]]}

def features_to_query(feats, label):
    lt = {"label_vasopressor":"vasopressor septic shock hemodynamic norepinephrine MAP",
          "label_intubation":"mechanical ventilation intubation respiratory failure ARDS",
          "label_septic_shock":"septic shock sepsis organ dysfunction lactate"}
    ft = {"lactate":"elevated lactate perfusion","map_invasive":"hypotension mean arterial pressure",
          "spo2":"hypoxemia oxygen saturation","resp_rate":"tachypnea respiratory rate",
          "creatinine":"acute kidney injury creatinine","wbc":"leukocytosis infection",
          "has_norepinephrine_obs":"norepinephrine infusion","has_vasopressin_obs":"vasopressin shock",
          "total_crystalloid_ml":"fluid resuscitation crystalloid","max_fio2_obs":"high FiO2 oxygen",
          "high_fio2_flag":"severe hypoxemia","max_peep_obs":"PEEP ventilator",
          "early_steroid":"corticosteroid dexamethasone","early_antibiotic":"empiric antibiotic",
          "has_propofol_midaz_obs":"sedation intubation","time_to_first_vaso_hrs":"early vasopressor timing"}
    terms = [lt.get(label, label.replace("label_",""))]
    for f in feats:
        c = f["feature"].replace("physio:","").replace("treat:","").replace("_mask","")
        if c in ft: terms.append(ft[c])
    return ". ".join(terms)

def select_cases(ds, li, n=N_CASES_PER_CONDITION):
    pos = [i for i in range(len(ds)) if ds.labels[i,li]==1]
    if len(pos)>n:
        pos = np.random.RandomState(SEED).choice(pos,n,replace=False).tolist()
    return pos


# ══════════════════════════════════════════════════════════════════════════════
# RUN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Running PubMed RAG evaluation (hard corpus)")
print("="*70)

all_explanations = {"post_drift": [], "pre_drift": []}

for era_label, ds, model, era_filter in [
    ("post_drift", ds_post, model_B, "post_covid"),
    ("pre_drift",  ds_pre,  model_A, "pre_covid"),
]:
    metrics_by_label = {lbl: [] for lbl in LABEL_COLS}
    print(f"\n{era_label} ({len(ds)} patients):")
    for li, ln in enumerate(LABEL_COLS):
        cases = select_cases(ds, li)
        print(f"  {ln}: {len(cases)} cases")
        for ci in cases:
            seq, treat, lbl = ds[ci]
            exp = explain(model, seq.unsqueeze(0).to(device), treat.unsqueeze(0).to(device),
                          li, ds.seq_cols, ds.treat_cols)
            query = features_to_query(exp["top_features"], ln)
            retrieved = retrieve_pubmed(query, era_filter=era_filter, k=TOP_K_DOCS)
            m = compute_metrics(retrieved)
            metrics_by_label[ln].append(m)
            all_explanations[era_label].append({
                "stay_id": int(ds.stay_ids[ci]), "label": ln,
                "true_label": int(lbl[li].item()),
                "probability": exp["predicted_probability"],
                "query": query, "top_features": exp["top_features"],
                "retrieved": retrieved, "metrics": m,
            })

    # Aggregate
    print(f"\n  {'Label':<22} {'Hit@1':>7} {'Hit@3':>7} {'P@3':>7} {'P@5':>7} {'MRR':>7} {'HN@5':>7} {'NI@5':>7}")
    print("  " + "-"*60)
    agg = {}
    for lbl, cases in metrics_by_label.items():
        if not cases: continue
        agg[lbl] = {k: float(np.mean([c[k] for c in cases]))
                    for k in ["hit@1","hit@3","hit@5","precision@3","precision@5",
                              "reciprocal_rank","hard_neg_rate@5","noise_intrusion@5"]}
        a = agg[lbl]
        print(f"  {lbl:<22} {a['hit@1']:>7.3f} {a['hit@3']:>7.3f} {a['precision@3']:>7.3f} "
              f"{a['precision@5']:>7.3f} {a['reciprocal_rank']:>7.3f} "
              f"{a['hard_neg_rate@5']:>7.3f} {a['noise_intrusion@5']:>7.3f}")

    all_cases = [c for cases in metrics_by_label.values() for c in cases]
    if all_cases:
        overall = {k: float(np.mean([c[k] for c in all_cases]))
                   for k in ["hit@1","hit@3","hit@5","precision@3","precision@5",
                             "reciprocal_rank","hard_neg_rate@5","noise_intrusion@5"]}
        print(f"  {'OVERALL':<22} {overall['hit@1']:>7.3f} {overall['hit@3']:>7.3f} "
              f"{overall['precision@3']:>7.3f} {overall['precision@5']:>7.3f} "
              f"{overall['reciprocal_rank']:>7.3f} {overall['hard_neg_rate@5']:>7.3f} "
              f"{overall['noise_intrusion@5']:>7.3f}")

        if era_label == "post_drift":
            post_agg, post_overall = agg, overall
        else:
            pre_agg, pre_overall = agg, overall

# ── Save everything ────────────────────────────────────────────────────────────
pubmed_metrics = {
    "corpus_source": "PubMed via NCBI E-utilities (real abstracts)",
    "corpus_size": len(corpus_entries),
    "n_relevant": counts["relevant"],
    "n_hard_negative": counts["hard_negative"],
    "n_noise": counts["noise"],
    "signal_to_noise_ratio": f"1:{(counts['hard_negative']+counts['noise'])/max(counts['relevant'],1):.1f}",
    "embedding_model": EMBEDDING_MODEL,
    "evaluation": "3-tier domain relevance (relevant / hard_negative / noise)",
    "post_drift_per_label": post_agg,
    "post_drift_overall": post_overall,
    "pre_drift_per_label": pre_agg,
    "pre_drift_overall": pre_overall,
}
with open(SAVE_PATH / "pubmed_rag_hard_metrics.json", "w") as f:
    json.dump(pubmed_metrics, f, indent=2)

with open(SAVE_PATH / "pubmed_rag_hard_explanations.json", "w") as f:
    json.dump(all_explanations, f, indent=2)

# ── Sample retrievals ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SAMPLE RETRIEVALS")
print("="*70)
for exp in all_explanations["post_drift"][:3]:
    print(f"\n{exp['label']} (stay {exp['stay_id']}, prob={exp['probability']:.3f}):")
    print(f"  Query: {exp['query'][:100]}...")
    for i, r in enumerate(exp["retrieved"][:5]):
        icon = {"relevant":"✅","hard_negative":"🟡","noise":"❌"}[r["domain"]]
        print(f"  [{i+1}] {icon} {r['domain']:14s} sim={r['relevance_score']:.3f} | {r['title'][:75]}...")
        print(f"       {r['source']}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Corpus: {len(corpus_entries)} real PubMed abstracts")
print(f"  Relevant: {counts['relevant']} | Hard negatives: {counts['hard_negative']} | Noise: {counts['noise']}")
print(f"  Ratio: {pubmed_metrics['signal_to_noise_ratio']}")
print(f"Embedding: {EMBEDDING_MODEL}")
print(f"\nPost-drift: Hit@1={post_overall['hit@1']:.3f} P@3={post_overall['precision@3']:.3f} "
      f"MRR={post_overall['reciprocal_rank']:.3f} HardNeg@5={post_overall['hard_neg_rate@5']:.3f}")
print(f"Pre-drift:  Hit@1={pre_overall['hit@1']:.3f} P@3={pre_overall['precision@3']:.3f} "
      f"MRR={pre_overall['reciprocal_rank']:.3f} HardNeg@5={pre_overall['hard_neg_rate@5']:.3f}")

print(f"\n✅ Outputs:")
print(f"  {SAVE_PATH / 'pubmed_corpus_hard.json'}")
print(f"  {SAVE_PATH / 'pubmed_rag_hard_metrics.json'}")
print(f"  {SAVE_PATH / 'pubmed_rag_hard_explanations.json'}")