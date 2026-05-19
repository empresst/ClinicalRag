#%%writefile rag/script11_rag_pubmed_easy.py
"""
script11_pubmed_rag.py
═════════════════════
Real-world RAG with PubMed abstracts — replaces the synthetic corpus.

What this does:
  1. Queries PubMed via NCBI E-utilities for real abstracts across:
     - ICU/sepsis/vasopressor/ventilation topics (relevant)
     - 12 unrelated medical domains (realistic noise)
  2. Builds a ~800-1000 passage corpus from actual published literature
  3. Embeds everything with sentence-transformers
  4. Runs retrieval evaluation on the same patient cases from script2/3
  5. Compares real-world RAG metrics against the synthetic corpus baseline

Requirements:
  pip install sentence-transformers biopython
  Set your NCBI API key (get free at https://www.ncbi.nlm.nih.gov/account/settings/)

Run AFTER script2 (needs two_stream_models.pt + parquet files).
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
SEED = 42
SEQ_LEN, HIDDEN_DIM, TREAT_DIM = 6, 64, 32
LSTM_LAYERS, BATCH_SIZE, DROPOUT = 2, 64, 0.3
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]
BASE_PATH  = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH  = Path("/kaggle/working")
TOP_K_FEATURES = 8
TOP_K_DOCS = 5
N_CASES_PER_CONDITION = 5
REQUESTS_PER_SEC = 9
DELAY = 1.0 / REQUESTS_PER_SEC
# ── NCBI API KEY ───────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

torch.manual_seed(SEED); np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: FETCH PUBMED ABSTRACTS
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("STEP 1: Fetching PubMed abstracts via NCBI E-utilities")
print("="*70)

import os
from Bio import Entrez

Entrez.email = os.environ.get("NCBI_EMAIL") 

api_key = os.environ.get("NCBI_API_KEY")

if api_key:
    Entrez.api_key = api_key
    print("✅ NCBI API Key found: Running at 10 requests/sec")
else:
    print("⚠️ No API Key found in environment: Limited to 3 requests/sec")
    
def fetch_pubmed_abstracts(query, max_results=50):
    """
    Search PubMed and return list of {pmid, title, abstract, journal, year, first_author}.
    Uses Biopython's native dictionary parser for stability.
    """
    try:
        # 1. Search for the PMIDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        search_results = Entrez.read(handle)
        pmids = search_results.get("IdList", [])
        
        if not pmids:
            print(f"    No results for: {query[:60]}...")
            return []
            
        time.sleep(DELAY) # Respect NCBI rate limits
        
        # 2. Fetch the actual article data
        articles = []
        # Batch requests in chunks of 100 just like your original code
        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]
            handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="medline", retmode="xml")
            records = Entrez.read(handle)
            
            # 3. Extract the data using dictionary keys
            for article in records.get('PubmedArticle', []):
                try:
                    medline = article['MedlineCitation']
                    article_data = medline['Article']
                    
                    # Safely grab the abstract (PubMed often splits abstracts into lists like Background, Methods)
                    abstract_list = article_data.get('Abstract', {}).get('AbstractText', [])
                    abstract = " ".join([str(text) for text in abstract_list]).strip()
                    
                    # Skip if no abstract or too short (matches your original logic)
                    if len(abstract) < 100:
                        continue
                        
                    # Extract remaining fields safely
                    pmid = str(medline.get('PMID', ''))
                    title = str(article_data.get('ArticleTitle', ''))
                    journal = str(article_data.get('Journal', {}).get('Title', ''))
                    year = str(article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {}).get('Year', 'Unknown'))
                    
                    # Get First Author
                    authors = article_data.get('AuthorList', [])
                    first_author = ""
                    if authors and 'LastName' in authors[0]:
                        first_author = authors[0]['LastName']
                        if 'Initials' in authors[0]:
                            first_author += f" {authors[0]['Initials']}"

                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "year": year,
                        "first_author": first_author,
                    })
                except KeyError:
                    # If an article is missing a crucial formatting key, just skip it and move on
                    continue
                    
            time.sleep(DELAY)
            
        return articles
        
    except Exception as e:
        print(f"    Error fetching '{query[:50]}...': {e}")
        return [] 


# ── Define search queries ─────────────────────────────────────────────────────
# RELEVANT queries: ICU, sepsis, vasopressors, ventilation, ARDS
RELEVANT_QUERIES = [
    # Sepsis / Vasopressor
    ("sepsis vasopressor norepinephrine ICU management", 40, "ICU_sepsis_vaso"),
    ("septic shock fluid resuscitation crystalloid", 30, "ICU_sepsis_fluid"),
    ("surviving sepsis campaign guidelines", 25, "ICU_ssc_guidelines"),
    ("vasopressin septic shock adjunct", 20, "ICU_vasopressin"),
    ("sepsis lactate clearance targeted resuscitation", 20, "ICU_lactate"),
    ("sepsis early antibiotic timing mortality", 20, "ICU_abx_timing"),
    ("sepsis qSOFA SOFA organ dysfunction", 20, "ICU_sepsis_scoring"),
    ("mean arterial pressure target septic shock", 15, "ICU_map_target"),
    ("corticosteroids septic shock hydrocortisone", 15, "ICU_steroids_sepsis"),
    ("sepsis bundle hour-1 compliance outcomes", 15, "ICU_sepsis_bundle"),
    
    # Respiratory / Ventilation / ARDS
    ("mechanical ventilation ARDS lung protective", 30, "ICU_ards_vent"),
    ("high flow nasal cannula respiratory failure ICU", 25, "ICU_hfnc"),
    ("prone positioning ARDS mortality", 20, "ICU_prone"),
    ("COVID-19 intubation delayed early ventilation", 25, "ICU_covid_intubation"),
    ("COVID-19 dexamethasone respiratory support", 20, "ICU_covid_dexa"),
    ("PEEP titration ARDS recruitment", 15, "ICU_peep"),
    ("driving pressure ARDS ventilator settings", 15, "ICU_driving_pressure"),
    ("spontaneous breathing trial extubation", 15, "ICU_sbt"),
    ("awake prone positioning COVID hypoxemia", 15, "ICU_awake_prone"),
    ("rapid sequence intubation ICU emergency", 15, "ICU_rsi"),
    ("neuromuscular blockade ARDS cisatracurium", 10, "ICU_nmba"),
    ("ventilator induced lung injury prevention", 10, "ICU_vili"),
    
    # Drift / Practice change
    ("COVID-19 treatment practice change ICU", 20, "ICU_covid_practice"),
    ("antibiotic stewardship COVID-19 ICU", 15, "ICU_abx_stewardship"),
]

# NOISE queries: unrelated medical domains
NOISE_QUERIES = [
    # Diabetes
    ("type 2 diabetes metformin first-line treatment", 15, "NOISE_diabetes_tx"),
    ("GLP-1 receptor agonist cardiovascular outcomes", 15, "NOISE_diabetes_glp1"),
    ("diabetic ketoacidosis management insulin", 10, "NOISE_diabetes_dka"),
    ("continuous glucose monitoring type 1 diabetes", 10, "NOISE_diabetes_cgm"),
    ("gestational diabetes screening treatment", 10, "NOISE_diabetes_gdm"),
    
    # Orthopedics
    ("hip fracture elderly surgical management", 15, "NOISE_ortho_hip"),
    ("ACL reconstruction rehabilitation outcomes", 10, "NOISE_ortho_acl"),
    ("osteoporosis bisphosphonate treatment", 10, "NOISE_ortho_osteo"),
    ("knee osteoarthritis arthroplasty", 10, "NOISE_ortho_knee"),
    
    # Oncology
    ("breast cancer screening mammography", 12, "NOISE_onc_breast"),
    ("lung cancer screening low-dose CT", 10, "NOISE_onc_lung"),
    ("immunotherapy checkpoint inhibitor toxicity", 12, "NOISE_onc_immuno"),
    ("chemotherapy induced nausea antiemetic", 10, "NOISE_onc_nausea"),
    ("CAR-T cell therapy cytokine release syndrome", 10, "NOISE_onc_cart"),
    
    # Pediatrics
    ("pediatric asthma exacerbation management", 10, "NOISE_peds_asthma"),
    ("bronchiolitis infant RSV management", 10, "NOISE_peds_bronch"),
    ("febrile infant evaluation sepsis workup", 10, "NOISE_peds_febrile"),
    ("pediatric obesity intervention", 8, "NOISE_peds_obesity"),
    
    # Neurology
    ("acute ischemic stroke thrombolysis thrombectomy", 12, "NOISE_neuro_stroke"),
    ("status epilepticus treatment algorithm", 10, "NOISE_neuro_epilepsy"),
    ("migraine CGRP prophylaxis", 8, "NOISE_neuro_migraine"),
    ("Parkinson disease levodopa treatment", 8, "NOISE_neuro_parkinson"),
    
    # Cardiology (non-ICU)
    ("atrial fibrillation anticoagulation DOAC", 12, "NOISE_cardio_afib"),
    ("heart failure reduced ejection fraction SGLT2", 12, "NOISE_cardio_hf"),
    ("hypertension treatment guidelines blood pressure", 10, "NOISE_cardio_htn"),
    ("statin therapy cardiovascular prevention", 8, "NOISE_cardio_lipid"),
    
    # GI
    ("inflammatory bowel disease biologic therapy", 10, "NOISE_gi_ibd"),
    ("GERD proton pump inhibitor management", 8, "NOISE_gi_gerd"),
    ("acute pancreatitis fluid resuscitation nutrition", 10, "NOISE_gi_panc"),
    ("cirrhosis variceal bleeding prophylaxis", 8, "NOISE_gi_cirrhosis"),
    
    # Dermatology
    ("melanoma surgical excision sentinel node", 8, "NOISE_derm_melanoma"),
    ("psoriasis biologic IL-17 IL-23 treatment", 8, "NOISE_derm_psoriasis"),
    ("atopic dermatitis dupilumab treatment", 8, "NOISE_derm_atopic"),
    
    # Psychiatry
    ("major depression SSRI treatment guidelines", 8, "NOISE_psych_depression"),
    ("schizophrenia antipsychotic treatment", 8, "NOISE_psych_schizo"),
    ("PTSD trauma-focused CBT treatment", 8, "NOISE_psych_ptsd"),
    
    # Rheumatology
    ("rheumatoid arthritis methotrexate biologic", 8, "NOISE_rheum_ra"),
    ("gout urate-lowering therapy allopurinol", 8, "NOISE_rheum_gout"),
    
    # Infectious disease (non-ICU)
    ("community acquired pneumonia antibiotic", 10, "NOISE_id_cap"),
    ("urinary tract infection nitrofurantoin", 8, "NOISE_id_uti"),
    ("HIV antiretroviral integrase inhibitor", 8, "NOISE_id_hiv"),
]

# ── Fetch all abstracts ───────────────────────────────────────────────────────
print(f"\nFetching relevant ICU literature ({len(RELEVANT_QUERIES)} queries)...")
relevant_corpus = []
for query, max_n, tag in RELEVANT_QUERIES:
    articles = fetch_pubmed_abstracts(query, max_results=max_n)
    for a in articles:
        a["tag"] = tag
        a["domain"] = "relevant"
    relevant_corpus.extend(articles)
    print(f"  [{tag}] '{query[:50]}...' → {len(articles)} articles")

print(f"\nFetching noise literature ({len(NOISE_QUERIES)} queries)...")
noise_corpus = []
for query, max_n, tag in NOISE_QUERIES:
    articles = fetch_pubmed_abstracts(query, max_results=max_n)
    for a in articles:
        a["tag"] = tag
        a["domain"] = "noise"
    noise_corpus.extend(articles)
    print(f"  [{tag}] '{query[:50]}...' → {len(articles)} articles")

# Deduplicate by PMID
all_articles = relevant_corpus + noise_corpus
seen_pmids = set()
deduped = []
for a in all_articles:
    if a["pmid"] not in seen_pmids:
        seen_pmids.add(a["pmid"])
        deduped.append(a)

n_relevant = sum(1 for a in deduped if a["domain"] == "relevant")
n_noise = sum(1 for a in deduped if a["domain"] == "noise")
print(f"\n✅ Total corpus: {len(deduped)} unique abstracts")
print(f"  Relevant (ICU/sepsis/ventilation): {n_relevant}")
print(f"  Noise (other domains): {n_noise}")

# Save raw corpus
with open(SAVE_PATH / "pubmed_corpus_raw.json", "w") as f:
    json.dump(deduped, f, indent=2)
print(f"  Saved → {SAVE_PATH / 'pubmed_corpus_raw.json'}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD RETRIEVAL INDEX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 2: Building semantic retrieval index")
print("="*70)

from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))
print(f"  Embedding model: {EMBEDDING_MODEL}")

# Build corpus entries with structured text for embedding
corpus_entries = []
for a in deduped:
    # Combine title + abstract for embedding (title provides topic signal)
    text_for_embedding = f"{a['title']}. {a['abstract']}"
    # Truncate very long abstracts (model has 256 token limit for MiniLM)
    if len(text_for_embedding) > 2000:
        text_for_embedding = text_for_embedding[:2000]
    
    corpus_entries.append({
        "id": f"PMID_{a['pmid']}",
        "pmid": a["pmid"],
        "title": a["title"],
        "text": a["abstract"],
        "source": f"{a['first_author']} et al. {a['journal']} {a['year']}",
        "domain": a["domain"],
        "tag": a["tag"],
        "year": a["year"],
        "text_for_embedding": text_for_embedding,
    })

print(f"  Embedding {len(corpus_entries)} passages...")
corpus_texts = [e["text_for_embedding"] for e in corpus_entries]
corpus_embeddings = embedder.encode(
    corpus_texts, convert_to_numpy=True,
    show_progress_bar=True, batch_size=32,
    normalize_embeddings=True
)
print(f"  Embedding shape: {corpus_embeddings.shape}")

# Era classification based on year + tag
def classify_era(entry):
    """Classify article as pre_covid, post_covid, or all based on year and topic."""
    year = int(entry["year"]) if entry["year"].isdigit() else 2020
    tag = entry["tag"]
    
    # COVID-specific topics are post_covid regardless of year
    if "covid" in tag.lower():
        return "post_covid"
    # Pre-2020 articles are pre_covid
    if year < 2020:
        return "pre_covid"
    # Post-2020 articles are post_covid if ICU-related
    if year >= 2020 and entry["domain"] == "relevant":
        return "post_covid"
    # Everything else is "all"
    return "all"

for e in corpus_entries:
    e["era"] = classify_era(e)

era_counts = defaultdict(int)
for e in corpus_entries:
    era_counts[e["era"]] += 1
print(f"  Era distribution: {dict(era_counts)}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: RETRIEVAL FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_pubmed(query, era_filter=None, k=TOP_K_DOCS):
    """Retrieve top-k PubMed abstracts by semantic similarity."""
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).ravel()
    
    candidates = sorted(zip(range(len(corpus_entries)), sims),
                        key=lambda x: x[1], reverse=True)
    results = []
    for idx, sim in candidates:
        e = corpus_entries[idx]
        if era_filter and e["era"] not in (era_filter, "all"):
            continue
        results.append({
            "id": e["id"],
            "pmid": e["pmid"],
            "title": e["title"],
            "text": e["text"][:500],  # Truncate for display
            "source": e["source"],
            "domain": e["domain"],
            "tag": e["tag"],
            "era": e["era"],
            "relevance_score": float(sim),
        })
        if len(results) >= k:
            break
    return results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: GROUND TRUTH FOR REAL-WORLD EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
# For PubMed retrieval, "relevant" = any article from the ICU/sepsis/ventilation
# domain (domain=="relevant"). This is a domain-relevance evaluation:
# can the retriever find ICU-relevant literature amid noise?


def compute_retrieval_metrics_domain(retrieved, k_values=(1, 3, 5)):
    """
    Domain-relevance metrics: is the retrieved article from a relevant domain?
    More realistic than exact-ID matching for real PubMed retrieval.
    """
    metrics = {}
    for k in k_values:
        top_k = retrieved[:k]
        n_relevant = sum(1 for r in top_k if r["domain"] == "relevant")
        metrics[f"hit@{k}"] = 1.0 if n_relevant > 0 else 0.0
        metrics[f"precision@{k}"] = n_relevant / k
        # Recall: what fraction of all relevant in top-k vs total relevant in corpus
        # Not meaningful for large corpus, so we use precision-oriented metrics
    
    # MRR: rank of first relevant result
    rr = 0.0
    for i, r in enumerate(retrieved):
        if r["domain"] == "relevant":
            rr = 1.0 / (i + 1)
            break
    metrics["reciprocal_rank"] = rr
    
    # Noise intrusion: how many noise articles in top-k
    for k in k_values:
        top_k = retrieved[:k]
        n_noise = sum(1 for r in top_k if r["domain"] == "noise")
        metrics[f"noise_intrusion@{k}"] = n_noise / k
    
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: LOAD MODELS AND RUN PATIENT-LEVEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 3-5: Loading models and running evaluation")
print("="*70)



# ── Load models and data ──────────────────────────────────────────────────────
print("Loading models...")
ckpt = torch.load("/kaggle/input/datasets/fatematamanna/ptfiles/two_stream_models (3).pt", map_location=device, weights_only=False)
seq_dim, treat_dim, n_targets = ckpt["seq_dim"], ckpt["treat_dim"], ckpt["n_targets"]
train_stats = ckpt["train_stats"]

model_A = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_A.load_state_dict(ckpt["run_a"]); model_A.eval()
model_B = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_B.load_state_dict(ckpt["run_b"]); model_B.eval()

test_df = normalize(load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES),train_stats)
test_pre  = test_df.filter(pl.col("anchor_year_group") == "2017 - 2019")
test_post = test_df.filter(pl.col("anchor_year_group") == "2020 - 2022")


# Reproduce held-out split
post_stays = test_post.filter(pl.col("hrs_from_admit") == 0).sort("intime")["stay_id"].to_list()
n_adapt_train = int(len(post_stays) * 0.30)
n_adapt_val   = int(len(post_stays) * 0.10)
eval_post_stays = post_stays[n_adapt_train + n_adapt_val:]
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))



ds_pre  = ICUDataset(test_pre,     SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
ds_post = ICUDataset(eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
print(f"  Pre-drift: {len(ds_pre)} patients | Post-drift: {len(ds_post)} patients")

# ── Integrated gradients + explain ────────────────────────────────────────────
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

def features_to_query_text(top_features, label_name):
    label_text = {
        "label_vasopressor": "vasopressor need septic shock hemodynamic support norepinephrine MAP",
        "label_intubation":  "mechanical ventilation intubation respiratory failure ARDS oxygen",
        "label_septic_shock": "septic shock sepsis organ dysfunction lactate vasopressor",
    }
    feat_text = {
        "lactate": "elevated lactate tissue perfusion",
        "map_invasive": "low mean arterial pressure hypotension",
        "spo2": "hypoxemia low oxygen saturation",
        "resp_rate": "tachypnea elevated respiratory rate",
        "creatinine": "acute kidney injury elevated creatinine",
        "wbc": "leukocytosis infection white blood cells",
        "has_norepinephrine_obs": "norepinephrine vasopressor infusion",
        "has_vasopressin_obs": "vasopressin refractory shock",
        "total_crystalloid_ml": "fluid resuscitation crystalloid volume",
        "max_fio2_obs": "high FiO2 supplemental oxygen requirement",
        "high_fio2_flag": "severe hypoxemia high inspired oxygen",
        "max_peep_obs": "PEEP positive end-expiratory pressure ventilator",
        "early_steroid": "corticosteroid dexamethasone early administration",
        "early_antibiotic": "empiric antibiotic administration sepsis",
        "has_propofol_midaz_obs": "sedation propofol midazolam intubation",
        "time_to_first_vaso_hrs": "early vasopressor initiation timing",
    }
    terms = [label_text.get(label_name, label_name.replace("label_", ""))]
    for f in top_features:
        clean = f["feature"].replace("physio:", "").replace("treat:", "").replace("_mask", "")
        if clean in feat_text:
            terms.append(feat_text[clean])
    return ". ".join(terms)

def select_cases_by_label(dataset, label_idx, n_positive=N_CASES_PER_CONDITION):
    pos_indices = [i for i in range(len(dataset)) if dataset.labels[i, label_idx] == 1]
    if len(pos_indices) > n_positive:
        rng = np.random.RandomState(SEED)
        pos_indices = rng.choice(pos_indices, n_positive, replace=False).tolist()
    return pos_indices


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: RUN PUBMED RAG EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("STEP 6: PubMed RAG evaluation on patient cases")
print("="*70)

# Post-drift cases (Run B adapted)
print("\nProcessing post-drift cases...")
pubmed_post_metrics = {lbl: [] for lbl in LABEL_COLS}
pubmed_post_explanations = []

for label_idx, label_name in enumerate(LABEL_COLS):
    cases = select_cases_by_label(ds_post, label_idx, N_CASES_PER_CONDITION)
    print(f"  {label_name}: {len(cases)} cases")
    for case_idx in cases:
        seq, treat, lbl = ds_post[case_idx]
        seq_b = seq.unsqueeze(0).to(device)
        treat_b = treat.unsqueeze(0).to(device)
        
        exp = explain_prediction(model_B, seq_b, treat_b, label_idx,
                                  ds_post.seq_cols, ds_post.treat_cols)
        query = features_to_query_text(exp["top_features"], label_name)
        retrieved = retrieve_pubmed(query, era_filter="post_covid", k=TOP_K_DOCS)
        
        m = compute_retrieval_metrics_domain(retrieved)
        pubmed_post_metrics[label_name].append(m)
        
        pubmed_post_explanations.append({
            "stay_id": int(ds_post.stay_ids[case_idx]),
            "era": "post_drift",
            "label": label_name,
            "true_label": int(lbl[label_idx].item()),
            "probability": exp["predicted_probability"],
            "query": query,
            "top_features": exp["top_features"],
            "retrieved": retrieved,
            "metrics": m,
        })

# Pre-drift cases (Run A source)
print("Processing pre-drift cases...")
pubmed_pre_metrics = {lbl: [] for lbl in LABEL_COLS}
pubmed_pre_explanations = []

for label_idx, label_name in enumerate(LABEL_COLS):
    cases = select_cases_by_label(ds_pre, label_idx, N_CASES_PER_CONDITION)
    for case_idx in cases:
        seq, treat, lbl = ds_pre[case_idx]
        seq_b = seq.unsqueeze(0).to(device)
        treat_b = treat.unsqueeze(0).to(device)
        
        exp = explain_prediction(model_A, seq_b, treat_b, label_idx,
                                  ds_pre.seq_cols, ds_pre.treat_cols)
        query = features_to_query_text(exp["top_features"], label_name)
        retrieved = retrieve_pubmed(query, era_filter="pre_covid", k=TOP_K_DOCS)
        
        m = compute_retrieval_metrics_domain(retrieved)
        pubmed_pre_metrics[label_name].append(m)
        
        pubmed_pre_explanations.append({
            "stay_id": int(ds_pre.stay_ids[case_idx]),
            "era": "pre_drift",
            "label": label_name,
            "true_label": int(lbl[label_idx].item()),
            "probability": exp["predicted_probability"],
            "query": query,
            "top_features": exp["top_features"],
            "retrieved": retrieved,
            "metrics": m,
        })

# Save explanations
with open(SAVE_PATH / "pubmed_rag_explanations_post.json", "w") as f:
    json.dump(pubmed_post_explanations, f, indent=2)
with open(SAVE_PATH / "pubmed_rag_explanations_pre.json", "w") as f:
    json.dump(pubmed_pre_explanations, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: AGGREGATE AND REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("PUBMED RAG RETRIEVAL RESULTS")
print(f"Corpus: {len(corpus_entries)} PubMed abstracts ({n_relevant} relevant + {n_noise} noise)")
print(f"Embedding: {EMBEDDING_MODEL}")
print("="*70)

def aggregate_and_print(title, metrics_dict):
    print(f"\n{title}")
    print(f"  {'Label':<22} {'Hit@1':>8} {'Hit@3':>8} {'Hit@5':>8} {'P@3':>8} {'P@5':>8} {'MRR':>8} {'NI@5':>8}")
    print("  " + "-"*72)
    
    agg = {}
    for label, case_list in metrics_dict.items():
        if not case_list: continue
        agg[label] = {}
        for key in ["hit@1","hit@3","hit@5","precision@3","precision@5",
                     "reciprocal_rank","noise_intrusion@5"]:
            agg[label][key] = float(np.mean([m[key] for m in case_list]))
    
    for lbl in LABEL_COLS:
        if lbl in agg:
            m = agg[lbl]
            print(f"  {lbl:<22} {m['hit@1']:>8.3f} {m['hit@3']:>8.3f} {m['hit@5']:>8.3f} "
                  f"{m['precision@3']:>8.3f} {m['precision@5']:>8.3f} {m['reciprocal_rank']:>8.3f} "
                  f"{m['noise_intrusion@5']:>8.3f}")
    
    all_cases = []
    for cases in metrics_dict.values():
        all_cases.extend(cases)
    if all_cases:
        overall = {k: float(np.mean([m[k] for m in all_cases]))
                   for k in ["hit@1","hit@3","hit@5","precision@3","precision@5",
                             "reciprocal_rank","noise_intrusion@5"]}
        print(f"  {'OVERALL':<22} {overall['hit@1']:>8.3f} {overall['hit@3']:>8.3f} "
              f"{overall['hit@5']:>8.3f} {overall['precision@3']:>8.3f} "
              f"{overall['precision@5']:>8.3f} {overall['reciprocal_rank']:>8.3f} "
              f"{overall['noise_intrusion@5']:>8.3f}")
        return agg, overall
    return agg, {}

post_agg, post_overall = aggregate_and_print("Post-drift (PubMed retrieval)", pubmed_post_metrics)
pre_agg, pre_overall   = aggregate_and_print("Pre-drift (PubMed retrieval)", pubmed_pre_metrics)

# Save metrics
pubmed_rag_metrics = {
    "corpus_source": "PubMed (NCBI E-utilities)",
    "corpus_size": len(corpus_entries),
    "n_relevant": n_relevant,
    "n_noise": n_noise,
    "embedding_model": EMBEDDING_MODEL,
    "top_k": TOP_K_DOCS,
    "n_cases_per_condition": N_CASES_PER_CONDITION,
    "post_drift_per_label": post_agg,
    "post_drift_overall": post_overall,
    "pre_drift_per_label": pre_agg,
    "pre_drift_overall": pre_overall,
}
with open(SAVE_PATH / "pubmed_rag_metrics.json", "w") as f:
    json.dump(pubmed_rag_metrics, f, indent=2)
print(f"\nMetrics saved → {SAVE_PATH / 'pubmed_rag_metrics.json'}")


# ── Sample retrieved articles ─────────────────────────────────────────────────
print("\n" + "="*70)
print("SAMPLE RETRIEVALS (first case per label)")
print("="*70)
for exp in pubmed_post_explanations[:3]:
    print(f"\n{exp['label']} (stay {exp['stay_id']}, prob={exp['probability']:.3f}):")
    print(f"  Query: {exp['query'][:100]}...")
    for i, r in enumerate(exp["retrieved"][:3]):
        domain_tag = "✅ ICU" if r["domain"] == "relevant" else "❌ noise"
        print(f"  [{i+1}] {domain_tag} | sim={r['relevance_score']:.3f} | {r['title'][:80]}...")
        print(f"       {r['source']}")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Corpus: {len(corpus_entries)} real PubMed abstracts")
print(f"  Relevant (ICU/sepsis/vent): {n_relevant}")
print(f"  Noise (other domains): {n_noise}")
print(f"  Ratio: 1:{n_noise/max(n_relevant,1):.1f} (signal:noise)")
print(f"Embedding: {EMBEDDING_MODEL}")
if post_overall:
    print(f"\nPost-drift: Hit@1={post_overall.get('hit@1',0):.3f} "
          f"P@3={post_overall.get('precision@3',0):.3f} "
          f"MRR={post_overall.get('reciprocal_rank',0):.3f} "
          f"Noise@5={post_overall.get('noise_intrusion@5',0):.3f}")
if pre_overall:
    print(f"Pre-drift:  Hit@1={pre_overall.get('hit@1',0):.3f} "
          f"P@3={pre_overall.get('precision@3',0):.3f} "
          f"MRR={pre_overall.get('reciprocal_rank',0):.3f} "
          f"Noise@5={pre_overall.get('noise_intrusion@5',0):.3f}")

print("\n✅ PubMed RAG complete. Output files:")
print(f"  {SAVE_PATH / 'pubmed_corpus_raw.json'}")
print(f"  {SAVE_PATH / 'pubmed_rag_metrics.json'}")
print(f"  {SAVE_PATH / 'pubmed_rag_explanations_post.json'}")
print(f"  {SAVE_PATH / 'pubmed_rag_explanations_pre.json'}")
print("\nThis replaces the synthetic corpus for your paper.")
print("Report pubmed_rag_metrics.json alongside expanded_rag_metrics.json")
print("to show synthetic→real-world generalization.")
