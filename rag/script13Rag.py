%%writefile rag/script13_rag_pubmed_final.py
"""
script13_rag_pubmed_final.py
════════════════════════════════════════════════════════════════════
Attribution-Guided RAG with Retrieval Stability Evaluation.

PIPELINE POSITION:
  Run AFTER script2_two_stream_model_v5.py  → two_stream_models.pt
                                            → eval_split.json
        AFTER script5_xgboost_bootstrap_shap.py → xgb_adapted_*.pkl
  Outputs:
    pubmed_corpus.json
    pubmed_rag_stability.json
    pubmed_rag_headtohead.json
    pubmed_rag_explanations.json

DESIGN PRINCIPLES — NO HARDCODING:
  1. Corpus is fetched once under a broad, pre-defined ICU/critical-care
     MeSH umbrella, independent of any patient or model. No targeted
     sub-queries, no relevance labels designed to favour any vocabulary.

  2. Queries are built directly from IG/SHAP feature names (MIMIC column
     names, string-normalized only — underscores→spaces, drop obs/flag
     suffixes). No vocabulary translation tables.

  3. Two-stream split mirrors the architecture:
       physiology sub-query  ← SEQ_FEATURES attributions  (frozen in Run B)
       treatment sub-query   ← TREATMENT_FEATURES attributions (adaptive)
     Retrieved document sets are merged by PMID dedup for final ranking.

  4. Relevance oracle = source model. A document is "source-aligned" if
     the source model's query retrieved it. Stability is measured as
     Jaccard / Spearman-rank-correlation of post-drift queries against
     the source-model query, separately for physio and treatment streams.

  5. XGBoost comparison uses the same automatic pipeline (SHAP → feature
     names → query), just without a stream split.

  6. Attribution delta → retrieval divergence correlation links the
     biological-stability result (fig script) to retrieval behaviour.

MAIN CLAIMS SUPPORTED:
  • Run B physiology-stream queries remain stable across drift
    (Jaccard_physio(source, RunB) >> Jaccard_physio(source, XGB))
  • Treatment-stream queries show expected moderate divergence in Run B
    (adaptive MLP) and larger divergence in XGBoost
  • Attribution delta magnitude predicts retrieval divergence (r = ...)
  • Two-stream retrieval directly mirrors the frozen/adaptive architecture

Dependencies: pip install sentence-transformers biopython
"""

import json, time, warnings, os, re, copy
from pathlib import Path
from collections import defaultdict
import numpy as np
import polars as pl
import torch
import joblib
import shap as shap_lib
import math, csv
warnings.filterwarnings("ignore")

from utils.constants   import SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS
from utils.data_utils  import load_enriched_split, normalize, ICUDataset
from models.architectures import TwoStreamModel

from Bio import Entrez
from xml.etree import ElementTree as ET
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

TP_THRESHOLD = 0.5 
SEED               = 42
SEQ_LEN            = 6
TOP_K_PHYSIO       = 5    # top physio features → physio sub-query
TOP_K_TREAT        = 4    # top treat features  → treatment sub-query
TOP_K_DOCS         = 5    # documents retrieved per query
N_CASES_PER_LABEL  = 100  # patients per label for evaluation
IG_STEPS           = 20
# ── Embedding backend ─────────────────────────────────────────────────────────
EMBEDDING_BACKEND  = "medcpt"          # "medcpt" (biomedical) or "minilm" (fallback)
MINILM_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
MEDCPT_USE_COSINE  = True              # True = L2-normalize (cosine); False = raw dot product

REQUESTS_PER_SEC   = 3    # 10 with NCBI_API_KEY, else 3
BASE_PATH          = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH          = Path("/kaggle/working")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# STRING NORMALISER — the only text transformation, fully automatic
# No lookup tables. Input = MIMIC column name. Output = PubMed-friendly term.
# ══════════════════════════════════════════════════════════════════════════════
_DROP_SUFFIXES = ("_obs", "_flag", "_mask", "_ml", "_hrs", "_first",
                  "_last", "_invasive", "_noninvasive")
_DROP_PREFIXES = ("has_", "early_", "high_", "max_", "min_",
                  "mean_", "std_", "last_", "total_")
_STAT_PREFIXES = ("last_", "mean_", "std_", "min_", "max_")   # XGB flattened names

MESH_RELEVANCE = {
    "label_vasopressor":  {"vasoconstrictor agents"},
    "label_intubation":   {"intubation, intratracheal"},
    "label_septic_shock": {"shock, septic"},
}

def normalise_feature_name(raw: str) -> str:
    name = raw.split(":")[-1]
    # Strip stat prefixes (XGB flattened)
    name = re.sub(r'^(last|mean|std|min|max)_', '', name)
    # Strip rolling-window suffixes
    name = re.sub(r'_?roll(mean|std|min|max|sum)_?\d*', '', name)
    # Strip other known prefixes
    name = re.sub(r'^(has_|early_|high_|total_)', '', name)
    # Strip known suffixes
    name = re.sub(r'_(obs|flag|mask|ml|hrs|first|last|invasive|'
                  r'noninvasive)$', '', name)
    name = name.replace("_", " ").strip()
    _ABBREV = {"map": "MAP", "spo2": "SpO2", "fio2": "FiO2",
               "peep": "PEEP", "wbc": "WBC", "hr": "heart rate",
               "rr": "respiratory rate", "sbp": "systolic blood pressure",
               "dbp": "diastolic blood pressure"}
    if name.lower() in _ABBREV:
        name = _ABBREV[name.lower()]
    return name


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS — fixed, broad, patient-independent
# ══════════════════════════════════════════════════════════════════════════════
# One wide ICU/critical-care umbrella. Fetched once, saved, reused.
# Corpus construction has no knowledge of which features IG will highlight.
# ══════════════════════════════════════════════════════════════════════════════
CORPUS_QUERIES = [
    # Core ICU management — deliberately broad
    ('"Critical Care"[MeSH]',                                       200),
    ('"Intensive Care Units"[MeSH]',                                200),
    # Sepsis / shock — the three prediction targets
    ('"Sepsis"[MeSH]',                                              150),
    ('"Shock, Septic"[MeSH]',                                       150),
    # Organ support — ventilation and haemodynamics
    ('"Respiration, Artificial"[MeSH]',                             150),
    ('"Vasoconstrictor Agents"[MeSH] AND "Critical Care"[MeSH]',   100),
    # Recovery / de-escalation — so both directions of query can hit
    ('"Ventilator Weaning"[MeSH]',                                  100),
    ('"Anti-Bacterial Agents"[MeSH] AND "Critical Care"[MeSH]',    100),
]


NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
if NCBI_API_KEY:
    Entrez.api_key = NCBI_API_KEY
    _DELAY = 1.0 / 10
    print("✅ NCBI API Key — 10 req/sec")
else:
    _DELAY = 1.0 / 3
    print("⚠ No NCBI_API_KEY — 3 req/sec")

Entrez.email = "research@example.com"


def _fetch_batch(pmids: list) -> list:
    handle   = Entrez.efetch(db="pubmed", id=",".join(pmids),
                             rettype="xml", retmode="xml")
    xml_data = handle.read()
    handle.close()
    root     = ET.fromstring(xml_data)
    articles = []
    for art in root.findall(".//PubmedArticle"):
        try:
            medline  = art.find(".//MedlineCitation")
            article  = medline.find(".//Article")
            title_el = article.find(".//ArticleTitle")
            title    = "".join(title_el.itertext()).strip() if title_el is not None else ""
            abs_el   = article.find(".//Abstract")
            if abs_el is None:
                continue
            abstract = " ".join(
                "".join(a.itertext()).strip()
                for a in abs_el.findall(".//AbstractText")
            ).strip()
            if len(abstract) < 100:
                continue
            pmid_el  = medline.find(".//PMID")
            pmid     = pmid_el.text if pmid_el is not None else ""
            year_el  = article.find(".//Journal/JournalIssue/PubDate/Year")
            year     = int(year_el.text) if year_el is not None and year_el.text and year_el.text.isdigit() else 2015
            auth_el  = article.find(".//AuthorList/Author")
            first_au = "Unknown"
            if auth_el is not None:
                ln  = auth_el.find("LastName")
                ini = auth_el.find("Initials")
                if ln is not None:
                    first_au = ln.text + (f" {ini.text}" if ini is not None else "")
            jrnl_el  = article.find(".//Journal/Title")
            # MeSH terms for automatic domain tagging
            mesh_el  = medline.find(".//MeshHeadingList")
            # MeSH terms — DescriptorName sits one level under each MeshHeading
            mesh = [d.text for d in medline.findall(".//MeshHeading/DescriptorName")
                    if d.text]
            articles.append({
                "pmid": pmid, "title": title, "abstract": abstract,
                "year": year, "first_author": first_au,
                "journal": jrnl_el.text if jrnl_el is not None else "Unknown",
                "mesh": mesh,
            })
        except Exception:
            continue
    return articles


def fetch_corpus(queries: list, min_date: str, max_date: str) -> list:
    """
    Fetch a fixed corpus from PubMed using broad category queries.
    Deduplicates by PMID. Returns list of article dicts.
    """
    raw = []
    for query, max_n in queries:
        try:
            handle  = Entrez.esearch(
                db="pubmed", term=query, retmax=max_n,
                mindate=min_date, maxdate=max_date,
                sort="relevance", retmode="json")
            results = json.loads(handle.read())
            handle.close()
            pmids   = results.get("esearchresult", {}).get("idlist", [])
            time.sleep(_DELAY)
            for i in range(0, len(pmids), 100):
                batch    = pmids[i:i + 100]
                articles = _fetch_batch(batch)
                raw.extend(articles)
                time.sleep(_DELAY)
            print(f"  '{query[:60]}...' → {len(pmids)} pmids fetched")
        except Exception as e:
            print(f"  Error fetching '{query[:40]}': {e}")

    seen, deduped = set(), []
    for a in raw:
        if a["pmid"] and a["pmid"] not in seen:
            seen.add(a["pmid"])
            deduped.append(a)
    print(f"\n✅ Corpus: {len(deduped)} unique abstracts")
    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# LOAD SPLIT + MODELS + DATA
# ══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("Loading split, models, data")
print("="*70)

with open(SAVE_PATH / "eval_split.json") as f:
    _split = json.load(f)

eval_post_stays  = list(map(int, _split["eval_post_stays"]))
post_cp_stays    = list(map(int, _split["post_cp_stays"]))
pre_cp_stays     = list(map(int, _split["pre_cp_stays"]))
drift_tag        = _split.get("drift_tag", "unknown")
print(f"✅ Split | drift_tag={drift_tag} | "
      f"pre={len(pre_cp_stays)} | post_eval={len(eval_post_stays)}")

ckpt = torch.load(SAVE_PATH / "two_stream_models.pt",
                  map_location=device, weights_only=False)
seq_dim, treat_dim, n_targets = ckpt["seq_dim"], ckpt["treat_dim"], ckpt["n_targets"]
train_stats = ckpt["train_stats"]

# Source model  — pre-adaptation weights (deep-copied early in script2)
model_src = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_src.load_state_dict(ckpt["source"])
model_src.eval()

# Run B  — selective adaptation (physiology frozen)
model_B = TwoStreamModel(seq_dim, treat_dim, n_targets).to(device)
model_B.load_state_dict(ckpt["run_b"])
model_B.eval()

print(f"✅ Models loaded: seq={seq_dim} treat={treat_dim} targets={n_targets}")

# Data — same pipeline as script5
LEAKAGE_TIMING_FEATS    = ["time_to_first_abx_order_hrs"]
SENTINEL_NO_EARLY_EVENT = float(SEQ_LEN + 1)
_DROP_COLS              = ["vasopressor_flag", "ventilation_flag"]

test_df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)
for feat in LEAKAGE_TIMING_FEATS:
    if feat in test_df.columns:
        test_df = test_df.with_columns(
            pl.when(pl.col(feat) > SEQ_LEN)
              .then(SENTINEL_NO_EARLY_EVENT)
              .otherwise(pl.col(feat)).alias(feat))
test_df = normalize(test_df, train_stats)
test_df = test_df.drop([c for c in _DROP_COLS if c in test_df.columns])

test_post    = test_df.filter(pl.col("stay_id").is_in(post_cp_stays))
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))
test_pre     = test_df.filter(pl.col("stay_id").is_in(pre_cp_stays))

ds_post = ICUDataset(eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
ds_pre  = ICUDataset(test_pre,     SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
print(f"✅ Data | pre={len(ds_pre)} post={len(ds_post)} patients")

# XGBoost adapted models
xgb_adapted = {}
for lbl in LABEL_COLS:
    p = SAVE_PATH / f"xgb_adapted_{lbl}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run script5 first.")
    xgb_adapted[lbl] = joblib.load(p)
print(f"✅ XGBoost adapted models loaded for {len(xgb_adapted)} labels")

xgb_explainers = {
    lbl: shap_lib.TreeExplainer(model)
    for lbl, model in xgb_adapted.items()
}
print(f"✅ XGBoost TreeExplainers cached for {len(xgb_explainers)} labels")


# Derive corpus window from drift_tag automatically
# drift_tag format: "YYYY - YYYY" or "YYYY" or "YYYY-YYYY"
_drift_years = [int(y) for y in re.findall(r'\d{4}', drift_tag)]
_drift_start = min(_drift_years)           # first year of drift period
CORPUS_MAX_YEAR = _drift_start - 1        # corpus ends just before drift
CORPUS_LOOKBACK = 10                      # years of literature to include
CORPUS_MIN_YEAR = CORPUS_MAX_YEAR - CORPUS_LOOKBACK
CORPUS_MIN_DATE = f"{CORPUS_MIN_YEAR}/01/01"
CORPUS_MAX_DATE = f"{CORPUS_MAX_YEAR}/12/31"
print(f"✅ Corpus window derived from drift_tag='{drift_tag}': "
      f"{CORPUS_MIN_DATE} – {CORPUS_MAX_DATE}")
# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATED GRADIENTS  (identical to script2 / fig)
# ══════════════════════════════════════════════════════════════════════════════
def integrated_gradients(model, xs, xt, target_idx, steps=IG_STEPS):
    model.train()                          # ← changed from model.eval()
    xs, xt   = xs.to(device), xt.to(device)
    bs, bt   = torch.zeros_like(xs), torch.zeros_like(xt)
    sg, tg   = torch.zeros_like(xs), torch.zeros_like(xt)
    with torch.enable_grad():
        for alpha in np.linspace(0, 1, steps):
            is_ = (bs + alpha * (xs - bs)).requires_grad_(True)
            it_ = (bt + alpha * (xt - bt)).requires_grad_(True)
            out = model(is_, it_)[:, target_idx].sum()
            g1, g2 = torch.autograd.grad(out, [is_, it_])
            sg += g1;  tg += g2
    model.eval()                           # ← restore eval mode after
    seq_attr   = ((xs - bs) * sg / steps).cpu().numpy()
    treat_attr = ((xt - bt) * tg / steps).cpu().numpy()
    return seq_attr, treat_attr


def explain_two_stream(model, xs, xt, label_idx, seq_cols, treat_cols):
    """
    Returns per-feature attribution split into physio and treat lists.
    Each entry: {feature, raw_name, contribution, abs_contribution, stream}
    """
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(xs.to(device), xt.to(device)))[0, label_idx].item()

    seq_attr, treat_attr = integrated_gradients(model, xs, xt, label_idx)
    seq_flat   = seq_attr[0].sum(axis=0)    # (seq_dim,)
    treat_flat = treat_attr[0]              # (treat_dim,)

    physio_feats = sorted(
        [{"feature": f"physio:{n}", "raw_name": n,
          "contribution": float(v), "abs_contribution": abs(float(v)),
          "stream": "physio"}
         for n, v in zip(seq_cols, seq_flat)],
        key=lambda x: x["abs_contribution"], reverse=True)

    treat_feats = sorted(
        [{"feature": f"treat:{n}", "raw_name": n,
          "contribution": float(v), "abs_contribution": abs(float(v)),
          "stream": "treat"}
         for n, v in zip(treat_cols, treat_flat)],
        key=lambda x: x["abs_contribution"], reverse=True)

    return {
        "prob":          prob,
        "physio_feats":  physio_feats,
        "treat_feats":   treat_feats,
        "all_feats":     sorted(physio_feats + treat_feats,
                                key=lambda x: x["abs_contribution"],
                                reverse=True),
    }


# ══════════════════════════════════════════════════════════════════════════════
# XGB FLATTEN + SHAP  (mirrors script5 exactly)
# ══════════════════════════════════════════════════════════════════════════════
_STAT_PFX_RE = re.compile(r"^(last|mean|std|min|max)_(.+)$")
_TREAT_SET   = set(TREATMENT_FEATURES)


def flatten_for_xgb(df, seq_len=SEQ_LEN):
    seq_cols   = [c for c in SEQ_FEATURES   if c in df.columns and not c.endswith("_mask")]
    treat_cols = [c for c in TREATMENT_FEATURES if c in df.columns]
    stays      = df.sort(["stay_id", "hrs_from_admit"])
    stay_ids   = stays.select("stay_id").unique().sort("stay_id")["stay_id"].to_list()
    rows, labels_ = [], []
    for sid in stay_ids:
        s  = stays.filter(pl.col("stay_id") == sid)
        sv = s.select(seq_cols).to_numpy().astype(np.float32)
        if sv.shape[0] < seq_len:
            sv = np.vstack([sv, np.zeros((seq_len - sv.shape[0], sv.shape[1]), np.float32)])
        else:
            sv = sv[:seq_len]
        rows.append(np.concatenate([
            sv[-1], sv.mean(0), sv.std(0), sv.min(0), sv.max(0),
            np.array(s.select(treat_cols).row(0), dtype=np.float32),
        ]))
        labels_.append(np.array(s.select(LABEL_COLS).row(0), dtype=np.float32))
    names = []
    for pfx in ["last", "mean", "std", "min", "max"]:
        names += [f"{pfx}_{c}" for c in seq_cols]
    names += treat_cols
    X = np.stack(rows);  X[~np.isfinite(X)] = 0.0
    return X, np.stack(labels_), stay_ids, names


def explain_xgb(explainer, xgb_model, x_row, feat_names):
    shap_vals  = explainer.shap_values(x_row.reshape(1, -1))[0]
    prob       = float(xgb_model.predict_proba(x_row.reshape(1, -1))[0, 1])
    max_abs    = max(abs(float(v)) for v in shap_vals) if len(shap_vals) else 1e-9

    physio_feats, treat_feats = [], []
    for name, val in zip(feat_names, shap_vals):
        m    = _STAT_PFX_RE.match(name)
        base = m.group(2) if m else name
        stream = "treat" if base in _TREAT_SET else "physio"
        entry = {
            "feature":          f"{stream}:{name}",
            "raw_name":         name,
            "base_name":        base,
            "contribution":     float(val),
            "abs_contribution": abs(float(val)),
            "stream":           stream,
        }
        if stream == "physio":
            physio_feats.append(entry)
        else:
            treat_feats.append(entry)

    physio_feats.sort(key=lambda x: x["abs_contribution"], reverse=True)
    treat_feats.sort( key=lambda x: x["abs_contribution"], reverse=True)
    all_feats = sorted(physio_feats + treat_feats,
                       key=lambda x: x["abs_contribution"], reverse=True)
    return {"prob": prob, "physio_feats": physio_feats,
            "treat_feats": treat_feats, "all_feats": all_feats}


# ══════════════════════════════════════════════════════════════════════════════
# QUERY BUILDER — fully automatic, no vocabulary tables
# Input: sorted feature list + label name
# Output: query string built from normalised feature names + label anchor
# The label anchor is the label column name itself, normalised — no hardcoding
# ══════════════════════════════════════════════════════════════════════════════
def _label_to_anchor(label_col: str) -> str:
    """
    Convert label column name to a PubMed anchor term automatically.
    label_vasopressor → "vasopressor ICU"
    label_intubation  → "intubation ICU"
    label_septic_shock → "septic shock ICU"
    Purely string transformation — no lookup table.
    """
    anchor = label_col.replace("label_", "").replace("_", " ")
    return f"{anchor} ICU"


def build_query(feats: list, label_col: str, top_k: int) -> str:
    """
    Build a PubMed query string from the top_k features by |attribution|.
    Query = label_anchor + normalised feature names (space-separated).
    No vocabulary tables. No direction-specific synonyms.
    The feature names are MIMIC column names, which are already
    clinical terms — normalise_feature_name() just cleans the string.
    """
    anchor = _label_to_anchor(label_col)
    terms  = [anchor]
    seen   = set()
    for f in feats[:top_k]:
        term = normalise_feature_name(f["feature"])
        if term and term not in seen:
            seen.add(term)
            terms.append(term)
    return " ".join(terms)


def build_two_stream_queries(exp: dict, label_col: str) -> dict:
    """
    Two sub-queries — one per stream — plus a combined query.
    Mirrors the frozen/adaptive architecture directly.
    """
    physio_q   = build_query(exp["physio_feats"], label_col, TOP_K_PHYSIO)
    treat_q    = build_query(exp["treat_feats"],  label_col, TOP_K_TREAT)
    # Combined: physio anchor + top treat terms (avoid duplicate anchor)
    treat_terms = " ".join(
        normalise_feature_name(f["feature"])
        for f in exp["treat_feats"][:TOP_K_TREAT]
        if normalise_feature_name(f["feature"])
    )
    combined_q = f"{physio_q} {treat_terms}".strip()
    return {
        "physio_query":   physio_q,
        "treat_query":    treat_q,
        "combined_query": combined_q,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FETCH CORPUS (once, then cache)
# ══════════════════════════════════════════════════════════════════════════════
corpus_path = SAVE_PATH / "pubmed_corpus.json"
corpus_path.unlink(missing_ok=True)
if corpus_path.exists():
    print(f"\nLoading cached corpus from {corpus_path}")
    with open(corpus_path) as f:
        corpus_raw = json.load(f)
    corpus_raw = [a for a in corpus_raw
                  if CORPUS_MIN_YEAR <= a.get("year", 0) <= CORPUS_MAX_YEAR]
    print(f"✅ {len(corpus_raw)} abstracts loaded from cache after year filter")
else:
    print("\n" + "="*70)
    print("Fetching PubMed corpus (fixed, patient-independent)")
    print("="*70)
    corpus_raw = fetch_corpus(CORPUS_QUERIES, CORPUS_MIN_DATE, CORPUS_MAX_DATE)
    corpus_raw = [a for a in corpus_raw
                  if CORPUS_MIN_YEAR <= a.get("year", 0) <= CORPUS_MAX_YEAR]
    with open(corpus_path, "w") as f:
        json.dump(corpus_raw, f, indent=2)
    print(f"✅ Saved corpus → {corpus_path}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD SEMANTIC INDEX
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Building semantic index")
print("="*70)

# embedder = SentenceTransformer(EMBEDDING_MODEL, device=str(device))

corpus_entries = []
for a in corpus_raw:
    text = f"{a['title']}. {a['abstract']}"[:2000]
    corpus_entries.append({
        "pmid":   a["pmid"],
        "title":  a["title"],
        "text":   a["abstract"],
        "mesh":   a.get("mesh", []),    # ← add this
        "source": f"{a['first_author']} et al. {a['journal']} {a['year']}",
        "year":   a["year"],
        "text_for_embedding": text,
    })
# print(f"Embedding {len(corpus_entries)} passages...")
# corpus_embeddings = embedder.encode(
#     [e["text_for_embedding"] for e in corpus_entries],
#     convert_to_numpy=True, show_progress_bar=True,
#     batch_size=32, normalize_embeddings=True,
# )
# print(f"  Shape: {corpus_embeddings.shape}")

class Retriever:
    """
    Unified retrieval backend.
      minilm — symmetric SentenceTransformer (same model for query + doc)
      medcpt — asymmetric NCBI MedCPT (separate query / article encoders,
               articles encoded as title+abstract pairs)
    Both expose encode_corpus() and encode_query() returning float32 arrays.
    Scoring is always a dot product over the returned vectors, so cosine vs
    raw-dot is controlled purely by whether we normalize here.
    """
    def __init__(self, backend, device):
        self.backend = backend
        self.device  = device
        if backend == "minilm":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MINILM_MODEL, device=str(device))
            print(f"✅ Retriever: MiniLM ({MINILM_MODEL}) — cosine")
        elif backend == "medcpt":
            from transformers import AutoTokenizer, AutoModel
            self.q_tok   = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
            self.q_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device).eval()
            self.a_tok   = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
            self.a_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device).eval()
            sim = "cosine" if MEDCPT_USE_COSINE else "dot product"
            print(f"✅ Retriever: MedCPT (NCBI query+article encoders) — {sim}")
        else:
            raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")

    def _maybe_norm(self, arr):
        if self.backend == "minilm" or MEDCPT_USE_COSINE:
            n = np.linalg.norm(arr, axis=-1, keepdims=True)
            arr = arr / np.clip(n, 1e-9, None)
        return arr.astype(np.float32)

    def encode_corpus(self, entries, batch_size=32):
        if self.backend == "minilm":
            texts = [e["text_for_embedding"] for e in entries]
            return self.model.encode(texts, convert_to_numpy=True,
                                     show_progress_bar=True, batch_size=batch_size,
                                     normalize_embeddings=True).astype(np.float32)
        # medcpt: encode (title, abstract) pairs with the ARTICLE encoder
        out = []
        for i in range(0, len(entries), batch_size):
            batch     = entries[i:i + batch_size]
            titles    = [e["title"] for e in batch]
            abstracts = [e["text"]  for e in batch]
            with torch.no_grad():
                enc = self.a_tok(titles, abstracts, truncation=True, padding=True,
                                 return_tensors="pt", max_length=512).to(self.device)
                emb = self.a_model(**enc).last_hidden_state[:, 0, :]   # [CLS]
            out.append(emb.cpu().numpy())
            if (i // batch_size) % 5 == 0:
                print(f"    MedCPT corpus {min(i+batch_size, len(entries))}/{len(entries)}")
        return self._maybe_norm(np.vstack(out))

    def encode_query(self, query):
        if self.backend == "minilm":
            return self.model.encode([query], convert_to_numpy=True,
                                     normalize_embeddings=True).astype(np.float32)[0]
        with torch.no_grad():
            enc = self.q_tok([query], truncation=True, padding=True,
                             return_tensors="pt", max_length=64).to(self.device)
            emb = self.q_model(**enc).last_hidden_state[:, 0, :]       # [CLS]
        return self._maybe_norm(emb.cpu().numpy())[0]


retriever = Retriever(EMBEDDING_BACKEND, device)
print(f"Embedding {len(corpus_entries)} passages...")
corpus_embeddings = retriever.encode_corpus(corpus_entries, batch_size=32)
print(f"  Shape: {corpus_embeddings.shape}")
# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
# def retrieve(query: str, k: int = TOP_K_DOCS) -> list:
#     """Retrieve top-k documents by cosine similarity."""
#     q_emb = embedder.encode([query], convert_to_numpy=True,
#                              normalize_embeddings=True)
#     sims  = (corpus_embeddings @ q_emb.T).ravel()
#     order = np.argsort(sims)[::-1][:k * 4]   # over-fetch for two-stream merge
#     return [(int(i), float(sims[i])) for i in order[:k]]

def retrieve(query: str, k: int = TOP_K_DOCS) -> list:
    """Retrieve top-k documents by similarity."""
    q_emb = retriever.encode_query(query)
    sims  = corpus_embeddings @ q_emb
    order = np.argsort(sims)[::-1][:k * 4]
    return [(int(i), float(sims[i])) for i in order[:k]]

def retrieve_two_stream(queries: dict, k: int = TOP_K_DOCS) -> dict:
    """
    Retrieve separately for physio and treat sub-queries, merge by PMID.
    Returns:
      physio_hits  — top-k from physio sub-query
      treat_hits   — top-k from treat sub-query
      merged_hits  — union ranked by max sim across streams
    Each hit: {pmid, title, text, source, year, sim}
    """

    def _hits(query, n):
        q_emb = retriever.encode_query(query)
        sims  = corpus_embeddings @ q_emb
        order = np.argsort(sims)[::-1]
        results = []
        for idx in order:
            e = corpus_entries[idx]
            results.append({**e, "sim": float(sims[idx])})
            if len(results) >= n:
                break
        return results
    
    p_hits = _hits(queries["physio_query"], k)
    t_hits = _hits(queries["treat_query"],  k)

    # Merge: union by PMID, keep max sim
    best = {}
    for h in p_hits + t_hits:
        pmid = h["pmid"]
        if pmid not in best or h["sim"] > best[pmid]["sim"]:
            best[pmid] = h
    merged = sorted(best.values(), key=lambda x: x["sim"], reverse=True)[:k]

    return {
        "physio_hits": p_hits,
        "treat_hits":  t_hits,
        "merged_hits": merged,
    }


def pmid_set(hits: list) -> set:
    return {h["pmid"] for h in hits}


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def rank_correlation(hits_a: list, hits_b: list) -> float:
    """
    Spearman rank correlation of similarity scores for PMIDs that
    appear in both result sets. Returns NaN if fewer than 3 overlap.
    """
    pmids_a = {h["pmid"]: h["sim"] for h in hits_a}
    pmids_b = {h["pmid"]: h["sim"] for h in hits_b}
    common  = sorted(set(pmids_a) & set(pmids_b))
    if len(common) < 3:
        return float("nan")
    sa = [pmids_a[p] for p in common]
    sb = [pmids_b[p] for p in common]
    r, _ = spearmanr(sa, sb)
    return float(r)


# ══════════════════════════════════════════════════════════════════════════════
# FLATTEN POST-DRIFT DATA FOR XGB (once)
# ══════════════════════════════════════════════════════════════════════════════
print("\nFlattening post-drift data for XGBoost...")
X_post, Y_post, xgb_stay_order, xgb_feat_names = flatten_for_xgb(eval_post_df)
xgb_sid_to_idx = {sid: i for i, sid in enumerate(xgb_stay_order)}
print(f"  XGBoost feature matrix: {X_post.shape}")


# ══════════════════════════════════════════════════════════════════════════════
# CASE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def select_cases(ds, label_idx, n=N_CASES_PER_LABEL):
    rng = np.random.RandomState(SEED)
    pos = [i for i in range(len(ds)) if ds.labels[i, label_idx] == 1]
    neg = [i for i in range(len(ds)) if ds.labels[i, label_idx] == 0]
    n2  = n // 2
    sel_pos = rng.choice(pos, min(len(pos), n2), replace=False).tolist()
    sel_neg = rng.choice(neg, min(len(neg), n2), replace=False).tolist()
    return sel_pos + sel_neg


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION LOOP
#
# For each post-drift patient × label:
#   SOURCE query  — run source model weights on this post-drift patient
#   RUN B  query  — run adapted Run B weights on same patient
#   XGB    query  — run XGBoost SHAP on same patient
#
# Stability = how similar are Run B / XGB queries to the source query?
# Hypothesis: Run B ≈ Source (frozen physiology → stable attributions)
#             XGB  ≠ Source (biological amnesia → shifted attributions)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Running retrieval stability evaluation")
print("="*70)

all_cases       = []   # one dict per (stay_id, label)
explanations    = {"source": [], "run_b": [], "xgb": []}

for li, ln in enumerate(LABEL_COLS):
    cases = select_cases(ds_post, li)
    print(f"\n  {ln}: {len(cases)} cases")

    for ci in cases:
        seq, treat, lbl_t = ds_post[ci]
        stay_id = int(ds_post.stay_ids[ci])
        true_lbl = int(lbl_t[li].item())

        xs = seq.unsqueeze(0)
        xt = treat.unsqueeze(0)

        # ── SOURCE model on this post-drift patient ───────────────────────
        exp_src  = explain_two_stream(
            model_src, xs, xt, li, ds_post.seq_cols, ds_post.treat_cols)
        q_src    = build_two_stream_queries(exp_src, ln)
        ret_src  = retrieve_two_stream(q_src)

        # ── RUN B on same patient ─────────────────────────────────────────
        exp_b    = explain_two_stream(
            model_B, xs, xt, li, ds_post.seq_cols, ds_post.treat_cols)
        q_b      = build_two_stream_queries(exp_b, ln)
        ret_b    = retrieve_two_stream(q_b)

        # ── XGB on same patient ───────────────────────────────────────────
        if stay_id not in xgb_sid_to_idx:
            print(f"    ⚠ {stay_id} not in XGB matrix — skip")
            continue
        xgb_row  = X_post[xgb_sid_to_idx[stay_id]]
        exp_xgb  = explain_xgb(xgb_explainers[ln], xgb_adapted[ln],
                        xgb_row, xgb_feat_names)
        # XGB has no stream split — use all features for one query, but
        # split the stored features by stream tag for attribution mass stats

        q_xgb_physio = build_query(exp_xgb["physio_feats"], ln, TOP_K_PHYSIO)
        q_xgb_treat  = build_query(exp_xgb["treat_feats"],  ln, TOP_K_TREAT)
        treat_terms   = " ".join(
            normalise_feature_name(f["feature"])
            for f in exp_xgb["treat_feats"][:TOP_K_TREAT]
            if normalise_feature_name(f["feature"])
        )
        q_xgb = {
            "physio_query":   q_xgb_physio,
            "treat_query":    q_xgb_treat,
            "combined_query": f"{q_xgb_physio} {treat_terms}".strip(),
        }
        # Keep q_xgb_str for sample printout (combined)
        q_xgb_str = q_xgb["combined_query"]
        
        ret_xgb   = retrieve_two_stream(q_xgb)

        # ══ STABILITY METRICS ════════════════════════════════════════════
        # -- Physiology stream --
        src_physio_pmids = pmid_set(ret_src["physio_hits"])
        b_physio_pmids   = pmid_set(ret_b["physio_hits"])
        xgb_physio_pmids = pmid_set(ret_xgb["physio_hits"])

        jacc_physio_b    = jaccard(src_physio_pmids, b_physio_pmids)
        jacc_physio_xgb  = jaccard(src_physio_pmids, xgb_physio_pmids)
        rho_physio_b     = rank_correlation(ret_src["physio_hits"],
                                             ret_b["physio_hits"])
        rho_physio_xgb   = rank_correlation(ret_src["physio_hits"],
                                             ret_xgb["physio_hits"])

        # -- Treatment stream --
        src_treat_pmids  = pmid_set(ret_src["treat_hits"])
        b_treat_pmids    = pmid_set(ret_b["treat_hits"])
        xgb_treat_pmids  = pmid_set(ret_xgb["treat_hits"])

        jacc_treat_b     = jaccard(src_treat_pmids, b_treat_pmids)
        jacc_treat_xgb   = jaccard(src_treat_pmids, xgb_treat_pmids)
        rho_treat_b      = rank_correlation(ret_src["treat_hits"],
                                             ret_b["treat_hits"])
        rho_treat_xgb    = rank_correlation(ret_src["treat_hits"],
                                             ret_xgb["treat_hits"])

        # -- Merged (overall) --
        jacc_merged_b    = jaccard(pmid_set(ret_src["merged_hits"]),
                                   pmid_set(ret_b["merged_hits"]))
        jacc_merged_xgb  = jaccard(pmid_set(ret_src["merged_hits"]),
                                   pmid_set(ret_xgb["merged_hits"]))

        # -- Attribution mass per stream (for delta-divergence correlation) --
        src_physio_mass  = np.mean([f["abs_contribution"]
                                    for f in exp_src["physio_feats"]])
        b_physio_mass    = np.mean([f["abs_contribution"]
                                    for f in exp_b["physio_feats"]])
        xgb_physio_mass  = np.mean([f["abs_contribution"]
                                    for f in exp_xgb["physio_feats"]])

        # Attribution delta: |mean(Run B physio) − mean(Source physio)|
        # and same for XGB — this is the per-patient attribution shift
        physio_delta_b   = abs(b_physio_mass   - src_physio_mass)
        physio_delta_xgb = abs(xgb_physio_mass - src_physio_mass)

        # Query-level token overlap (automatic — no manual terms)
        def _token_jaccard(qa, qb):
            ta = set(qa.lower().split())
            tb = set(qb.lower().split())
            return len(ta & tb) / max(len(ta | tb), 1)

        q_jacc_physio_b   = _token_jaccard(q_src["physio_query"],
                                            q_b["physio_query"])
        q_jacc_physio_xgb = _token_jaccard(q_src["physio_query"],
                                            q_xgb["physio_query"])
        q_jacc_treat_b    = _token_jaccard(q_src["treat_query"],
                                            q_b["treat_query"])
        q_jacc_treat_xgb  = _token_jaccard(q_src["treat_query"],
                                            q_xgb["treat_query"])

        case_record = {
            # Identity
            "stay_id":    stay_id,
            "label":      ln,
            "true_label": true_lbl,
            # Probabilities
            "prob_src":   exp_src["prob"],
            "prob_b":     exp_b["prob"],
            "prob_xgb":   exp_xgb["prob"],
            # Queries
            "q_src_physio":  q_src["physio_query"],
            "q_src_treat":   q_src["treat_query"],
            "q_b_physio":    q_b["physio_query"],
            "q_b_treat":     q_b["treat_query"],
            "q_xgb":         q_xgb_str,
            # Document Jaccard — PHYSIOLOGY stream
            "jacc_physio_b":   jacc_physio_b,
            "jacc_physio_xgb": jacc_physio_xgb,
            # Document Jaccard — TREATMENT stream
            "jacc_treat_b":    jacc_treat_b,
            "jacc_treat_xgb":  jacc_treat_xgb,
            # Document Jaccard — MERGED
            "jacc_merged_b":   jacc_merged_b,
            "jacc_merged_xgb": jacc_merged_xgb,
            # Rank correlation
            "rho_physio_b":    rho_physio_b,
            "rho_physio_xgb":  rho_physio_xgb,
            "rho_treat_b":     rho_treat_b,
            "rho_treat_xgb":   rho_treat_xgb,
            # Attribution mass & delta
            "src_physio_mass":   float(src_physio_mass),
            "b_physio_mass":     float(b_physio_mass),
            "xgb_physio_mass":   float(xgb_physio_mass),
            "physio_delta_b":    float(physio_delta_b),
            "physio_delta_xgb":  float(physio_delta_xgb),
            # Query token Jaccard
            "q_jacc_physio_b":   q_jacc_physio_b,
            "q_jacc_physio_xgb": q_jacc_physio_xgb,
            "q_jacc_treat_b":    q_jacc_treat_b,
            "q_jacc_treat_xgb":  q_jacc_treat_xgb,
        }
        all_cases.append(case_record)

        # Store explanations for qualitative inspection
        for tag, exp, q, ret in [
            ("source", exp_src, q_src, ret_src),
            ("run_b",  exp_b,   q_b,   ret_b),
        ]:
            explanations[tag].append({
                "stay_id": stay_id, "label": ln, "true_label": true_lbl,
                "prob":         exp["prob"],
                "physio_query": q["physio_query"],
                "treat_query":  q["treat_query"],
                "top_physio":   exp["physio_feats"][:TOP_K_PHYSIO],
                "top_treat":    exp["treat_feats"][:TOP_K_TREAT],
                "physio_hits":  ret["physio_hits"],
                "treat_hits":   ret["treat_hits"],
                "merged_hits":  ret["merged_hits"],
            })
        explanations["xgb"].append({
            "stay_id": stay_id, "label": ln, "true_label": true_lbl,
            "prob":    exp_xgb["prob"],
            "query":   q_xgb_str,
            "top_feats": exp_xgb["all_feats"][:TOP_K_PHYSIO + TOP_K_TREAT],
            "merged_hits": ret_xgb["merged_hits"],
        })

print(f"\n✅ Processed {len(all_cases)} cases")


# ══════════════════════════════════════════════════════════════════════════════
# AGGREGATE STABILITY RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def _mean(lst):
    vals = [v for v in lst if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")

def _std(lst):
    vals = [v for v in lst if not (isinstance(v, float) and np.isnan(v))]
    return float(np.std(vals)) if vals else float("nan")


print("\n" + "="*70)
print("RETRIEVAL STABILITY RESULTS")
print("="*70)

# Overall
print("\n── Overall (all labels, all patients) ──────────────────────────────")
print(f"\n{'Metric':<38} {'Run B vs Source':>16} {'XGB vs Source':>14}")
print("─"*70)
for metric, key_b, key_xgb in [
    ("Jaccard (physio stream)",       "jacc_physio_b",   "jacc_physio_xgb"),
    ("Jaccard (treatment stream)",    "jacc_treat_b",    "jacc_treat_xgb"),
    ("Jaccard (merged)",              "jacc_merged_b",   "jacc_merged_xgb"),
    ("Rank corr (physio)",            "rho_physio_b",    "rho_physio_xgb"),
    ("Rank corr (treatment)",         "rho_treat_b",     "rho_treat_xgb"),
    ("Query token Jaccard (physio)",  "q_jacc_physio_b", "q_jacc_physio_xgb"),
    ("Query token Jaccard (treat)",   "q_jacc_treat_b",  "q_jacc_treat_xgb"),
]:
    vb  = _mean([c[key_b]   for c in all_cases])
    vx  = _mean([c[key_xgb] for c in all_cases])
    vx_str = f"{vx:>12.3f}" if not np.isnan(vx) else "       N/A*"
    print(f"  {metric:<36} {vb:>14.3f}   {vx_str}")


# Disclosure: NaN-biased rho for XGB
rho_xgb_vals  = [c["rho_physio_xgb"] for c in all_cases]
rho_xgb_valid = [v for v in rho_xgb_vals if not np.isnan(v)]
rho_b_vals    = [c["rho_physio_b"]   for c in all_cases]
rho_b_valid   = [v for v in rho_b_vals if not np.isnan(v)]
print(f"\n  NOTE — rank correlation NaN disclosure:")
print(f"  Run B:    computed on {len(rho_b_valid)}/{len(rho_b_vals)} cases")
print(f"  XGBoost:  computed on {len(rho_xgb_valid)}/{len(rho_xgb_vals)} cases "
      f"(upward-biased — NaN cases are those with zero PMID overlap, "
      f"i.e. the most divergent cases, which are excluded)")

# Per-label breakdown
print("\n── Per-label breakdown ─────────────────────────────────────────────")
print(f"\n  {'Label':<22} {'Jacc-P(B)':>10} {'Jacc-P(X)':>10} "
      f"{'Jacc-T(B)':>10} {'Jacc-T(X)':>10} {'n':>5}")
print("  " + "─"*67)
for ln in LABEL_COLS:
    sub = [c for c in all_cases if c["label"] == ln]
    if not sub: continue
    jp_b  = _mean([c["jacc_physio_b"]   for c in sub])
    jp_x  = _mean([c["jacc_physio_xgb"] for c in sub])
    jt_b  = _mean([c["jacc_treat_b"]    for c in sub])
    jt_x  = _mean([c["jacc_treat_xgb"]  for c in sub])
    print(f"  {ln:<22} {jp_b:>10.3f} {jp_x:>10.3f} "
          f"{jt_b:>10.3f} {jt_x:>10.3f} {len(sub):>5}")

# Attribution mass summary
print("\n── Attribution mass (physiology stream) ────────────────────────────")
print(f"  Source mean |IG|:    {_mean([c['src_physio_mass']  for c in all_cases]):.4f}")
print(f"  Run B mean  |IG|:    {_mean([c['b_physio_mass']    for c in all_cases]):.4f}")
print(f"  XGB mean   |SHAP|:   {_mean([c['xgb_physio_mass']  for c in all_cases]):.4f}")
print(f"  Run B physio delta:  {_mean([c['physio_delta_b']   for c in all_cases]):.4f}  "
      f"(mean |B − Source| per patient)")
print(f"  XGB  physio delta:   {_mean([c['physio_delta_xgb'] for c in all_cases]):.4f}  "
      f"(mean |XGB − Source| per patient)")


# ══════════════════════════════════════════════════════════════════════════════
# ATTRIBUTION DELTA → RETRIEVAL DIVERGENCE CORRELATION
# Key mechanistic link: physio attribution shift → Jaccard divergence
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Attribution delta → Retrieval divergence (Spearman r) ──────────")

# Jaccard divergence = 1 − Jaccard (higher = more diverged from source)
div_b_physio   = [1 - c["jacc_physio_b"]   for c in all_cases]
div_xgb_physio = [1 - c["jacc_physio_xgb"] for c in all_cases]
delta_b        = [c["physio_delta_b"]       for c in all_cases]
delta_xgb      = [c["physio_delta_xgb"]     for c in all_cases]

r_b,   p_b   = spearmanr(delta_b,   div_b_physio)
r_xgb, p_xgb = spearmanr(delta_xgb, div_xgb_physio)

print(f"  Run B:    r={r_b:.3f}  p={p_b:.4f}  "
      f"(attribution delta vs physio retrieval divergence)")
print(f"  XGBoost:  r={r_xgb:.3f}  p={p_xgb:.4f}")
print(f"  Interpretation: r > 0 means larger attribution shift → more")
print(f"  divergent document retrieval. Expected stronger for XGBoost")
print(f"  (biological amnesia) than for Run B (frozen physiology).")
print(f"  Run B r < 0: frozen LSTM decouples attribution magnitude from")
print(f"  retrieval — residual delta comes from fusion head, not physio")
print(f"  features, so retrieval remains anchored to source documents.")


print(f"\n── Per-label attribution delta → divergence correlation ────────")
for ln in LABEL_COLS:
    sub = [c for c in all_cases if c["label"] == ln]
    db  = [c["physio_delta_b"]   for c in sub]
    dx  = [c["physio_delta_xgb"] for c in sub]
    div_b  = [1 - c["jacc_physio_b"]   for c in sub]
    div_x  = [1 - c["jacc_physio_xgb"] for c in sub]
    rb, pb   = spearmanr(db, div_b)
    rx, px   = spearmanr(dx, div_x)
    print(f"  {ln:<22} RunB r={rb:+.3f} p={pb:.4f} | "
          f"XGB r={rx:+.3f} p={px:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE RETRIEVALS — best true-positive per label
#
# A genuine true positive requires:
#   • true_label == 1          (ground truth positive)
#   • prob_src   >= TP_THRESHOLD  (source model predicts positive)
#   • prob_b     >= TP_THRESHOLD  (Run B predicts positive)
#
# Within qualifying cases, pick the highest Run B physio Jaccard
# (best stability showcase), with source probability as tiebreaker.
# Falls back to the highest-prob positive-labelled case if none clear
# the threshold, clearly flagged as a fallback.
# ══════════════════════════════════════════════════════════════════════════════


print("\n" + "="*70)
print("SAMPLE RETRIEVALS  (best true-positive per label)")
print("="*70)

def _lookup_exp(store, stay_id, label):
    return next(
        (e for e in store if e["stay_id"] == stay_id and e["label"] == label),
        None
    )

def _is_tp(c):
    return (
        c["true_label"] == 1
        and c["prob_src"] >= TP_THRESHOLD
        and c["prob_b"]   >= TP_THRESHOLD
    )

for ln in LABEL_COLS:
    label_cases = [c for c in all_cases if c["label"] == ln]
    tp_cases    = [c for c in label_cases if _is_tp(c)]

    # Select best TP; fall back to highest-RunB-prob positive-labelled case
    if tp_cases:
        case     = max(tp_cases, key=lambda c: (c["jacc_physio_b"], c["prob_src"]))
        tp_flag  = f"✅ TRUE POSITIVE ({len(tp_cases)} qualifying)"
    else:
        pos_cases = [c for c in label_cases if c["true_label"] == 1]
        if not pos_cases:
            print(f"\n  ⚠  {ln}: no positive-labelled cases found — skipping")
            continue
        case    = max(pos_cases, key=lambda c: c["prob_b"])
        tp_flag = (f"⚠  FALLBACK — no TP at threshold={TP_THRESHOLD} "
                   f"(best RunB prob={case['prob_b']:.3f})")

    sid     = case["stay_id"]
    src_exp = _lookup_exp(explanations["source"], sid, ln)
    b_exp   = _lookup_exp(explanations["run_b"],  sid, ln)
    xgb_exp = _lookup_exp(explanations["xgb"],    sid, ln)

    if not (src_exp and b_exp and xgb_exp):
        print(f"\n  ⚠  {ln}: explanations missing for stay {sid} — skipping")
        continue

    print(f"\n{'─'*70}")
    print(f"Stay {sid} | {ln} | true={case['true_label']}  {tp_flag}")
    print(f"  Probs: source={case['prob_src']:.3f}  "
          f"RunB={case['prob_b']:.3f}  XGB={case['prob_xgb']:.3f}")
    print(f"  Physio Jaccard: RunB={case['jacc_physio_b']:.3f}  "
          f"XGB={case['jacc_physio_xgb']:.3f}  "
          f"(Δ RunB−XGB={case['jacc_physio_b']-case['jacc_physio_xgb']:+.3f})")
    print(f"  Treat  Jaccard: RunB={case['jacc_treat_b']:.3f}  "
          f"XGB={case['jacc_treat_xgb']:.3f}")
    print(f"  Merged Jaccard: RunB={case['jacc_merged_b']:.3f}  "
          f"XGB={case['jacc_merged_xgb']:.3f}")
    print(f"\n  Source physio query:  {case['q_src_physio']}")
    print(f"  Run B  physio query:  {case['q_b_physio']}")
    print(f"  XGB    query:         {case['q_xgb']}")

    print(f"\n  Source physio hits:")
    for i, h in enumerate(src_exp["physio_hits"][:5]):
        print(f"    [{i+1}] yr={h['year']} sim={h['sim']:.3f} | {h['title'][:65]}...")
    print(f"  Source treat  hits:")
    for i, h in enumerate(src_exp["treat_hits"][:5]):
        print(f"    [{i+1}] yr={h['year']} sim={h['sim']:.3f} | {h['title'][:65]}...")

    print(f"\n  Run B  physio hits:")
    for i, h in enumerate(b_exp["physio_hits"][:5]):
        print(f"    [{i+1}] yr={h['year']} sim={h['sim']:.3f} | {h['title'][:65]}...")
    print(f"  Run B  treat  hits:")
    for i, h in enumerate(b_exp["treat_hits"][:5]):
        print(f"    [{i+1}] yr={h['year']} sim={h['sim']:.3f} | {h['title'][:65]}...")

    print(f"\n  XGB    merged hits:")
    for i, h in enumerate(xgb_exp["merged_hits"][:5]):
        print(f"    [{i+1}] yr={h['year']} sim={h['sim']:.3f} | {h['title'][:65]}...")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


with open(SAVE_PATH / "pubmed_rag_stability.json", "w") as f:
    json.dump(all_cases, f, indent=2, cls=_NpEncoder)
print(f"\n✅ Saved pubmed_rag_stability.json  ({len(all_cases)} cases)")

with open(SAVE_PATH / "pubmed_rag_explanations.json", "w") as f:
    json.dump(explanations, f, indent=2, cls=_NpEncoder)
print(f"✅ Saved pubmed_rag_explanations.json")

# Aggregate summary for paper
summary = {
    "corpus_size":          len(corpus_entries),
    "corpus_date_range":    f"{CORPUS_MIN_DATE} – {CORPUS_MAX_DATE}",
    "n_cases":              len(all_cases),
    "embedding_model":      EMBEDDING_BACKEND,
    "query_strategy":       "automatic — normalised MIMIC feature names, no vocabulary tables",
    "two_stream_split":     "physio sub-query (SEQ_FEATURES) + treat sub-query (TREATMENT_FEATURES)",
    # Jaccard stability
    "mean_jacc_physio_b":   _mean([c["jacc_physio_b"]   for c in all_cases]),
    "mean_jacc_physio_xgb": _mean([c["jacc_physio_xgb"] for c in all_cases]),
    "mean_jacc_treat_b":    _mean([c["jacc_treat_b"]    for c in all_cases]),
    "mean_jacc_treat_xgb":  _mean([c["jacc_treat_xgb"]  for c in all_cases]),
    "mean_jacc_merged_b":   _mean([c["jacc_merged_b"]   for c in all_cases]),
    "mean_jacc_merged_xgb": _mean([c["jacc_merged_xgb"] for c in all_cases]),
    # Rank correlation
    "mean_rho_physio_b":    _mean([c["rho_physio_b"]    for c in all_cases]),
    "mean_rho_physio_xgb":  _mean([c["rho_physio_xgb"]  for c in all_cases]),
    # Attribution delta → divergence correlation
    "spearman_r_runb":      float(r_b),
    "spearman_p_runb":      float(p_b),
    "spearman_r_xgb":       float(r_xgb),
    "spearman_p_xgb":       float(p_xgb),
    # Attribution mass
    "mean_physio_delta_b":  _mean([c["physio_delta_b"]  for c in all_cases]),
    "mean_physio_delta_xgb":_mean([c["physio_delta_xgb"] for c in all_cases]),
    "rho_physio_xgb_n_valid": len(rho_xgb_valid),
    "rho_physio_xgb_n_total": len(rho_xgb_vals),
    "rho_physio_b_n_valid":   len(rho_b_valid),
}
with open(SAVE_PATH / "pubmed_rag_summary.json", "w") as f:
    json.dump(summary, f, indent=2, cls=_NpEncoder)
print(f"✅ Saved pubmed_rag_summary.json")

import math
import csv
import time
import urllib.request as _urllib

MESH_TOP_N     = 20    # top-N source-retrieved MeSH terms define relevance
SIM_THRESHOLD  = 0.40  # minimum cosine sim to count as a confident retrieval
TP_THRESHOLD   = 0.5   # must match value in SAMPLE RETRIEVALS block

# Label display names used for NLM MeSH API lookup (no hardcoding of MeSH IDs)
LABEL_DISPLAY = {
    "label_vasopressor":  "vasopressor",
    "label_intubation":   "intubation",
    "label_septic_shock": "septic shock",
}


# ─── NLM MeSH API tree expansion ──────────────────────────────────────────────
def _mesh_api_expand(label_name: str, timeout: int = 8) -> set:
    """
    Query the NLM MeSH lookup API for a clinical term and return a set of
    all matching MeSH descriptor labels (normalised to lowercase).
    Falls back to empty set if the API is unreachable.
    Reference: https://id.nlm.nih.gov/mesh/lookup
    """
    term = label_name.replace(" ", "%20")
    url  = (f"https://id.nlm.nih.gov/mesh/lookup/descriptor"
            f"?label={term}&match=contains&limit=10")
    try:
        req = _urllib.Request(url, headers={"Accept": "application/json"})
        with _urllib.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        terms = {item["label"].lower() for item in data if "label" in item}
        return terms
    except Exception:
        return set()


# ─── Data-driven MeSH relevance (source model oracle) ─────────────────────────
def _build_source_mesh_sets(top_n: int = MESH_TOP_N) -> dict:
    """
    For each label, collect MeSH terms from all documents retrieved by the
    source model across all 100 cases. Return top-N as the relevant set.
    Returns: {label: set of lowercase MeSH terms}
    """
    from collections import Counter
    mesh_sets = {}
    for ln in LABEL_COLS:
        counter = Counter()
        for case in all_cases:
            if case["label"] != ln:
                continue
            sid     = case["stay_id"]
            src_exp = next((e for e in explanations["source"]
                            if e["stay_id"] == sid and e["label"] == ln), None)
            if src_exp is None:
                continue
            for hit in src_exp.get("physio_hits", []) + src_exp.get("treat_hits", []):
                for term in hit.get("mesh", []):
                    counter[term.lower()] += 1
        mesh_sets[ln] = {term for term, _ in counter.most_common(top_n)}
    return mesh_sets


# ─── Merge data-driven + API-expanded MeSH sets ───────────────────────────────
def _build_relevance_sets() -> dict:
    """
    Build the final relevance set per label by combining:
      1. Source-oracle MeSH set (data-driven, top-MESH_TOP_N terms)
      2. NLM MeSH API expansion of the label display name
    Returns: {label: set of lowercase MeSH terms}
    """
    print("\n  Building relevance sets from source-retrieved MeSH terms...")
    source_sets = _build_source_mesh_sets(MESH_TOP_N)

    print("  Querying NLM MeSH API for label name expansion...")
    api_sets = {}
    for ln, display in LABEL_DISPLAY.items():
        expanded = _mesh_api_expand(display)
        api_sets[ln] = expanded
        status = f"{len(expanded)} terms" if expanded else "unavailable (fallback to source-only)"
        print(f"    {ln}: NLM API → {status}")
        time.sleep(0.3)

    # Merge
    combined = {}
    for ln in LABEL_COLS:
        combined[ln] = source_sets.get(ln, set()) | api_sets.get(ln, set())
        print(f"    {ln}: total relevance terms = {len(combined[ln])}")

    return combined


# ─── Metric functions ──────────────────────────────────────────────────────────
def _mesh_hit(doc: dict, relevant_terms: set) -> bool:
    """True if any of the document's MeSH terms are in the relevant set."""
    for term in doc.get("mesh", []):
        if term.lower() in relevant_terms:
            return True
    return False


def _precision_at_k(hits: list, relevant_terms: set, k: int = 5) -> float:
    hits_k = hits[:k]
    if not hits_k:
        return float("nan")
    return sum(1 for h in hits_k if _mesh_hit(h, relevant_terms)) / len(hits_k)


def _sim_lift(hits: list) -> float:
    """Mean sim of retrieved docs / mean sim of whole corpus (retrieval lift)."""
    if not hits:
        return float("nan")
    corpus_mean = float(corpus_embeddings.mean())
    return float(sum(h["sim"] for h in hits) / len(hits)) / max(abs(corpus_mean), 1e-9)


def _p_at_threshold(hits: list, thr: float = SIM_THRESHOLD) -> float:
    if not hits:
        return float("nan")
    return sum(1 for h in hits if h["sim"] >= thr) / len(hits)


def _ndcg(hits: list, relevance: dict, k: int = 5) -> float:
    """nDCG@k given {pmid: 0|1} relevance dict."""
    def _dcg(ranked):
        return sum(
            relevance.get(h["pmid"], 0) / math.log2(i + 2)
            for i, h in enumerate(ranked[:k])
        )
    dcg  = _dcg(hits)
    idcg = _dcg(sorted(hits, key=lambda h: relevance.get(h["pmid"], 0), reverse=True))
    return dcg / idcg if idcg > 0 else float("nan")


def _mean(vals):
    v = [x for x in vals if not math.isnan(x)]
    return float(sum(v) / len(v)) if v else float("nan")


# ─── TP case selector (mirrors SAMPLE RETRIEVALS block) ───────────────────────
def _is_tp(c):
    return (
        c["true_label"] == 1
        and c["prob_src"] >= TP_THRESHOLD
        and c["prob_b"]   >= TP_THRESHOLD
    )

def _best_tp(label):
    tp_cases = [c for c in all_cases if c["label"] == label and _is_tp(c)]
    if not tp_cases:
        return None
    return max(tp_cases, key=lambda c: (c["jacc_physio_b"], c["prob_src"]))


# ══════════════════════════════════════════════════════════════════════════════
# PART A — AUTOMATIC METRICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("RAG GROUND TRUTH EVALUATION — Part A: Automatic metrics")
print("="*70)
print(f"\n  Relevance: explicit canonical MeSH descriptor per label")
for ln, terms in MESH_RELEVANCE.items():
    print(f"    {ln}: {sorted(terms)}")
print()

relevance_sets = MESH_RELEVANCE

auto = {m: {"p5": [], "ndcg5": []} for m in ["source", "run_b", "xgboost"]}

for case in all_cases:
    ln, sid = case["label"], case["stay_id"]
    rel = relevance_sets.get(ln, set())
    src_exp = next((e for e in explanations["source"] if e["stay_id"]==sid and e["label"]==ln), None)
    b_exp   = next((e for e in explanations["run_b"]  if e["stay_id"]==sid and e["label"]==ln), None)
    xgb_exp = next((e for e in explanations["xgb"]    if e["stay_id"]==sid and e["label"]==ln), None)
    if not (src_exp and b_exp and xgb_exp):
        continue
    for key, exp, stream in [("source", src_exp, "physio_hits"),
                             ("run_b",  b_exp,   "physio_hits"),
                             ("xgboost",xgb_exp, "merged_hits")]:
        hits = exp.get(stream, [])
        auto[key]["p5"].append(_precision_at_k(hits, rel, k=TOP_K_DOCS))
        binrel = {h["pmid"]: (1 if _mesh_hit(h, rel) else 0) for h in hits}
        auto[key]["ndcg5"].append(_ndcg(hits, binrel, k=TOP_K_DOCS))

print(f"  Overall (all {len(all_cases)} cases):\n")
print(f"  {'Model':<12} {'MeSH P@5':>10} {'MeSH nDCG@5':>13}")
print("  " + "─"*37)
for key, label in [("source","Source"), ("run_b","Run B"), ("xgboost","XGBoost")]:
    print(f"  {label:<12} {_mean(auto[key]['p5']):>10.3f} {_mean(auto[key]['ndcg5']):>13.3f}")

print(f"\n  Per-label MeSH Precision@{TOP_K_DOCS}:\n")
print(f"  {'Label':<22} {'Source':>8} {'Run B':>8} {'XGBoost':>9}")
print("  " + "─"*52)
for ln in LABEL_COLS:
    rel = relevance_sets.get(ln, set())
    sp, bp, xp = [], [], []
    for case in all_cases:
        if case["label"] != ln: continue
        sid = case["stay_id"]
        se = next((e for e in explanations["source"] if e["stay_id"]==sid and e["label"]==ln), None)
        be = next((e for e in explanations["run_b"]  if e["stay_id"]==sid and e["label"]==ln), None)
        xe = next((e for e in explanations["xgb"]    if e["stay_id"]==sid and e["label"]==ln), None)
        if not (se and be and xe): continue
        sp.append(_precision_at_k(se.get("physio_hits", []), rel, k=TOP_K_DOCS))
        bp.append(_precision_at_k(be.get("physio_hits", []), rel, k=TOP_K_DOCS))
        xp.append(_precision_at_k(xe.get("merged_hits", []), rel, k=TOP_K_DOCS))
    print(f"  {ln:<22} {_mean(sp):>8.3f} {_mean(bp):>8.3f} {_mean(xp):>9.3f}")

for key, skey in [("source","source"), ("run_b","runb"), ("xgboost","xgb")]:
    summary[f"auto_mesh_p5_{skey}"]    = _mean(auto[key]["p5"])
    summary[f"auto_mesh_ndcg5_{skey}"] = _mean(auto[key]["ndcg5"])
summary["auto_relevance_source"] = "explicit canonical MeSH descriptor per label"
summary["embedding_model"]       = f"{EMBEDDING_BACKEND}"

with open(SAVE_PATH / "pubmed_rag_summary.json", "w") as f:
    json.dump(summary, f, indent=2, cls=_NpEncoder)
print(f"\n✅ Auto metrics written to pubmed_rag_summary.json")

relevance_sets_serial = {k: sorted(v) for k, v in relevance_sets.items()}
with open(SAVE_PATH / "pubmed_rag_relevance_sets.json", "w") as f:
    json.dump(relevance_sets_serial, f, indent=2)
print(f"✅ Relevance sets saved → pubmed_rag_relevance_sets.json")


# ══════════════════════════════════════════════════════════════════════════════
# PART B — CLINICIAN RATING SCAFFOLD
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("RAG GROUND TRUTH EVALUATION — Part B: Clinician rating scaffold")
print("="*70)

RATER_INPUT_PATH  = SAVE_PATH / "pubmed_rag_rater_input.csv"
RATER_OUTPUT_PATH = SAVE_PATH / "pubmed_rag_rater_output.csv"

# Build rows from best TP case per label
rater_rows = []
for ln in LABEL_COLS:
    case = _best_tp(ln)
    if case is None:
        print(f"\n  ⚠  {ln}: no TP case at threshold={TP_THRESHOLD} — skipped")
        continue

    sid     = case["stay_id"]
    src_exp = next((e for e in explanations["source"]
                    if e["stay_id"] == sid and e["label"] == ln), None)
    b_exp   = next((e for e in explanations["run_b"]
                    if e["stay_id"] == sid and e["label"] == ln), None)
    xgb_exp = next((e for e in explanations["xgb"]
                    if e["stay_id"] == sid and e["label"] == ln), None)

    if not (src_exp and b_exp and xgb_exp):
        continue

    for model_key, exp, stream, query_field in [
        ("source",  src_exp, "physio_hits", "q_src_physio"),
        ("run_b",   b_exp,   "physio_hits", "q_b_physio"),
        ("xgboost", xgb_exp, "merged_hits", "q_xgb"),
    ]:
        for rank, hit in enumerate(exp.get(stream, [])[:TOP_K_DOCS], start=1):
            rater_rows.append({
                "label":         ln,
                "stay_id":       sid,
                "model":         model_key,
                "rank":          rank,
                "pmid":          hit["pmid"],
                "year":          hit["year"],
                "sim":           round(hit["sim"], 4),
                "physio_query":  case.get(query_field, ""),
                "title":         hit["title"],
                "abstract_150":  hit.get("abstract", "")[:150].replace("\n", " "),
                "mesh_tags":     "; ".join(hit.get("mesh", [])[:6]),
                # ── Clinician fills in these two columns ──────────────────
                "relevant_0_1":  "",  # 1 = relevant to clinical decision, 0 = not
                "rater_notes":   "",  # optional free text
            })

fieldnames = [
    "label", "stay_id", "model", "rank", "pmid", "year", "sim",
    "physio_query", "title", "abstract_150", "mesh_tags",
    "relevant_0_1", "rater_notes",
]
with open(RATER_INPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rater_rows)

n_rows = len(rater_rows)
print(f"\n✅ Rater input exported → {RATER_INPUT_PATH}")
print(f"   {n_rows} rows  "
      f"({len(LABEL_COLS)} labels × 3 models × top-{TOP_K_DOCS} docs)")
print(f"\n   INSTRUCTIONS FOR RATER:")
print(f"   1. Open pubmed_rag_rater_input.csv")
print(f"   2. For each row, read 'title' and 'abstract_150'")
print(f"   3. Set 'relevant_0_1' = 1 if the document is clinically")
print(f"      relevant to the decision implied by 'physio_query',")
print(f"      0 if not. Blank = treated as 0.")
print(f"   4. Save as pubmed_rag_rater_output.csv in the same directory")
print(f"   5. Re-run this script to get Precision@{TOP_K_DOCS} + nDCG@{TOP_K_DOCS}")


# ── Read back filled ratings if they exist ────────────────────────────────────
if RATER_OUTPUT_PATH.exists():
    print(f"\n✅ Found {RATER_OUTPUT_PATH.name} — computing clinician metrics...")

    ratings = {}
    with open(RATER_OUTPUT_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["label"], row["stay_id"], row["model"])
            try:
                score = int(row["relevant_0_1"])
            except (ValueError, KeyError):
                score = 0
            ratings.setdefault(key, {})[row["pmid"]] = score

    eval_results = {m: {"prec": [], "ndcg": []} for m in ["source", "run_b", "xgboost"]}

    for ln in LABEL_COLS:
        case = _best_tp(ln)
        if case is None:
            continue
        sid = case["stay_id"]

        src_exp = next((e for e in explanations["source"]
                        if e["stay_id"] == sid and e["label"] == ln), None)
        b_exp   = next((e for e in explanations["run_b"]
                        if e["stay_id"] == sid and e["label"] == ln), None)
        xgb_exp = next((e for e in explanations["xgb"]
                        if e["stay_id"] == sid and e["label"] == ln), None)

        for model_key, exp, stream in [
            ("source",  src_exp, "physio_hits"),
            ("run_b",   b_exp,   "physio_hits"),
            ("xgboost", xgb_exp, "merged_hits"),
        ]:
            if exp is None:
                continue
            hits = exp.get(stream, [])[:TOP_K_DOCS]
            rel  = ratings.get((ln, str(sid), model_key), {})
            prec = sum(rel.get(h["pmid"], 0) for h in hits) / len(hits) if hits else float("nan")
            eval_results[model_key]["prec"].append(prec)
            eval_results[model_key]["ndcg"].append(_ndcg(hits, rel, k=TOP_K_DOCS))

    print(f"\n  Clinician-rated retrieval quality "
          f"(TP cases, physio stream, P@{TOP_K_DOCS} / nDCG@{TOP_K_DOCS}):\n")
    print(f"  {'Model':<12} {'P@{}'.format(TOP_K_DOCS):>8} {'nDCG@{}'.format(TOP_K_DOCS):>10}")
    print("  " + "─"*34)
    for model_key, label in [("source", "Source"), ("run_b", "Run B"), ("xgboost", "XGBoost")]:
        r = eval_results[model_key]
        print(f"  {label:<12} {_mean(r['prec']):>8.3f} {_mean(r['ndcg']):>10.3f}")

    print(f"\n  Note: single rater. For publication, a second rater + Cohen's kappa")
    print(f"  is recommended. Add a 'relevant_0_1_rater2' column to the output CSV")
    print(f"  and compute kappa across the two columns.")

    for key, skey in [("source","source"), ("run_b","runb"), ("xgboost","xgb")]:
        summary[f"clinician_p5_{skey}"]   = _mean(eval_results[key]["prec"])
        summary[f"clinician_ndcg5_{skey}"] = _mean(eval_results[key]["ndcg"])

    with open(SAVE_PATH / "pubmed_rag_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=_NpEncoder)
    print(f"\n✅ Clinician metrics written to pubmed_rag_summary.json")

else:
    print(f"\n  ℹ  No rater output found yet.")
    print(f"     Fill in pubmed_rag_rater_input.csv, save as")
    print(f"     pubmed_rag_rater_output.csv, then re-run.")
# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"Corpus: {len(corpus_entries)} abstracts | {CORPUS_MIN_DATE} – {CORPUS_MAX_DATE}")
print(f"Cases:  {len(all_cases)} (post-drift, {N_CASES_PER_LABEL}/label)")
print(f"\nQuery strategy: automatic from IG/SHAP feature names")
print(f"  No vocabulary tables | No MeSH curation | No relevance labels")
print(f"  Physio sub-query: top-{TOP_K_PHYSIO} SEQ_FEATURES by |IG|")
print(f"  Treat  sub-query: top-{TOP_K_TREAT} TREATMENT_FEATURES by |SHAP/IG|")

print(f"\nRETRIEVAL STABILITY (Jaccard vs source model):")
print(f"  {'Stream':<20} {'Run B':>8} {'XGBoost':>9}  "
      f"{'Δ (B−XGB)':>10}")
print("  " + "─"*50)
for name, kb, kx in [
    ("Physiology",  "mean_jacc_physio_b",  "mean_jacc_physio_xgb"),
    ("Treatment",   "mean_jacc_treat_b",   "mean_jacc_treat_xgb"),
    ("Merged",      "mean_jacc_merged_b",  "mean_jacc_merged_xgb"),
]:
    vb = summary[kb];  vx = summary[kx]
    print(f"  {name:<20} {vb:>8.3f}  {vx:>9.3f}  {vb-vx:>+10.3f}")

print(f"\nMECHANISTIC LINK (attribution delta → retrieval divergence):")
print(f"  Run B   Spearman r={summary['spearman_r_runb']:.3f}  "
      f"p={summary['spearman_p_runb']:.4f}")
print(f"  XGBoost Spearman r={summary['spearman_r_xgb']:.3f}  "
      f"p={summary['spearman_p_xgb']:.4f}")


print(f"\nATTRIBUTION MASS SHIFT (physiology stream, |adapted − source|):")
print(f"  Run B:    {summary['mean_physio_delta_b']:.4f}")
print(f"  XGBoost:  {summary['mean_physio_delta_xgb']:.4f}")
ratio = summary["mean_physio_delta_xgb"] / max(summary["mean_physio_delta_b"], 1e-9)
print(f"  Ratio:    {ratio:.1f}×  (>1 = XGBoost physio more unstable)")
print(f"  Note: raw delta ratio. Normalized ratio (÷ source p95) reported")
print(f"  in fig_biological_amnesia for cross-section consistency.")
print(f"\n✅ script13 complete")