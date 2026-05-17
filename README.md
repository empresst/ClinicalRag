# A Drift-Adaptive Framework for Clinical Time-Series: Two-Stream Architectures with Attribution-Driven Semantic Retrieval

**Fatema Ferdous Tamanna, K. M. Merajul Arefin, Md. Abdul Masud**

*Manuscript prepared for Artificial Intelligence in Medicine (Elsevier), 2026*

---

## Overview

This repository contains all code for the paper. The framework integrates four components:

1. **Ongoing-need label formulation** — eliminates anti-correlation leakage in ICU treatment prediction
2. **Two-stream neural architecture** — structurally decouples stable physiological dynamics (LSTM) from evolving treatment patterns (MLP)
3. **PSI-triggered selective adaptation** — updates only the treatment stream upon distributional shift
4. **Attribution-driven Temporal RAG** — uses Integrated Gradients to construct era-filtered PubMed queries

Evaluated on MIMIC-IV v3.1 (40,155 ICU stays, 2014–2022) with strict chronological splitting and subject-level decontamination.

---

## Requirements

Python 3.9+

```bash
pip install -r requirements.txt
```

Developed on Kaggle (GPU P100). Expected runtime end-to-end: ~4–6 hours on GPU. CPU runtime significantly longer.

---

## Data Access

This study uses **MIMIC-IV v3.1**, freely available via PhysioNet upon credentialing:
https://physionet.org/content/mimiciv/3.1/
Preprocessed patient data and model weights **cannot be shared** due to the PhysioNet Data Use Agreement. All results are reproducible from raw MIMIC-IV v3.1 using the provided scripts.

Set your paths at the top of each script:

```python
DATA_PATH = Path("your/mimic-iv/path")   # raw MIMIC-IV
BASE_PATH = Path("your/processed/path")  # parquet files from script1
SAVE_PATH = Path("your/outputs/path")    # all outputs
```

---

## Repository Structure
```text
drift-adaptive-clinical-rag/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   └── README.md                          # MIMIC-IV access instructions
│
├── preprocessing/
│   └── script1_preprocessing.py           # cohort extraction, feature engineering,
│                                          # label formulation, forward purging
│
├── models/
│   ├── architecures.py                    # Single stream, double stream model architecture
│   ├── script2_two_stream_model.py        # Run A (static) + Run B (selective adapt)
│   │                                      # + Run C (full adapt)
│   └── script3_run_d_single_stream.py     # Run D (monolithic single-stream baseline)
│
├── evaluation/
│   ├── script4_model_comparison.py        # 4-model AUROC/AUPRC comparison table
│   ├── script5_xgboost_bootstrap_shap.py  # XGBoost training + bootstrap BCa CIs
│                                          # + SHAP delta attribution
│   ├── script6_disagreement_matrix.py     # bidirectional disagreement matrix
│                                          # all 6 models, worked examples
│   ├── script7_psi_sensitivity.py         # PSI threshold sensitivity sweep
│   ├── script8_ablation_studies.py        # split ratio + replay buffer ablation
│   └── script9_shap_analysis.py           # standalone SHAP CI + delta analysis
│
├── rag/
│   ├── script10_rag_synthetic.py          # expanded synthetic corpus (proof of concept)
│   ├── script11_rag_pubmed_easy.py        # PubMed easy corpus (815 abstracts, 1:1 ratio)
│   ├── script12_rag_pubmed_hard.py        # PubMed hard corpus (917 abstracts, 1:4.9
│                                          # ratio) — PRIMARY REPORTED RESULTS
│   └── script13_rag_pubmed_mesh.py        # MeSH corpus (749 abstracts) — secondary
│
├── corpus/
│   └── pubmed_corpus_hard.json            # 917 PubMed abstracts used in RAG evaluation
│                                          # (155 relevant / 274 hard-negative / 488 noise)
│                                          # No MIMIC data — freely shareable
│
├── supplements/
│   ├── script14_decontamination_verification.py  # subject leakage verification table
│   ├── script15_septic_shock_24_cases.py         # 24 caught septic shock cases analysis
│   └── script16_table2_demographics.py           # Table 2 patient characteristics
│
└──   utils/
    ├── __init__.py           # Marks directory as module
    ├── constants.py          # Global features and configurations
    ├── data_utils.py         # Loading, normalizing, and datasets
    ├── train_utils.py        # Custom loss and weighting
    ├── metrics_utils.py      # Evaluation and bootstrap math
    ├── xai_utils.py          # SHAP explainability toolset
    └── rag_utils.py          # Document retrieval and generation  

```
---

## Execution Order

### Step 1 — Preprocessing

```bash
python preprocessing/script1_preprocessing.py
```

**Input:** Raw MIMIC-IV CSVs (hosp/, icu/ directories)  
**Output:**
- `train_final_enriched.parquet`
- `val_final_enriched.parquet`
- `test_final_enriched.parquet`
- `mimiciv_demographics.parquet`
- `feature_meta.json`

---

### Step 2 — Two-Stream Model (Run A, B, C)

```bash
python models/script2_two_stream_model.py
```

**Input:** Parquet files from Step 1  
**Output:**
- `two_stream_models.pt` — source weights (Run A) + adapted weights (Run B)
- `temp_run_c_weights.pt` — full adaptation ablation (Run C)
- `eval_post_stays.json` — locked held-out evaluation set (5,761 stays)

---

### Step 3 — Single-Stream Baseline (Run D)

```bash
python models/script3_run_d_single_stream.py
```

**Input:** Parquet files + `two_stream_models.pt`  
**Output:**
- `full_adapt_models.pt` — Run C and Run D weights

---

### Step 4 — Model Comparison Table

```bash
python evaluation/script4_model_comparison.py
```

**Input:** Parquet files + both `.pt` files + `eval_post_stays.json`  
**Output:** Printed 4-model AUROC/AUPRC comparison (Table 3 and Table 4)

---

### Step 5 — XGBoost Baseline + Bootstrap CIs + SHAP

```bash
python evaluation/script5_xgboost_bootstrap_shap.py
```

**Input:** Parquet files + `two_stream_models.pt` + `eval_post_stays.json`  
**Output:**
- `bootstrap_ci_results.json` — BCa-corrected 95% CIs (Table 5)
- `full_comparison_results.json` — all model metrics
- `xgb_predictions.npz` — raw XGBoost predictions
- `xgb_source_{label}.pkl` — saved XGBoost source models
- `xgb_adapted_{label}.pkl` — saved XGBoost adapted models

---

### Step 6 — Disagreement Matrix

```bash
python evaluation/script6_disagreement_matrix.py
```

**Input:** Both `.pt` files + `xgb_predictions_corrected.npz` + `eval_post_stays.json`  
**Output:**
- `post_drift_predictions_with_runD.npz` — aligned prediction arrays all 6 models
- `disagreement_matrix_results_with_runD.json` — Table 6 numbers
- `disagreement_matrices_abc.png`
- `disagreement_matrices_runD.png`

> **Note:** This script requires `xgb_predictions_corrected.npz` which is generated
> by running the correction cell at the end of script5. See script5 comments for details.
> The correction ensures XGBoost and PyTorch prediction arrays share identical
> patient ordering before comparison.

---

### Step 7 — PSI Threshold Sensitivity

```bash
python evaluation/script7_psi_sensitivity.py
```

**Input:** Parquet files + `two_stream_models.pt` + `eval_post_stays.json`  
**Output:** `psi_sensitivity.png` + printed results (Section 3.2)

> **Note:** Uses quantile-based PSI binning for continuous features and
> two-bin PSI for binary features. Because observed maximum PSI (1.035)
> substantially exceeds all tested thresholds (0.10–0.40), adaptation
> triggers uniformly across the sweep — demonstrating robustness of the
> 0.20 threshold rather than threshold sensitivity in the traditional sense.
> This is acknowledged in the paper (Section 3.2).

---

### Step 8 — Ablation Studies

```bash
python evaluation/script8_ablation_studies.py
```

**Input:** Parquet files + `two_stream_models.pt`  
**Output:** Printed ablation results (Table 9 and Table 10)

---

### Step 9 — SHAP Analysis (Standalone)

```bash
python evaluation/script9_shap_analysis.py
```

**Input:** Parquet files + `two_stream_models.pt` + XGBoost `.pkl` files from Step 5  
**Output:**
- `shap_ci_results.json` — per-feature SHAP importance with 95% CIs
- `shap_delta_results.json` — ΔΦ delta-attribution between adapted and source

> **Note:** SHAP analysis is also embedded within script5 as part of the full
> pipeline. This standalone script enables independent reproduction without
> rerunning the full XGBoost training pipeline.

---

### Steps 10–13 — RAG Evaluation

RAG scripts require a free NCBI API key:
https://www.ncbi.nlm.nih.gov/account/settings/
Set as environment variable:

```bash
export NCBI_API_KEY="your_key_here"
```

```bash
python rag/script10_rag_synthetic.py    # synthetic corpus — proof of concept
python rag/script11_rag_pubmed_easy.py  # easy PubMed corpus
python rag/script12_rag_pubmed_hard.py  # hard corpus — PRIMARY REPORTED RESULTS
python rag/script13_rag_pubmed_mesh.py  # MeSH corpus — secondary corroboration
```

**Input:** Parquet files + `two_stream_models.pt` + `eval_post_stays.json`  
**Output:** RAG retrieval metrics (Table 8) + retrieved evidence per patient

> The 917-abstract corpus used in the primary evaluation is pre-built and
> available at `corpus/pubmed_corpus_hard.json`. RAG retrieval results
> can be verified against this corpus without MIMIC-IV access.

---

### Supplementary Scripts

```bash
python supplements/script14_decontamination_verification.py
```

**Output:** Subject leakage verification across all 6 partition pairs + supplementary decontamination table

```bash
python supplements/script15_septic_shock_24_cases.py
```

**Output:** Clinical feature summary and SHAP analysis of 24 septic shock cases caught by Run B and missed by XGBoost-adapted

```bash
python supplements/script16_table2_demographics.py
```

**Output:** Table 2 patient characteristics across temporal cohorts

---

## RAG Corpus

The 917-abstract PubMed corpus used in the primary RAG evaluation is included at:

This file contains no MIMIC-IV patient data and is freely shareable. It includes:
- 155 target-relevant abstracts
- 274 semantically proximate hard negatives
- 488 general noise abstracts
- Temporal labels (pre-COVID / post-COVID) based on publication date

Anyone can verify RAG retrieval results against this corpus **without MIMIC-IV access**.

---

## Key Results

| Model | Vasopressor AUROC | Intubation AUROC | Septic Shock AUROC | mAUROC |
|---|---|---|---|---|
| XGBoost (Source) | 0.9497 | 0.9565 | 0.9584 | 0.9549 |
| XGBoost (Adapted) | 0.9728 | 0.9553 | 0.9687 | 0.9656 |
| Run A (Static) | 0.9460 | 0.9495 | 0.9596 | 0.9517 |
| Run B (Proposed) | **0.9601** | **0.9526** | **0.9672** | **0.9600** |
| Run C (Full Adapt) | 0.9587 | 0.9536 | 0.9600 | 0.9574 |
| Run D (Single-Stream) | 0.9532 | 0.9520 | 0.9573 | 0.9541 |

**Key clinical finding:** Run B caught 24 true-positive septic shock cases missed by XGBoost-adapted (zero reverse misses), while halving vasopressor calibration error (Brier 0.0852 → 0.0419).

**RAG performance (PubMed Hard corpus):** Post-drift Precision@3 = 0.956, HardNeg@5 reduced from 0.240 (pre-drift) to 0.053 (post-drift).

---

## Important Implementation Notes

**XGBoost–PyTorch ordering alignment**

The disagreement matrix (script6) compares XGBoost and PyTorch model predictions at the patient level. Both models sort patients by `stay_id` ascending internally, but the saved `xgb_predictions.npz` file may not match this ordering. Script6 includes an explicit ordering verification and realignment step. A corrected predictions file (`xgb_predictions_corrected.npz`) is generated automatically.

**PSI binning**

Script7 uses quantile-based binning for continuous treatment features and two-bin PSI for binary features (e.g., `has_norepinephrine_obs`, `gender_M`). Linear binning is avoided due to skewed distributions in features like `total_crystalloid_ml`.

**Repeated code across scripts**

`ICUDataset`, `TwoStreamModel`, `load_split()`, and `normalize()` are intentionally duplicated across scripts for standalone reproducibility. Each script can be run independently without importing from other scripts.

---

## Citation

```bibtex
@article{tamanna2026drift,
  title   = {A Drift-Adaptive Framework for Clinical Time-Series: 
             Two-Stream Architectures with Attribution-Driven Semantic Retrieval},
  author  = {Tamanna, Fatema Ferdous and Arefin, K. M. Merajul and Masud, Md. Abdul},
  journal = {Artificial Intelligence in Medicine},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

MIT License. See `LICENSE` for details.

---

## Contact

**Fatema Ferdous Tamanna** (Corresponding Author)  
Dept. of Computer Science and Engineering  
Patuakhali Science and Technology University, Bangladesh  
fatimatamannaah@gmail.com  
ORCID: 0009-0002-2101-4391
