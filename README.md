# Biological Amnesia in ICU Time-Series Prediction: A Drift-Adaptive Two-Stream Architecture with Temporal Retrieval

## Overview
This repository contains the complete codebase for an adaptive clinical intelligence architecture designed to mitigate "biological amnesia"—the silent overwriting of stable physiological representations as a model adapts to shifting treatment protocols—and provide a safe, fully governable blueprint for Clinical Decision Support Systems (CDSS) deployed in non-stationary Intensive Care Units. 

The framework integrates four primary components:
1. **Ongoing-need label formulation** — Replaces initiation-only labels to eliminate anti-correlation leakage and accurately capture the clinical need for interventions.
2. **Two-stream neural architecture** — Structurally decouples stable physiological dynamics (LSTM) from evolving treatment protocols (MLP).
3. **Dual-signal drift detection & selective adaptation with Explanation** — Uses a composite distributional and accuracy trigger to update *only* the treatment stream, freezing physiological representations.
4. **Attribution-driven Temporal RAG** — Uses per-instance Integrated Gradients (IG) to build patient-specific PubMed queries, conditioning retrieved evidence on the detected drift era.

Evaluated on **84,792 MIMIC-IV v3.1 ICU stays (2008–2022)** using strict chronological splitting and subject-level decontamination.

--------------------------------------------------------------------------------

## Requirements & Data Access
* **Python 3.9+** (Developed on Kaggle using a P100 GPU).
* **MIMIC-IV v3.1 Access:** This study uses MIMIC-IV, which is freely available via PhysioNet upon credentialing. 
* **Data Privacy:** Due to the strict PhysioNet Data Use Agreement (DUA), we **cannot share** the preprocessed patient data splits, prediction arrays, or trained model weights. All results are fully reproducible from the raw MIMIC-IV files using the provided pipeline. Aggregate population-level logs and metrics are provided in the `results/` folder.

--------------------------------------------------------------------------------

## Execution Order
The pipeline is strictly sequential. Scripts must be run in the following order:

### 1. Data Generation
* **`preprocessing/script1_preprocessing.py`**
  * **Input:** Raw MIMIC-IV CSVs (`hosp/`, `icu/`)
  * **Output:** `train/val/test_final_enriched.parquet`, `mimiciv_demographics.parquet`, `feature_meta.json`
  * **Note:** Applies the "Ongoing Need" label formulation.

### 2. Model Training & Drift Adaptation
* **`models/script2_two_stream_model.py`**
  * **Action:** Trains Run A (static), Run B (selective adaptation), Run C (full adaptation). Detects 2020-2022 drift. 
  * **Output:** `two_stream_models.pt`, `eval_split.json` (subject-level locks), `clinical_drift_audit.txt`.
  * **`utils/drift_explainability.py`**: The automated governance module. It is called directly during `script2`'s adaptation phase to perform permutation-based feature ablation. It generates the human-readable `clinical_drift_audit.txt` log.
* **`models/script3_run_d_single_stream.py`**
  * **Action:** Trains Run D (monolithic single-stream LSTM baseline with late-fusion).

### 3. Evaluation & Baseline Comparisons
* **`evaluation/script5_xgboost_bootstrap_shap.py`**
  * **Action:** Trains XGBoost baselines (Source and Adapted), calculates Bootstrap CIs, and extracts population-level SHAP values. 
  * **Output:** `bootstrap_ci_results.json`, `full_comparison_results.json`, `xgb_predictions_corrected.npz`.
* **`evaluation/script6_disagreement_matrix.py`**
  * **Action:** Generates the bedside disagreement analysis comparing all models.
  * **Output:** `disagreement_matrix_results_with_runD.json`, and disagreement matrix `.png` figures.
* **`evaluation/script7_absolute_counts.py`**
  * **Action:** Computes absolute TP/FP/TN/FN counts across models at standard thresholds.
  * **Output:** `absolute_counts_per_model.json`.

### 4. Explainability & Retrieval (RAG)
* **`evaluation/script_mc4_delta_attribution.py`**
  * **Action:** Computes population-level ∆-Attribution metrics to formally quantify biological amnesia.
  * **Output:** `delta_attribution_table.json`, `delta_attribution_summary.txt`.
* **`evaluation/fig.py`**
  * **Action:** Generates the 5-panel Biological Amnesia figure illustrating stable IG attributions (Run B) vs. shifting SHAP attributions (XGBoost).
  * **Output:** `fig_biological_amnesia_v3.png`.
* **`rag/script13_rag_pubmed_final.py`**
  * **Action:** Executes the Attribution-Driven Temporal RAG pipeline using MedCPT. 
  * **Output:** `pubmed_rag_summary.json` and outputs for the clinician-rated scaffold.

--------------------------------------------------------------------------------

## Key Results

### 1. Bedside Clinical Safety and Calibration
While monolithic retraining (XGBoost) achieved high aggregate AUROC, it exhibited probability mass compression on the rarest condition evaluated (septic shock). Our Two-Stream framework (Run B) mitigated this compression and safely triggered true-positive alerts:

* **Disagreement Asymmetry:** In a threshold-based disagreement analysis (Catch ≥ 0.50, Miss < 0.10), Run B identified **26 true-positive septic shock cases** that the XGBoost-adapted model missed. Zero cases were missed in the reverse direction.
* **Precision-Recall (AUPRC):** After adaptation, the monolithic XGBoost baseline's septic shock AUPRC dropped from 0.3313 to 0.2600. Our selective adaptation framework improved septic shock AUPRC to **0.4131**.
* **Calibration:** Run B improved the Septic Shock Brier score to **0.0184**, whereas the static baseline yielded 0.0613.

### 2. Mitigating Biological Amnesia
* **Architectural Guarantee:** Monolithic retraining unintentionally altered stable physiological representations while adapting to new protocols. By structurally decoupling the streams, the Two-Stream framework mathematically preserves the physiological representations (physio mean_rel_Δ = 0.0000).
* **Population-Level $\Delta$-Attribution:** Population-level analysis showed that full retraining significantly distorted 93.6% of feature attributions. Run B confined all adaptations exclusively to the treatment and fusion streams, leaving physiological weights bitwise identical to the source model.

### 3. Attribution-Driven Temporal RAG
Because Run B structurally locks the physiological representations, it maintains consistent medical evidence retrieval even as clinical protocols drift.
* **Retrieval Stability:** Run B maintained a Physiology Jaccard overlap of **0.573** with the pre-drift era documents, compared to 0.330 for the retrained baseline. 
* **Retrieval Quality:** Run B achieved an automatic **MeSH P@5 of 0.635** and a **Clinician-rated P@5 of 0.800**, ensuring that retrieved PubMed literature remains anchored to the patient's actual biology.

--------------------------------------------------------------------------------

## Automated Governance Audit Logs
Unlike monolithic retraining, our Two-Stream framework generates human-readable, causal audit logs at each adaptation event. This ensures model updates remain transparent, interpretable, and contestable by clinicians. 

For example, when detecting the severe protocol shifts in the 2020–2022 COVID-19 cohort, the system automatically generated the following explanation (`clinical_drift_audit.txt`) before adapting its weights:

> **🚨 CLINICAL PRACTICE SHIFT DETECTED**
> Between the training period (2014-2016) and the new data (2020-2022), patient treatment patterns have changed significantly.
> 
> **Key clinical changes driving this update:**
> * **Total Crystalloid Ml:** Average value decreased from 791.01 to 254.97 (Severe Drift PSI: 0.76)
> * **Early Antibiotic:** Usage increased by 14.3% in the newer data.
> * **Insulin Infusion:** Usage dropped by 8.0% in the newer data.
> * **Blood Products:** Usage dropped by 8.0% in the newer data.
> * **Early Steroid:** Usage increased by 6.5% in the newer data.
> 
> *FINAL MODEL UPDATE AUDIT LOG:*
> *To maintain accuracy safely, the model locked its physiological representation and exclusively updated its treatment and fusion layers. This restores accuracy by forcing the network to relearn the specific features that carry high predictive importance but suffered from severe real-world drift (most notably Total Crystalloid Ml and Early Antibiotic).*

--------------------------------------------------------------------------------

## RAG Evaluation Data
If a local corpus is not found, the script `rag/script13_rag_pubmed_final.py` will automatically ping the NCBI API to reconstruct the era-matched 522-abstract PubMed corpus. 

For transparency, we have included `pubmed_rag_rater_output.csv` in the repository. This file contains the specific abstracts retrieved during our Temporal RAG evaluation and the human-annotated relevance scores used to calculate the Clinician-rated P@5 metric. Because this file contains only public medical abstracts and binary ratings, it contains no patient data and is freely shareable.

--------------------------------------------------------------------------------

## License
MIT License. See `LICENSE` for details.

--------------------------------------------------------------------------------

## Contact
**Fatema Ferdous Tamanna** (Corresponding Author)  
Dept. of Computer Science and Information Technology  
Patuakhali Science and Technology University, Bangladesh  
fatimatamannaah@gmail.com  
ORCID: 0009-0002-2101-4391
