# Bounded model updates and explanation stability — code

Code for the manuscript "Where does explanation movement go when part of a clinical model is frozen? A bounded-update case study in intensive care using MIMIC-IV".
It trains a two-stream ICU model (LSTM physiology encoder, MLP treatment stream,
fusion head) on MIMIC-IV 2008–2013, detects drift at 2020–2022, adapts it two ways
(Run B: encoder excluded from the optimiser; Run C: every layer updates), and
compares Integrated Gradients attributions and attribution-conditioned PubMed
retrieval against the source model.

## Data

MIMIC-IV v3.1 requires credentialed PhysioNet access and is not redistributed here.
No patient-level data or stay-level outputs are included in this repository; the
files in `results/` are aggregate results only.

## Setup

```
pip install -r requirements.txt
export MIMIC_DIR=/path/to/mimic-iv-3.1     # raw MIMIC-IV (hosp/, icu/)
export DATA_DIR=/path/to/enriched_parquets  # output of step 1
export OUT_DIR=/path/to/working_dir         # models and results
```

Run every script from the repository root so that `utils/` and `models/` import.
Steps 1–3 were run on Kaggle (GPU); the rest ran locally on CPU.

## Pipeline

| Step | Script | Produces | Paper |
|---|---|---|---|
| 1 | `script1_preprocessing.py` | `{train,val,test}_final_enriched4.parquet` | Cohort, labels, features |
| 2 | `script2_two_stream_model.py` | `two_stream_models.pt` (source + Run B), `temp_run_c_weights.pt`, `eval_split.json`, `post_drift_predictions.npz`, console log | Source training, drift trigger, Runs A–C, AUROC, B−C bootstrap, weight change, update audit |
| 3 | `script3_run_d_single_stream.py` | `run_d_model.pt`, `run_d_results.json`, `post_drift_predictions_d.npz` | Run D (supplementary material) |
| 4 | `script13Rag_BvsC.py` | `pubmed_corpus.json`, `pubmed_rag_explanations.json`, `pubmed_rag_stability.json`, `pubmed_rag_summary.json` | Integrated Gradients, MedCPT retrieval, Jaccard, attribution-mass shift, MeSH P@5 |
| 5 | `script14_rag_significance_rbo.py` | `pubmed_rag_significance.json` | RBO, paired randomisation tests, bootstrap CIs, stream contrast, per-label results |
| 6 | `script15_bca_and_brier.py` | console | BCa AUROC intervals from `post_drift_predictions*.npz` |
| 7 | `script18_attribution_similarity.py` | `attribution_similarity.json` | Table 1 attribution rows, sign flips, encoder bitwise-identity check |
| 8 | `script19_regime_separability.py` | `regime_separability.json` | AUC 0.757 / 82.7% separability, treatment-stream attribution similarity, contrasts without septic shock |
| 9 | `script20_multiseed_adaptation.py` | `multiseed_adapted_weights.pt`, `multiseed_adaptation.json` | Runs B and C re-adapted under all five seeds (rebuilt adaptation partitions, evaluation population fixed) |
| 10 | `script22_relocation_check.py` | `relocation_check.json` | Five-seed replication of the attribution contrasts and weight change (supplementary material, Table S4) |
| 11 | `script23_faithfulness.py` | `faithfulness.json` | Attribution deletion test (supplementary material, Table S5) |
| 12 | `script8_ablation_studies.py` | `ablation_results.json` | Replay-buffer and split-ratio ablation (supplementary material; log in `provenance/ablation_output.txt`) |
| 13 | `script27_cohort_table.py` | console | Cohort characteristics by era (supplementary material, Table S3) |

Label checks (independent, need raw MIMIC-IV):

- `check_vasopressor_itemset.py` — share of vasopressor positives affected by the mannitol code and the missing vasopressin

Shared modules: `utils/constants.py`, `utils/data_utils.py`, `utils/train_utils.py`,
`utils/drift_explainability.py`, `models/architectures.py`.

## Notes

- Seeds: source models use seeds 42, 123, 7, 2024, 99 and the median by validation
  AUROC is the source. Each adaptation arm is run under the same five seeds and the
  seed with the median mean AUROC on the post-drift evaluation partition is kept
  (Run B: 2024, Run C: 123).
- Bootstrap intervals in `script2` are percentile intervals (1,000 resamples).
- `provenance/` holds the original logs from the Kaggle run (n = 5,749): the Run B vs
  Run A BCa interval (`script5_xgboost_bootstrap_shap.py`, `script5_output.txt`), the
  Run D log and the replay-buffer ablation log. `script15` recomputes the BCa intervals
  from `post_drift_predictions.npz`.

## Licence

See `LICENSE`.
