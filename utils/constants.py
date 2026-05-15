#%%writefile utils/constants.py

# Paste your exact lists here without changing a single word
SEQ_FEATURES = [
    "heart_rate","sbp_noninvasive","dbp_noninvasive","sbp_invasive","dbp_invasive",
    "map_invasive","temperature_c","spo2","resp_rate",
    "creatinine","wbc","platelets","lactate","bun","bilirubin_total","glucose",
    "hematocrit","potassium","sodium","troponin_t","ph_venous","pco2_venous",
    "base_excess","rbc","chloride","calcium",
    "urine_output","urine_output_ml_kg_hr","weight",
    "heart_rate_time_delta","map_invasive_time_delta","lactate_time_delta",
    "creatinine_baseline","creatinine_delta","creatinine_ratio",
    "lactate_baseline","lactate_delta","lactate_ratio",
    "bun_baseline","bun_delta","bun_ratio",
    "glucose_baseline","glucose_delta","glucose_ratio",
    "bilirubin_total_baseline","bilirubin_total_delta","bilirubin_total_ratio",
    "resp_rate_rollmean_3","resp_rate_rollstd_3","spo2_rollmean_6","spo2_rollstd_4",
    "heart_rate_mask","sbp_noninvasive_mask","dbp_noninvasive_mask",
    "sbp_invasive_mask","dbp_invasive_mask","map_invasive_mask",
    "temperature_c_mask","spo2_mask","resp_rate_mask",
    "creatinine_mask","wbc_mask","platelets_mask","lactate_mask","bun_mask",
    "bilirubin_total_mask","glucose_mask","hematocrit_mask","potassium_mask",
    "sodium_mask","troponin_t_mask","ph_venous_mask","pco2_venous_mask",
    "base_excess_mask","rbc_mask","chloride_mask","calcium_mask",
    "urine_output_mask","urine_output_ml_kg_hr_mask","weight_mask",
]

TREATMENT_FEATURES = [
    "total_crystalloid_ml", "has_norepinephrine_obs", "has_phenylephrine_obs",
    "has_dopamine_obs", "has_vasopressin_obs", "time_to_first_vaso_hrs",
    "early_steroid", "early_antibiotic", "n_distinct_meds",
    "steroid_ordered", "time_to_first_abx_order_hrs",
    "max_fio2_obs", "mean_fio2_obs", "max_peep_obs", "max_tidal_volume_obs",
    "high_fio2_flag", "on_peep_flag",
    "has_propofol_midaz_obs", "total_sedation_dose_obs",
    "age","gender_M","eth_WHITE","eth_BLACK","eth_HISPANIC","eth_ASIAN",
]

BINARY_COLS = {c for c in TREATMENT_FEATURES
               if c.startswith("has_") or c.startswith("eth_") or c == "gender_M"
               or c in ("early_steroid","early_antibiotic","steroid_ordered",
                        "high_fio2_flag","on_peep_flag")}

LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]