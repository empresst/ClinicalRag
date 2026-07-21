%%writefile utils/constants.py


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
    "gcs_eye", "gcs_verbal", "gcs_motor",
    "gcs_eye_mask", "gcs_verbal_mask", "gcs_motor_mask",
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
    "total_crystalloid_ml",
    "early_steroid",
    "early_antibiotic",
    "n_distinct_meds",
    "steroid_ordered",
    "has_blood_products_obs",
    "total_prbc_ml",
    "has_rrt_obs",
    "has_insulin_infusion_obs",
    "time_to_first_abx_order_hrs",
    "age",
    "gender_M",
]

BINARY_COLS = {
    "early_steroid",
    "early_antibiotic",
    "steroid_ordered",
    "has_blood_products_obs",
    "has_rrt_obs",
    "has_insulin_infusion_obs",
    "gender_M",
}
LABEL_COLS = ["label_vasopressor", "label_intubation", "label_septic_shock"]