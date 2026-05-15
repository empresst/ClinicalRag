def map_features_to_query(top_features: list[str]) -> str:
    """
    Maps top model features (from IG or SHAP) to natural language descriptors
    for semantic PubMed retrieval (Based on Table 1).
    """
    lookup_table = {
        "total_crystalloid_ml": "high crystalloid volume resuscitation",
        "time_to_first_vaso_hrs": "early vasopressor initiation",
        "n_distinct_meds": "intensive polypharmacy",
        "max_fio2_obs": "elevated FiO2 mechanical ventilation",
        "max_peep_obs": "PEEP titration",
        "max_tidal_volume_obs": "tidal volume",
        "total_sedation_dose_obs": "high sedation dose",
        "has_propofol_midaz_obs": "propofol midazolam sedation",
        "has_norepinephrine_obs": "norepinephrine infusion"
    }
    
    # Map features, defaulting to the raw feature name if not in table
    mapped_terms = [lookup_table.get(feat, feat.replace("_", " ")) for feat in top_features]
    
    return " ".join(mapped_terms)