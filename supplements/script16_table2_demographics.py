"""
script16_table2_demographics.py
════════════════════════════════
Generates Table 2 (Baseline Patient Characteristics) from the paper.

Requires:
  - train_final_enriched.parquet  (from script1)
  - test_final_enriched.parquet   (from script1)

Note: length_of_stay and mortality columns must be present in parquet files.
These are extracted in script1 preprocessing but not used in the ML pipeline.
"""

import polars as pl
from pathlib import Path

BASE_PATH = Path("/kaggle/working")

# 1. Load Data
train_df = pl.read_parquet(BASE_PATH / "train_final_enriched.parquet")
test_df = pl.read_parquet(BASE_PATH / "test_final_enriched.parquet")

# Add 'Cohort' labels
train_df = train_df.with_columns(pl.lit("2014-2016 (Train/Val)").alias("Cohort"))
test_df = test_df.with_columns(
    pl.when(pl.col("anchor_year_group") == "2020 - 2022")
    .then(pl.lit("2020-2022 (Post-drift)"))
    .otherwise(pl.lit("2017-2019 (Pre-drift)"))
    .alias("Cohort")
)

# Combine into one dataframe
all_data = pl.concat([
    train_df.select(["stay_id", "Cohort", "age", "gender", "ethnicity", 
                     "length_of_stay", "mortality",
                     "label_vasopressor", "label_intubation", "label_septic_shock"]),
    test_df.select(["stay_id", "Cohort", "age", "gender", "ethnicity", 
                    "length_of_stay", "mortality",
                    "label_vasopressor", "label_intubation", "label_septic_shock"])
])

# Ensure one row per stay
unique_stays = all_data.group_by("stay_id").first()

# 2. Calculate Table 1 Statistics
table_1 = unique_stays.group_by("Cohort", maintain_order=True).agg([
    pl.len().alias("Total Stays (N)"),
    
    # Demographics
    pl.col("age").mean().round(1).alias("Age (Mean)"),
    pl.col("age").std().round(1).alias("Age (Std)"),
    (pl.col("gender") == "M").sum().alias("Male (N)"),
    ((pl.col("gender") == "M").sum() / pl.len() * 100).round(1).alias("Male (%)"),
    (pl.col("ethnicity") == "WHITE").sum().alias("White (N)"),
    ((pl.col("ethnicity") == "WHITE").sum() / pl.len() * 100).round(1).alias("White (%)"),
    (pl.col("ethnicity") == "BLACK").sum().alias("Black (N)"),
    ((pl.col("ethnicity") == "BLACK").sum() / pl.len() * 100).round(1).alias("Black (%)"),
    
    # Clinical Outcomes (NEW)
    pl.col("length_of_stay").mean().round(1).alias("LOS Days (Mean)"),
    pl.col("length_of_stay").std().round(1).alias("LOS Days (Std)"),
    pl.col("mortality").sum().alias("Mortality (N)"),
    (pl.col("mortality").mean() * 100).round(1).alias("Mortality (%)"),
    
    # Target Labels
    pl.col("label_vasopressor").sum().alias("Vasopressor (N)"),
    (pl.col("label_vasopressor").mean() * 100).round(1).alias("Vasopressor (%)"),
    pl.col("label_intubation").sum().alias("Intubation (N)"),
    (pl.col("label_intubation").mean() * 100).round(1).alias("Intubation (%)"),
    pl.col("label_septic_shock").sum().alias("Septic Shock (N)"),
    (pl.col("label_septic_shock").mean() * 100).round(1).alias("Septic Shock (%)")
]).sort("Cohort")

# 3. Display
pl.Config.set_tbl_rows(10)
pl.Config.set_tbl_cols(25)
print(table_1)