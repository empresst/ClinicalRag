"""
script14_decontamination_verification.py
════════════════════════════════════════
Verifies zero subject-level leakage across all six partition pairs
and generates the decontamination summary table for supplementary materials.

Requires:
  - train_final_enriched.parquet from script1
  - val_final_enriched.parquet from script1
  - test_final_enriched.parquet from script1
  - mimiciv_demographics.parquet from script1
"""

import polars as pl
from pathlib import Path

# ══════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════
BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")
SAVE_PATH = Path("/kaggle/working")

# ══════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════
train_df = pl.read_parquet(BASE_PATH / "train_final_enriched.parquet")
val_df   = pl.read_parquet(BASE_PATH / "val_final_enriched.parquet")
test_df  = pl.read_parquet(BASE_PATH / "test_final_enriched.parquet")

# Get one row per stay
train_stays = train_df.filter(pl.col("hrs_from_admit") == 0)
val_stays   = val_df.filter(pl.col("hrs_from_admit") == 0)
test_stays  = test_df.filter(pl.col("hrs_from_admit") == 0)

# Split test into pre and post drift
pre_stays  = test_stays.filter(pl.col("anchor_year_group") == "2017 - 2019")
post_stays = test_stays.filter(pl.col("anchor_year_group") == "2020 - 2022")

# Unique subjects per partition
train_subjects = set(train_stays["subject_id"].to_list())
val_subjects   = set(val_stays["subject_id"].to_list())
pre_subjects   = set(pre_stays["subject_id"].to_list())
post_subjects  = set(post_stays["subject_id"].to_list())

# ══════════════════════════════════════
# PARTITION STATISTICS
# ══════════════════════════════════════
print("=== PARTITION STATISTICS ===")
print(f"Train:      {train_stays.height} stays | {len(train_subjects)} unique subjects")
print(f"Val:        {val_stays.height} stays | {len(val_subjects)} unique subjects")
print(f"Pre-drift:  {pre_stays.height} stays | {len(pre_subjects)} unique subjects")
print(f"Post-drift: {post_stays.height} stays | {len(post_subjects)} unique subjects")

# ══════════════════════════════════════
# ZERO LEAKAGE VERIFICATION
# ══════════════════════════════════════
train_val_overlap  = train_subjects & val_subjects
train_pre_overlap  = train_subjects & pre_subjects
train_post_overlap = train_subjects & post_subjects
val_pre_overlap    = val_subjects   & pre_subjects
val_post_overlap   = val_subjects   & post_subjects
pre_post_overlap   = pre_subjects   & post_subjects

print("\n=== SUBJECT OVERLAP VERIFICATION ===")
print(f"Train ∩ Val:        {len(train_val_overlap)} subjects")
print(f"Train ∩ Pre-drift:  {len(train_pre_overlap)} subjects")
print(f"Train ∩ Post-drift: {len(train_post_overlap)} subjects")
print(f"Val ∩ Pre-drift:    {len(val_pre_overlap)} subjects")
print(f"Val ∩ Post-drift:   {len(val_post_overlap)} subjects")
print(f"Pre ∩ Post-drift:   {len(pre_post_overlap)} subjects")

all_zero = all([
    len(train_val_overlap)  == 0,
    len(train_pre_overlap)  == 0,
    len(train_post_overlap) == 0,
    len(val_pre_overlap)    == 0,
    len(val_post_overlap)   == 0,
    len(pre_post_overlap)   == 0,
])
print(f"\n✅ Zero subject leakage confirmed: {all_zero}")

# ══════════════════════════════════════
# STAYS BEFORE AND AFTER PURGING
# ══════════════════════════════════════
demo_df = pl.read_parquet(BASE_PATH / "mimiciv_demographics.parquet")

total_train_val = demo_df.filter(
    pl.col("anchor_year_group") == "2014 - 2016").height
total_pre  = demo_df.filter(
    pl.col("anchor_year_group") == "2017 - 2019").height
total_post = demo_df.filter(
    pl.col("anchor_year_group") == "2020 - 2022").height

train_val_final = train_stays.height + val_stays.height

print("\n=== STAYS BEFORE AND AFTER FORWARD PURGING ===")
print(f"Train+Val (2014-2016): {total_train_val} before → "
      f"{train_val_final} after | removed: {total_train_val - train_val_final}")
print(f"Pre-drift (2017-2019): {total_pre} before → "
      f"{pre_stays.height} after | removed: {total_pre - pre_stays.height}")
print(f"Post-drift (2020-2022): {total_post} before → "
      f"{post_stays.height} after | removed: {total_post - post_stays.height}")

# ══════════════════════════════════════
# SUPPLEMENTARY TABLE
# ══════════════════════════════════════
print("\n=== SUPPLEMENTARY TABLE ===")
print(f"{'Partition':<30} {'Unique Subjects':>15} {'Before':>10} "
      f"{'Final':>10} {'Removed':>10}")
print("-" * 75)
print(f"{'Train+Val (2014-2016)':<30} "
      f"{len(train_subjects)+len(val_subjects):>15} "
      f"{total_train_val:>10} {train_val_final:>10} "
      f"{total_train_val-train_val_final:>10}")
print(f"{'  — Train':<30} {len(train_subjects):>15} "
      f"{'—':>10} {train_stays.height:>10} {'—':>10}")
print(f"{'  — Val':<30} {len(val_subjects):>15} "
      f"{'—':>10} {val_stays.height:>10} {'—':>10}")
print(f"{'Pre-drift (2017-2019)':<30} {len(pre_subjects):>15} "
      f"{total_pre:>10} {pre_stays.height:>10} "
      f"{total_pre-pre_stays.height:>10}")
print(f"{'Post-drift (2020-2022)':<30} {len(post_subjects):>15} "
      f"{total_post:>10} {post_stays.height:>10} "
      f"{total_post-post_stays.height:>10}")
print("-" * 75)
total_subjects = (len(train_subjects) + len(val_subjects) +
                  len(pre_subjects)   + len(post_subjects))
total_before = total_train_val + total_pre + total_post
total_final  = train_val_final + pre_stays.height + post_stays.height
print(f"{'Total':<30} {total_subjects:>15} "
      f"{total_before:>10} {total_final:>10} "
      f"{total_before-total_final:>10}")