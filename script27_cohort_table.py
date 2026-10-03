"""
script27_cohort_table.py
════════════════════════
Cohort characteristics by era (supplementary material, Table S3).

Eras follow the anchor-year group: 2008-2013 is the source train + validation
split, 2014-2019 the pre-drift test stream and 2020-2022 the post-drift era.
One row per stay (the hrs_from_admit == 0 row).

Inputs   <DATA_DIR>/{train,val,test}_final_enriched4.parquet  (script1)
Outputs  console table
"""
import os
from pathlib import Path
import polars as pl

DATA_DIR = Path(os.environ.get("DATA_DIR", "/kaggle/input/datasets/fatematamanna/allnew"))
COLS = ["stay_id", "anchor_year_group", "age", "gender", "ethnicity", "length_of_stay",
        "mortality", "label_vasopressor", "label_intubation", "label_septic_shock"]


def first_rows(split):
    df = pl.read_parquet(DATA_DIR / f"{split}_final_enriched4.parquet")
    if "hrs_from_admit" in df.columns:
        df = df.filter(pl.col("hrs_from_admit") == 0)
    return df.select(COLS).unique(subset="stay_id")


source = pl.concat([first_rows("train"), first_rows("val")]).with_columns(pl.lit("2008-13").alias("era"))
test = first_rows("test").with_columns(
    pl.when(pl.col("anchor_year_group") == "2020 - 2022").then(pl.lit("2020-22"))
      .otherwise(pl.lit("2014-19")).alias("era"))
df = pl.concat([source, test])


def pct(expr):
    return (expr.mean() * 100).round(1)


table = (df.group_by("era").agg(
    pl.len().alias("stays"),
    pl.col("age").mean().round(1).alias("age_mean"), pl.col("age").std().round(1).alias("age_sd"),
    (pl.col("gender") == "M").sum().alias("male_n"), pct(pl.col("gender") == "M").alias("male_pct"),
    pl.col("ethnicity").str.starts_with("WHITE").sum().alias("white_n"),
    pct(pl.col("ethnicity").str.starts_with("WHITE")).alias("white_pct"),
    pl.col("ethnicity").str.starts_with("BLACK").sum().alias("black_n"),
    pct(pl.col("ethnicity").str.starts_with("BLACK")).alias("black_pct"),
    pl.col("length_of_stay").mean().round(1).alias("los_mean"),
    pl.col("length_of_stay").std().round(1).alias("los_sd"),
    pct(pl.col("mortality")).alias("mortality_pct"),
    *[x for l in ("vasopressor", "intubation", "septic_shock")
      for x in (pl.col(f"label_{l}").sum().alias(f"{l}_n"), pct(pl.col(f"label_{l}")).alias(f"{l}_pct"))],
).sort("era"))

with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250):
    print(table.transpose(include_header=True, column_names="era"))
