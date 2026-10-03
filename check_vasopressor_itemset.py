"""
check_vasopressor_itemset.py
════════════════════════════════════════════════════════════════════
Answers the two unverified claims in Sec. 2.1:

  (1) What is itemid 227531?  (is it really mannitol?)
  (2) What share of vasopressor positives are "mannitol-only", and what
      share would be ADDED if vasopressin (222315) were included?

Mirrors the label_vasopressor logic in script1_preprocessing.py exactly:
    itemid IN (221906,221289,221662,221749,227531)
    AND starttime < intime + 14h
    AND COALESCE(endtime, starttime) > intime + 8h

Run on Kaggle with the MIMIC-IV dataset attached.
"""

import duckdb
import polars as pl
from pathlib import Path

import os
DATA = Path(os.environ.get("MIMIC_DIR", "/kaggle/input/datasets/fatematamanna/mimic4/mimic-iv-3.1"))
ENRICHED = Path(os.environ.get("DATA_DIR", "/kaggle/input/datasets/fatematamanna/allnew"))

PRED_START_H, PRED_END_H, MIN_STAY_HOURS = 8, 14, 14
CURRENT_SET = (221906, 221289, 221662, 221749, 227531)
VASOPRESSIN = 222315

con = duckdb.connect()
for tbl, path in [("icustays", "icu/icustays.csv"),
                  ("inputevents", "icu/inputevents.csv"),
                  ("d_items", "icu/d_items.csv")]:
    con.execute(f"CREATE OR REPLACE VIEW {tbl} AS SELECT * FROM '{DATA}/{path}'")

# ── (1) What are these itemids actually called? ───────────────────────────────
print("=" * 64)
print("ITEMID IDENTITIES  (from d_items)")
print("=" * 64)
ids = ",".join(str(i) for i in CURRENT_SET + (VASOPRESSIN,))
print(con.execute(f"""
    SELECT itemid, label, category, unitname
    FROM d_items WHERE itemid IN ({ids}) ORDER BY itemid
""").pl())

# ── Cohort: the stays actually used in the paper ──────────────────────────────
stay_ids = set()
for split in ["train", "val", "test"]:
    for cand in [ENRICHED / f"{split}_final_enriched4.parquet",
                 ENRICHED / f"{split}_final_enriched.parquet"]:
        if cand.exists():
            stay_ids |= set(pl.read_parquet(cand, columns=["stay_id"])["stay_id"].to_list())
            break
print(f"\nCohort stays loaded: {len(stay_ids):,}")
con.execute("CREATE OR REPLACE TABLE cohort AS SELECT * FROM (VALUES "
            + ",".join(f"({s})" for s in stay_ids) + ") t(stay_id)")

# ── (2) Attribute each positive to the itemids that caused it ─────────────────
q = f"""
WITH win AS (
    SELECT i.stay_id,
           i.intime + INTERVAL '{PRED_START_H}' HOUR AS w_start,
           i.intime + INTERVAL '{PRED_END_H}'   HOUR AS w_end
    FROM icustays i JOIN cohort c ON c.stay_id = i.stay_id
),
hits AS (
    SELECT w.stay_id, v.itemid
    FROM win w JOIN inputevents v ON v.stay_id = w.stay_id
    WHERE v.itemid IN ({",".join(str(i) for i in CURRENT_SET + (VASOPRESSIN,))})
      AND v.starttime < w.w_end
      AND COALESCE(v.endtime, v.starttime) > w.w_start
),
per_stay AS (
    SELECT stay_id,
      MAX(CASE WHEN itemid IN (221906,221289,221662,221749) THEN 1 ELSE 0 END) AS real_vaso,
      MAX(CASE WHEN itemid = 227531  THEN 1 ELSE 0 END) AS item_227531,
      MAX(CASE WHEN itemid = {VASOPRESSIN} THEN 1 ELSE 0 END) AS vasopressin
    FROM hits GROUP BY stay_id
)
SELECT
  SUM(CASE WHEN real_vaso=1 OR item_227531=1 THEN 1 ELSE 0 END) AS positives_as_labelled,
  SUM(CASE WHEN item_227531=1 AND real_vaso=0 THEN 1 ELSE 0 END) AS only_227531,
  SUM(CASE WHEN vasopressin=1 AND real_vaso=0 AND item_227531=0 THEN 1 ELSE 0 END) AS only_vasopressin
FROM per_stay
"""
r = con.execute(q).pl().row(0, named=True)
pos, only531, onlyvp = r["positives_as_labelled"], r["only_227531"], r["only_vasopressin"]

print("\n" + "=" * 64)
print("VASOPRESSOR LABEL COMPOSITION")
print("=" * 64)
print(f"  Positives as labelled (current 5-item set): {pos:,}")
print(f"  Positives from 227531 ALONE:                {only531:,}"
      f"   ({100*only531/max(pos,1):.2f}% of positives)")
print(f"  Stays vasopressin would ADD (not currently positive): {onlyvp:,}"
      f"   ({100*onlyvp/max(pos,1):.2f}% of positives)")
print()
print("  Paper claims each is < 1%. Check both figures above.")
