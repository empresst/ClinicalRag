%%writefile evaluation/fig.py

"""
fig_biological_amnesia_v3_patch.py
Run AFTER script2_two_stream_model_v5.py
Loads pre-computed attribution data from fig_amnesia_data.pkl
4-panel "Biological Stability" figure comparing:
  Left column:  Run B (Proposed Two-Stream) — Source vs Adapted IG attributions
  Right column: XGBoost — Source vs Adapted SHAP values

Key visual: Run B physiology bars are nearly identical top-to-bottom
(LSTM frozen), while XGBoost physiology bars change dramatically
(biological amnesia).
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.transforms import blended_transform_factory
from pathlib import Path
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt
from utils.constants import SEQ_FEATURES, TREATMENT_FEATURES, BINARY_COLS, LABEL_COLS
from utils.data_utils import load_enriched_split, calculate_train_stats, normalize
from utils.train_utils import FocalBCEWithLogitsLoss, compute_pos_weights
from utils.data_utils import ICUDataset, SingleStreamDataset
from models.architectures import TwoStreamModel, PhysiologyStream, TreatmentStream, FusionHead, TwoStreamModel
import json, copy
import torch


SEED    = 42
SEQ_LEN = 6
BATCH_SIZE = 64
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_YEARS = ["2008 - 2010", "2011 - 2013"]
LEAKAGE_TIMING_FEATS = ["time_to_first_abx_order_hrs"]
SENTINEL_NO_EARLY_EVENT = float(SEQ_LEN + 1)
LABEL_IDX = 0   # label_vasopressor


# ── Load all fig variables ────────────────────────────────────────────────────
SAVE_PATH = Path("/kaggle/working")
BASE_PATH = Path("/kaggle/input/datasets/fatematamanna/allnew")

# ── Load eval split (all IDs from script2 — no hardcoded group names) ─────────
with open(SAVE_PATH / "eval_split.json") as f:
    _split = json.load(f)
eval_post_stays = list(map(int, _split["eval_post_stays"]))
post_cp_stays   = list(map(int, _split["post_cp_stays"]))
drift_tag            = _split.get("drift_tag", None)
# ── Load checkpoint — train_stats comes from here, not recomputed ─────────────
ckpt = torch.load(SAVE_PATH / "two_stream_models.pt",
                  map_location=DEVICE, weights_only=False)
seq_dim, treat_dim, n_targets = ckpt["seq_dim"], ckpt["treat_dim"], ckpt["n_targets"]
train_stats = ckpt["train_stats"]

# ── Load only val + test (train not needed — train_stats from checkpoint) ──────
print("Loading data...")
val_df  = load_enriched_split(BASE_PATH, "val",  SEQ_FEATURES, TREATMENT_FEATURES)
test_df = load_enriched_split(BASE_PATH, "test", SEQ_FEATURES, TREATMENT_FEATURES)

# ── Leakage clip (must match script2 exactly) ─────────────────────────────────
for feat in LEAKAGE_TIMING_FEATS:
    for df_ref, df_name in [(val_df, "val"), (test_df, "test")]:
        if feat not in df_ref.columns:
            continue
    val_df = val_df.with_columns(
        pl.when(pl.col(feat) > SEQ_LEN).then(SENTINEL_NO_EARLY_EVENT)
          .otherwise(pl.col(feat)).alias(feat))
    test_df = test_df.with_columns(
        pl.when(pl.col(feat) > SEQ_LEN).then(SENTINEL_NO_EARLY_EVENT)
          .otherwise(pl.col(feat)).alias(feat))

# ── Normalize using checkpoint train_stats ────────────────────────────────────
test_raw = test_df.clone()   # unnormalized — used for gender/lactate in suptitle
val_df   = normalize(val_df,  train_stats)
test_df  = normalize(test_df, train_stats)

# ── Drop direct label proxies (must match script2) ────────────────────────────
_drop = ["vasopressor_flag", "ventilation_flag"]
val_df  = val_df.drop([c for c in _drop if c in val_df.columns])
test_df = test_df.drop([c for c in _drop if c in test_df.columns])

# ── Splits from json IDs — no hardcoded group names ──────────────────────────
test_post    = test_df.filter(pl.col("stay_id").is_in(post_cp_stays))
eval_post_df = test_post.filter(pl.col("stay_id").is_in(eval_post_stays))
print(f"✅ eval_post_df: {eval_post_df['stay_id'].n_unique()} stays")

# ── Load models ───────────────────────────────────────────────────────────────
model_B = TwoStreamModel(seq_dim, treat_dim, n_targets).to(DEVICE)
model_B.load_state_dict(ckpt["run_b"])
model_B.eval()

model_B_source = TwoStreamModel(seq_dim, treat_dim, n_targets).to(DEVICE)
model_B_source.load_state_dict(ckpt["source"])
model_B_source.eval()
print("✅ Models loaded")


# ── Load saved XGBoost models ─────────────────────────────────────────────────

with open(SAVE_PATH / "fig_amnesia_data.pkl", "rb") as f:
    _d = pickle.load(f)

# ── All precomputed attribution data — loaded directly, nothing recomputed ─────
all_phys_deltas  = _d["all_phys_deltas"]
all_treat_deltas = _d["all_treat_deltas"]
all_xgb_deltas   = _d["all_xgb_deltas"]
shap_src_all     = _d["shap_src_all"]
xgb_feat_names   = _d["xgb_feat_names"]
xi_best          = _d["xi_best"]        # always 0 — single-patient array in pkl
p_b_src          = _d["p_b_src"]
p_b_adp          = _d["p_b_adp"]
p_xgb_src        = _d["p_xgb_src"]
p_xgb_adp        = _d["p_xgb_adp"]
runb_phys_norm   = _d["runb_phys_norm"]
runb_treat_norm  = _d["runb_treat_norm"]
xgb_phys_norm    = _d["xgb_phys_norm"]
xgb_treat_norm   = _d["xgb_treat_norm"]
runb_p95         = _d["runb_p95"]
xgb_p95          = _d["xgb_p95"]
amnesia_feat     = _d["amnesia_feat"]
delta            = _d["delta"]
stay_id          = _d["stay_id"]
age              = _d["age"]
true_label       = _d["true_label"]
treat_cols       = _d["treat_cols"]

# Derived immediately from pkl arrays — no recomputation needed
norm_ratio = xgb_phys_norm.mean() / max(runb_phys_norm.mean(), 1e-6)

# ── Helper functions ──────────────────────────────────────────────────────────
treat_col_set = set(treat_cols)

def clean(name):
    return (name.replace("physio:", "")
                .replace("treat:", "")
                .replace("_mask", "")
                .replace("_", " ").strip())

def is_treat(name):
    return (name in treat_col_set or
            name.replace("treat:", "") in treat_col_set)


_seq_no_mask = [c for c in SEQ_FEATURES if not c.endswith("_mask")]
BIO_FEATURES_TO_MONITOR = [
    f"{prefix}_{col}"
    for prefix in ["last", "mean", "std", "min", "max"]
    for col in _seq_no_mask
]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Precompute SHAP for all post-drift patients
# ══════════════════════════════════════════════════════════════════════════════

print("="*60)
print("GOLDEN PATIENT — loaded directly from pkl (script5 bridge)")
print("="*60)

# ── All attribution values already loaded from pkl above ──────────────────────
shap_src_val = all_xgb_deltas[amnesia_feat]["src"]
shap_adp_val = all_xgb_deltas[amnesia_feat]["adp"]
# delta, amnesia_feat, stay_id, age, true_label all from pkl

# ── Patient metadata from test_raw (unnormalized) ─────────────────────────────
pat   = test_raw.filter(pl.col("stay_id") == stay_id)
lac_v = pat["lactate"].drop_nulls() if "lactate" in pat.columns else pl.Series([])
lac   = float(lac_v.mean()) if len(lac_v) > 0 else 0.0

print(f"Golden patient: stay_id={stay_id}  amnesia_feat={amnesia_feat}")
print(f"  SHAP src={shap_src_val:+.4f}  adp={shap_adp_val:+.4f}  Δ={delta:.4f}")
print(f"  Age={int(age)}  Lactate={lac:.1f}")
print(f"  true_label={true_label} (from pkl — script2 selected true positive)")

# ── Build eval dataset — locate golden patient by stay_id, NOT by index 0 ─────
ds = ICUDataset(eval_post_df, SEQ_FEATURES, TREATMENT_FEATURES, LABEL_COLS, SEQ_LEN)
labels_post_arr = ds.labels

if stay_id not in ds.stay_ids:
    raise ValueError(
        f"stay_id {stay_id} from pkl not found in eval_post_df — "
        f"check that post_cp_stays and eval_post_stays are loaded correctly")
case_i = ds.stay_ids.index(stay_id)
print(f"  Located at dataset index {case_i} (out of {len(ds)} patients)")

# Verify true_label consistency between pkl and dataset
dataset_label = int(labels_post_arr[case_i, LABEL_IDX])
assert dataset_label == true_label, (
    f"true_label mismatch: pkl={true_label}, dataset={dataset_label}. "
    f"Data pipeline may differ from script2.")
print(f"  ✅ true_label verified consistent: {true_label}")

# ── Get tensors for the golden patient ───────────────────────────────────────
seq_t, treat_t, _ = ds[case_i]
seq_t   = seq_t.unsqueeze(0).to(DEVICE)
treat_t = treat_t.unsqueeze(0).to(DEVICE)

# ── Summary printout using all-pkl values (no recomputation) ──────────────────
phys_delta_vals  = np.array([v["delta"] for v in all_phys_deltas.values()])
treat_delta_vals = np.array([v["delta"] for v in all_treat_deltas.values()])
mean_phys_delta_b  = float(phys_delta_vals.mean())
mean_treat_delta_b = float(treat_delta_vals.mean())

print(f"\n  Run B IG deltas (from pkl):")
print(f"    Physiology: mean={mean_phys_delta_b:.4f}  max={phys_delta_vals.max():.4f}")
print(f"    Treatment:  mean={mean_treat_delta_b:.4f}  max={treat_delta_vals.max():.4f}")
print(f"\n  Normalized ratio (XGB/RunB physiology): {norm_ratio:.1f}×")
print(f"\n  Run B Source p={p_b_src:.3f}  |  Run B Adapted p={p_b_adp:.3f}")
print(f"  XGB Source p={p_xgb_src:.3f}   |  XGB Adapted p={p_xgb_adp:.3f}")
print(f"  True label = {true_label}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Draw 5-panel figure  
# ══════════════════════════════════════════════════════════════════════════════

# ── How many features to show ─────────────────────────────────────────────────
N_SHOW_P = 8   # physiology bars per panel  (was 10)
N_SHOW_T = 5   # treatment bars per panel   (was 8)
N_TOTAL  = N_SHOW_P + N_SHOW_T

# Re-derive display lists with the new limits
physio_imps_b_src = sorted(
    [(n, v["src"]) for n, v in all_phys_deltas.items()],
    key=lambda x: abs(x[1]), reverse=True)[:N_SHOW_P]
physio_feat_order = [n for n, _ in physio_imps_b_src]
physio_imps_b_adp = [(n, all_phys_deltas[n]["adp"]) for n in physio_feat_order]
phys_deltas_b = {n: all_phys_deltas[n]["delta"] for n in physio_feat_order}

treat_imps_b_src = sorted(
    [(n, v["src"]) for n, v in all_treat_deltas.items()],
    key=lambda x: abs(x[1]), reverse=True)[:N_SHOW_T]
treat_feat_order = [n for n, _ in treat_imps_b_src]
treat_imps_b_adp = [(n, all_treat_deltas[n]["adp"]) for n in treat_feat_order]
treat_deltas_b = {n: all_treat_deltas[n]["delta"] for n in treat_feat_order}

# XGBoost: also limit to N_TOTAL
top_src_idx    = np.argsort(np.abs(shap_src_all[xi_best]))[::-1][:N_TOTAL]
xgb_feat_order = [xgb_feat_names[i] for i in top_src_idx]
xgb_src_vals   = [all_xgb_deltas[n]["src"] for n in xgb_feat_order]
xgb_adp_vals   = [all_xgb_deltas[n]["adp"] for n in xgb_feat_order]
xgb_delta_vals_display = {n: all_xgb_deltas[n]["delta"] for n in xgb_feat_order}

# ── Shared color palette (all panels now use the same set) ───────────────────
C_PHYSIO_POS = "#4C78A8"   # blue
C_PHYSIO_NEG = "#9BBCD6"   # light-blue
C_TREAT_POS  = "#54A24B"   # green
C_TREAT_NEG  = "#A8D5A2"   # light-green
C_AMNESIA    = "#FF6B00"   # orange (amnesia label only, no arrow)
C_STABLE     = "#1D9E75"   # teal

print("\nDrawing figure (v3) ...")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig = plt.figure(figsize=(20, 20))          # taller than v2 (was 18)
gs  = gridspec.GridSpec(3, 2, figure=fig,
                         height_ratios=[1, 1, 0.50],
                         hspace=0.55, wspace=0.42)  # more hspace

ax_b_src  = fig.add_subplot(gs[0, 0])
ax_b_adp  = fig.add_subplot(gs[1, 0])
ax_x_src  = fig.add_subplot(gs[0, 1])
ax_x_adp  = fig.add_subplot(gs[1, 1])
ax_deltas = fig.add_subplot(gs[2, :])


# ─────────────────────────────────────────────────────────────────────────────
def draw_runb_panel(ax, physio_imps, treat_imps, title, p_val,
                    bg_color, deltas_dict=None):
    """IG panel — shared color palette, no overlapping labels."""
    ax.set_facecolor(bg_color)

    all_items = ([(n, v, "P") for n, v in physio_imps] +
                 [(None, None, None)] +
                 [(n, v, "T") for n, v in treat_imps])
    names, values, tags, feat_names_raw = [], [], [], []
    for item in all_items:
        if item[0] is None:
            names.append(""); values.append(0); tags.append("GAP"); feat_names_raw.append(None)
        else:
            names.append(clean(item[0])); values.append(item[1])
            tags.append(item[2]); feat_names_raw.append(item[0])

    yp = np.arange(len(names))
    colors = []
    for v, tag in zip(values, tags):
        if tag == "GAP":   colors.append("none")
        elif tag == "P":   colors.append(C_PHYSIO_POS if v > 0 else C_PHYSIO_NEG)
        else:              colors.append(C_TREAT_POS if v > 0 else C_TREAT_NEG)

    bars = ax.barh(yp, values, color=colors,
                   edgecolor="white", linewidth=0.4, height=0.60)
    ax.axvline(0, color="black", lw=0.8, zorder=5)
    ax.set_yticks(yp)
    ax.set_yticklabels(names, fontsize=8)
    ax.tick_params(axis="y", pad=6)          # push tick labels left of bars
    ax.invert_yaxis()
    ax.yaxis.grid(False); ax.xaxis.grid(True, alpha=0.15, linestyle="--")

    gap_idx = next(i for i, t in enumerate(tags) if t == "GAP")
    ax.axhline(y=gap_idx, color="#999", lw=0.8, linestyle=":")

    correct = p_val >= 0.5 and true_label == 1
    bc = "#E1F5EE" if correct else "#FCEBEB"
    ec = "#1D9E75" if correct else "#E24B4A"
    tc = "#085041" if correct else "#791F1F"
    verdict = "CORRECT" if correct else "MISSED"
    # Bottom-left: bars end on the right side, left corner is always clear
    ax.text(0.02, 0.02, f"p={p_val:.3f}  {verdict}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", color=tc,
            bbox=dict(boxstyle="round,pad=0.3", fc=bc, ec=ec, lw=1.2))

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Integrated Gradients Attribution", fontsize=9)

    non_zero = [abs(v) for v in values if v != 0]
    xr = max(non_zero) if non_zero else 1.0
    x_neg = min((v for v in values if v < 0), default=0)
    ax.set_xlim(x_neg - xr * 0.05, xr * 1.05)

    # ── Three fixed annotation columns (axes-fraction coords) ────────────────
    # IMPORTANT: invert_yaxis() flips the display but NOT axes-fraction coords.
    # Bar i=0 displays at the TOP visually but sits at axes-fraction y≈0 (bottom).
    # So we flip: row_y = 1 - (i + 0.5) / n_rows  → i=0 → y≈1 (top of box).
    n_rows    = len(values)
    col_val = min(values) / ax.get_xlim()[1] + 0.02
    col_delta = 0.76
    col_badge = 0.90

    for i, (bar, val, tag, fn) in enumerate(zip(bars, values, tags, feat_names_raw)):
        if tag == "GAP" or val == 0:
            continue
        # Flip to match visual order after invert_yaxis
        row_y = 1.0 - (i + 0.5) / n_rows

        ax.text(val + xr * 0.025, i, f"{val:+.3f}",
        va="center", ha="left", fontsize=7, color="#333")

        if deltas_dict and fn and fn in deltas_dict:
            d  = deltas_dict[fn]
            dc = C_STABLE if d < 0.02 else (C_AMNESIA if d > 0.1 else "#888")
            ax.text(col_delta, row_y, f"Δ={d:.3f}",
                    va="center", ha="left", fontsize=6.5,
                    fontweight="bold", color=dc,
                    transform=ax.transAxes)

    # "LSTM FROZEN" badge — at visual bottom-right (last row = highest i = lowest row_y)
    # Sits beside the last feature's Δ value, well away from p-value at top-right
    last_row_y = 1.0 - (n_rows - 0.5) / n_rows
    ax.text(col_badge, last_row_y,
            "LSTM\nFROZEN", va="center", ha="left",
            fontsize=6.5, fontweight="bold", color="#085041",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25",
                      fc="#E1F5EE", ec="#1D9E75", lw=0.8))


# ─────────────────────────────────────────────────────────────────────────────
def draw_xgb_panel(ax, feat_names, shap_vals, title, p_val, bg_color,
                   deltas_dict=None, amnesia_feature=None, amnesia_delta=None):
    """
    SHAP panel — now uses IDENTICAL color scheme as IG panel.
    No bent-arrow annotation; amnesia bar gets an inline bold label instead.
    """
    ax.set_facecolor(bg_color)

    names  = [clean(n) for n in feat_names]
    values = list(shap_vals)
    yp     = np.arange(len(names))

    # Build colors using the SAME palette as IG
    colors = []
    is_amnesia_bar = []
    for n, v in zip(feat_names, values):
        is_t = is_treat(n)
        amnesia = bool(amnesia_feature and amnesia_feature in n)
        is_amnesia_bar.append(amnesia)
        if is_t:
            colors.append(C_TREAT_POS if v > 0 else C_TREAT_NEG)
        else:
            colors.append(C_PHYSIO_POS if v > 0 else C_PHYSIO_NEG)

    bars = ax.barh(yp, values, color=colors,
                   edgecolor="white", linewidth=0.4, height=0.60)
    ax.axvline(0, color="black", lw=0.8, zorder=5)
    ax.set_yticks(yp)
    ax.set_yticklabels(names, fontsize=8)
    ax.tick_params(axis="y", pad=6)
    ax.invert_yaxis()
    ax.yaxis.grid(False); ax.xaxis.grid(True, alpha=0.15, linestyle="--")

    non_zero = [abs(v) for v in values if v != 0]
    xr = max(non_zero) if non_zero else 1.0

    # Keep xlim tight to bars — annotation columns use axes-fraction coords
    # so they need no data-space padding at all.
    x_neg = min((v for v in values if v < 0), default=0)
    ax.set_xlim(x_neg - xr * 0.05, xr * 1.05)

    # ── Three fixed annotation columns (axes-fraction coords) ────────────────
    # row_y = 1 - (i+0.5)/n_rows  so i=0 maps to top of axes box,
    # matching visual position after invert_yaxis().
    # AFTER
    col_val = min(values) / ax.get_xlim()[1] + 0.02
    col_delta = 0.76
    col_badge = 0.90
    n_rows    = len(values)
    amnesia_row_y = None

    for i, (bar, val, n, amnesia) in enumerate(
            zip(bars, values, feat_names, is_amnesia_bar)):

        row_y = 1.0 - (i + 0.5) / n_rows   # flipped to match invert_yaxis

        ax.text(val + xr * 0.025, i, f"{val:+.3f}",
        va="center", ha="left", fontsize=7, color="#333")

        if deltas_dict and n in deltas_dict:
            d  = deltas_dict[n]
            dc = C_STABLE if d < 0.02 else (C_AMNESIA if d > 0.1 else "#888")
            ax.text(col_delta, row_y, f"Δ={d:.3f}",
                    va="center", ha="left", fontsize=6.5,
                    fontweight="bold", color=dc,
                    transform=ax.transAxes)

        if amnesia and amnesia_delta is not None:
            amnesia_row_y = row_y   # remember for badge placement after loop

    # "FORGOTTEN" badge — same visual row as amnesia feature, right of Δ
    if amnesia_row_y is not None:
        ax.text(col_badge, amnesia_row_y, "◀ FORGOTTEN",
                va="center", ha="left", fontsize=7,
                fontweight="bold", color="white",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.25",
                          fc=C_AMNESIA, ec="none", alpha=0.92))

    # p-value badge — bottom-left corner, always clear (bars extend rightward)
    correct = p_val >= 0.5 and true_label == 1
    bc = "#E1F5EE" if correct else "#FCEBEB"
    ec2 = "#1D9E75" if correct else "#E24B4A"
    tc3 = "#085041" if correct else "#791F1F"
    verdict = "CORRECT" if correct else "MISSED"
    ax.text(0.13, 0.02, f"p={p_val:.3f}  {verdict}",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", color=tc3,
            bbox=dict(boxstyle="round,pad=0.3", fc=bc, ec=ec2, lw=1.2))

    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("SHAP Value", fontsize=9)


# ── Draw panels ───────────────────────────────────────────────────────────────
pre_era  = f"pre-{drift_tag}"   # e.g. "pre-2020 - 2022"
post_era = drift_tag  

draw_runb_panel(
    ax_b_src, physio_imps_b_src, treat_imps_b_src,
    f"Run B — Source Weights ({pre_era})\nBefore Adaptation",
    p_b_src, "#F0F4F8")

all_display_deltas_b = {**phys_deltas_b, **treat_deltas_b}
draw_runb_panel(
    ax_b_adp, physio_imps_b_adp, treat_imps_b_adp,
    f"Run B — Adapted Weights ({post_era})\nPhysiology LSTM Frozen, Treatment+Fusion Updated",
    p_b_adp, "#E8F4F8",
    deltas_dict=all_display_deltas_b)

draw_xgb_panel(
    ax_x_src, xgb_feat_order, xgb_src_vals,
    f"XGBoost — Source  ({pre_era})\nAll features jointly trained",
    p_xgb_src, "#F0F4F8")          # same light-blue bg as Run B source

draw_xgb_panel(
    ax_x_adp, xgb_feat_order, xgb_adp_vals,
    f"XGBoost — Adapted (Retrained ({post_era}))\n{clean(amnesia_feat)} — Biological Amnesia",
    p_xgb_adp, "#E8F4F8",
    deltas_dict=xgb_delta_vals_display,
    amnesia_feature=amnesia_feat, amnesia_delta=delta)

# "All trees retrained" badge — top-left of XGBoost adapted panel
# ax_x_adp.text(0.02, 0.98, "ALL TREES\nRETRAINED",
#               va="top", ha="left", fontsize=6.5,
#               fontweight="bold", color="#791F1F",
#               transform=ax_x_adp.transAxes,
#               bbox=dict(boxstyle="round,pad=0.25",
#                         fc="#FCEBEB", ec="#E24B4A", lw=0.8))

# ── Bottom panel: normalized delta comparison (unchanged logic) ───────────────
ax_deltas.set_facecolor("#FAFAFA")
norm_ratio = xgb_phys_norm.mean() / max(runb_phys_norm.mean(), 1e-6)

bar_data = [
    ("Run B\nPhysiology",  float(runb_phys_norm.mean()),  C_PHYSIO_POS,
     f"n={len(runb_phys_norm)} feats",  float(runb_phys_norm.max())),
    ("Run B\nTreatment",   float(runb_treat_norm.mean()), C_TREAT_POS,
     f"n={len(runb_treat_norm)} feats", float(runb_treat_norm.max())),
    ("XGBoost\nPhysiology",float(xgb_phys_norm.mean()),  C_PHYSIO_POS,
     f"n={len(xgb_phys_norm)} base feats", float(xgb_phys_norm.max())),
    ("XGBoost\nTreatment", float(xgb_treat_norm.mean()), C_TREAT_POS,
     f"n={len(xgb_treat_norm)} feats",  float(xgb_treat_norm.max())),
]

x_pos      = np.arange(len(bar_data))
bar_vals   = [d[1] for d in bar_data]
bar_colors = [d[2] for d in bar_data]
bar_labels = [d[0] for d in bar_data]
bar_counts = [d[3] for d in bar_data]
bar_maxes  = [d[4] for d in bar_data]

bars_d = ax_deltas.bar(x_pos, bar_vals, color=bar_colors,
                        edgecolor="white", width=0.55, alpha=0.85)

for i, (bv, mx) in enumerate(zip(bar_vals, bar_maxes)):
    ax_deltas.plot(i, mx, marker="v", color="black", markersize=7, zorder=5)
    ax_deltas.plot([i, i], [bv, mx], color="black", lw=1, linestyle="--", alpha=0.5)
    ax_deltas.text(i, mx + 0.01, f"max={mx:.3f}", ha="center", va="bottom",
                   fontsize=7.5, fontweight="bold", color="#333")

for bar, val, cnt in zip(bars_d, bar_vals, bar_counts):
    ax_deltas.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                   f"mean={val:.4f}", ha="center", va="bottom",
                   fontsize=8, fontweight="bold", color="#333")
    ax_deltas.text(bar.get_x() + bar.get_width() / 2, -0.015,
                   cnt, ha="center", va="top", fontsize=7, color="#666")

ax_deltas.set_xticks(x_pos)
ax_deltas.set_xticklabels(bar_labels, fontsize=9)
ax_deltas.set_ylabel("Normalized Mean |Δ|\n(Δ / 95th-pctl of source)", fontsize=9)
ax_deltas.set_title(
    f"Attribution Stability — Normalized (Lower = More Stable)  |  "
    f"XGBoost physio. grouped by base feature  |  Ratio = {norm_ratio:.1f}×",
    fontsize=10, fontweight="bold", pad=10)
ax_deltas.axhline(0, color="black", lw=0.5)
ax_deltas.grid(axis="y", alpha=0.2, linestyle="--")

ax_deltas.annotate(
    "Physiology FROZEN\n→ small relative Δ",
    xy=(0, bar_vals[0]), xytext=(0.5, max(bar_vals) * 0.7),
    fontsize=8.5, color=C_STABLE, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_STABLE, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", fc="#E1F5EE", ec=C_STABLE, lw=1))

ax_deltas.annotate(
    f"Full retrain → {norm_ratio:.1f}× larger\nrelative Δ (amnesia)",
    xy=(2, bar_vals[2]),
    xytext=(2 - 0.5, max(bar_vals) * 0.7),
    fontsize=8.5, color=C_AMNESIA, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=C_AMNESIA, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3E0", ec=C_AMNESIA, lw=1))

# ── Column headers ────────────────────────────────────────────────────────────
fig.text(0.27, 0.97,
         "PROPOSED: Two-Stream Run B — Selective Adaptation",
         ha="center", va="top", fontsize=12, fontweight="bold",
         color=C_STABLE,
         bbox=dict(boxstyle="round,pad=0.4", fc="#E1F5EE", ec=C_STABLE, lw=1.5))
fig.text(0.73, 0.97,
         "BASELINE: XGBoost — Full Retrain",
         ha="center", va="top", fontsize=12, fontweight="bold",
         color=C_AMNESIA,
         bbox=dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec=C_AMNESIA, lw=1.5))

fig.text(0.01, 0.82, "SOURCE\n(Pre-drift)", va="center", ha="left",
         fontsize=8, fontweight="bold", color="#888", rotation=90,
         transform=fig.transFigure)
fig.text(0.01, 0.52, "ADAPTED\n(Post-drift)", va="center", ha="left",
         fontsize=8, fontweight="bold", color="#888", rotation=90,
         transform=fig.transFigure)

# ── Suptitle ──────────────────────────────────────────────────────────────────
pat_info = test_raw.filter(pl.col("stay_id") == stay_id)
gender_v = pat_info["gender"][0] if "gender" in pat_info.columns else "N/A"

fig.suptitle(
    f"Biological Stability: Two-Stream Run B (Proposed) vs XGBoost (Baseline)\n"
    f"Patient stay {stay_id} (age {int(age)}, {gender_v}, 2020-2022)  |  "
    f"Vasopressor true label = {true_label}  |  "
    f"Amnesia feature: {clean(amnesia_feat)} (ΔSHAP={delta:.3f})",
    fontsize=11, fontweight="bold", y=1.01
)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(color=C_PHYSIO_POS, label="Physiology (+risk)"),
    mpatches.Patch(color=C_PHYSIO_NEG, label="Physiology (−risk)"),
    mpatches.Patch(color=C_TREAT_POS,  label="Treatment (+risk)"),
    mpatches.Patch(color=C_TREAT_NEG,  label="Treatment (−risk)"),
    mpatches.Patch(color=C_AMNESIA,    label="Forgotten feature (amnesia)"),
    plt.Line2D([0], [0], marker="v", color="black", linestyle="None",
               markersize=7, label="Max |Δ| in category"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=3, fontsize=8.5, frameon=True,
           bbox_to_anchor=(0.5, -0.02))

# ── Caption ───────────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.06,
    f"Top 4 panels: per-feature attributions (IG for Run B, SHAP for XGBoost) with Δ annotations on adapted panels.\n"
    f"Bottom panel: deltas normalized by each model's 95th-pctl source attribution "
    f"(IG p95={runb_p95:.3f}, SHAP p95={xgb_p95:.3f}).\n"
    f"XGBoost physiology grouped by base feature (max Δ across last/mean/std/min/max). "
    f"Normalized ratio: {norm_ratio:.1f}× (>1 = XGBoost less stable).",
    ha="center", va="top", fontsize=8.5,
    style="italic", color="#444", linespacing=1.5
)

plt.subplots_adjust(
    left=0.09, right=0.95,
    top=0.90, bottom=0.10,
    hspace=0.55, wspace=0.42
)
plt.savefig(SAVE_PATH / "fig_biological_amnesia_v3.png",
            dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ Saved → fig_biological_amnesia_v3.png")



# Derive xgb physiology delta base from pkl for final summary
import re
_prefix_re = re.compile(r'^(last|mean|std|min|max)_(.+)$')
xgb_base_phys_deltas = {}
for name, v in all_xgb_deltas.items():
    if is_treat(name):
        continue
    m = _prefix_re.match(name)
    base = m.group(2) if m else name
    if base not in xgb_base_phys_deltas or v["delta"] > xgb_base_phys_deltas[base]:
        xgb_base_phys_deltas[base] = v["delta"]
xgb_phys_delta_base = np.array(list(xgb_base_phys_deltas.values()))

print(f"Key numbers for paper:")
print(f"  Patient: stay_id={stay_id}, age={int(age)}, lactate={lac:.1f}")
print(f"  Amnesia feature: {clean(amnesia_feat)}")
print(f"  XGB Source SHAP:  {shap_src_val:+.4f}")
print(f"  XGB Adapted SHAP: {shap_adp_val:+.4f}")
print(f"  SHAP Delta:       {delta:.4f}")
print(f"  Run B Source p={p_b_src:.3f}  |  Run B Adapted p={p_b_adp:.3f}")
print(f"  XGB Source p={p_xgb_src:.3f}   |  XGB Adapted p={p_xgb_adp:.3f}")
print(f"  Run B physiology:    raw mean |ΔIG|={mean_phys_delta_b:.4f}  normalized={runb_phys_norm.mean():.4f}")
print(f"  XGBoost physiology:  raw mean |ΔSHAP|={xgb_phys_delta_base.mean():.4f}  normalized={xgb_phys_norm.mean():.4f}")
print(f"  Normalization: Run B p95={runb_p95:.4f}  XGBoost p95={xgb_p95:.4f}")
print(f"  Normalized ratio (XGB/RunB physiology): {norm_ratio:.1f}×")