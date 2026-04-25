"""
Dataset characterization figures for paper (IP-safe, aggregate statistics only).

Figure 1 — Dataset scale and target size distribution
  (a) Binder pairs per target (sorted bar, log scale)
  (b) Epitope length distribution across 65 targets

Figure 2 — Target chemical diversity via clustering
  (a) Clustered AA composition heatmap (65 targets × 20 AA) with
      side annotation bars showing epitope length and dataset size
  (b) Clustered physicochemical property heatmap (65 targets × 8 properties)
      — same target ordering from (a), colour-coded by cluster

Both panels in Figure 2 share the same hierarchical clustering order so
the two views can be read jointly.

Usage:
    python plot_dataset_figures.py

Outputs (same folder as script):
    dataset_fig1_scale.pdf / .png
    dataset_fig2_clustering.pdf / .png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────
DATA_PATH = Path("/app/data/training.csv")
OUT_DIR   = Path(__file__).parent

# ── Kyte-Doolittle scale ────────────────────────────────────────────────────
KD = {'A': 1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C': 2.5,'Q':-3.5,'E':-3.5,
      'G':-0.4,'H':-3.2,'I': 4.5,'L': 3.8,'K':-3.9,'M': 1.9,'F': 2.8,
      'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V': 4.2}
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# ── load ────────────────────────────────────────────────────────────────────
print("Loading training data …")
df = pd.read_csv(DATA_PATH)
df["pep_len"] = df["peptide"].str.len()

pairs_per_target = (
    df.groupby("peptide").size()
    .sort_values(ascending=False)
    .reset_index(drop=False)
)
pairs_per_target.columns = ["peptide", "n_pairs"]
n_targets = len(pairs_per_target)
print(f"  {len(df):,} pairs | {n_targets} targets")

target_df = df.groupby("peptide").agg(
    pep_len=("pep_len", "first"),
    n_pairs=("pep_len", "count"),
).reset_index()

# ── per-target feature engineering ─────────────────────────────────────────
def aa_freq(seq):
    """20-dim AA frequency vector."""
    n = max(len(seq), 1)
    return np.array([seq.upper().count(aa) / n for aa in AA_ORDER])

def physico(seq):
    """8 interpretable physicochemical features."""
    s = seq.upper()
    n = max(len(s), 1)
    gravy      = sum(KD.get(a, 0) for a in s) / n
    pct_chrg   = sum(s.count(a) for a in "RKDE") / n
    pct_pos    = sum(s.count(a) for a in "RK")   / n
    pct_neg    = sum(s.count(a) for a in "DE")   / n
    net_charge = (pct_pos - pct_neg)
    pct_hydro  = sum(s.count(a) for a in "ILVFMW") / n
    pct_arom   = sum(s.count(a) for a in "FWY")    / n
    pct_dis    = sum(s.count(a) for a in "GPST")   / n   # disorder-promoting
    pct_polar  = sum(s.count(a) for a in "STNQ")   / n
    return [gravy, pct_chrg, net_charge, pct_hydro, pct_arom, pct_dis, pct_polar, pct_pos]

comp_mat   = np.vstack([aa_freq(p)  for p in target_df["peptide"]])   # (65, 20)
physco_mat = np.vstack([physico(p)  for p in target_df["peptide"]])   # (65, 8)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Scale and length distribution
# ════════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 4.8))
fig1.subplots_adjust(wspace=0.40)

# (a) pairs per target
ax = axes1[0]
colors_bar = ["#4C72B0" if n >= 1000 else "#DD8452" for n in pairs_per_target["n_pairs"]]
ax.bar(range(n_targets), pairs_per_target["n_pairs"],
       color=colors_bar, width=0.85, linewidth=0)
ax.set_yscale("log")
ax.set_xlabel("Target (ranked by dataset size)", fontsize=11)
ax.set_ylabel("Binder pairs (log scale)", fontsize=11)
ax.set_title("(a) Binder pairs per epitope target", fontsize=11, fontweight="bold")
ax.set_xlim(-1, n_targets)
ax.set_xticks([0, 16, 32, 48, 64])
ax.set_xticklabels(["T-1", "T-17", "T-33", "T-49", "T-65"], fontsize=9)
ax.tick_params(axis="y", labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.axhline(1000, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
# legend — upper right, no frame, explicit location so no ambiguity
p1 = mpatches.Patch(color="#4C72B0", label="≥1,000 pairs")
p2 = mpatches.Patch(color="#DD8452", label="<1,000 pairs")
ax.legend(handles=[p1, p2], fontsize=9,
          loc="upper right", bbox_to_anchor=(1.0, 1.0))
# (b) epitope length histogram
ax2 = axes1[1]
epi_lengths = target_df["pep_len"].values
bins = np.logspace(np.log10(10), np.log10(900), 32)
short_mask = epi_lengths <= 50
ax2.hist(epi_lengths[short_mask],  bins=bins, color="#4C72B0", alpha=0.85,
         label=f"Short (≤50 AA,  n={short_mask.sum()})",
         edgecolor="white", linewidth=0.4)
ax2.hist(epi_lengths[~short_mask], bins=bins, color="#DD8452", alpha=0.85,
         label=f"Long  (>50 AA,  n={(~short_mask).sum()})",
         edgecolor="white", linewidth=0.4)
ax2.axvline(50, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7)
ax2.set_xscale("log")
ax2.set_xlabel("Epitope length (AA, log scale)", fontsize=11)
ax2.set_ylabel("Number of targets", fontsize=11)
ax2.set_title("(b) Epitope length distribution across targets",
              fontsize=11, fontweight="bold", pad=8)
ax2.tick_params(labelsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
# legend — upper right
ax2.legend(fontsize=9, loc="upper right")
# stats box — lower right, below the tail of the distribution
for ext in ("pdf", "png"):
    fig1.savefig(OUT_DIR / f"dataset_fig1_scale.{ext}", bbox_inches="tight", dpi=300)
print("Saved dataset_fig1_scale.pdf/.png")
plt.close(fig1)


# ── summary ──────────────────────────────────────────────────────────────────
print(f"\nSummary:")
print(f"  Targets:                 {n_targets}")
print(f"  Total SFT pairs:         {len(df):,}")
print(f"  Total DPO pairs:         {len(dpo):,}")
print(f"  Short targets (≤50 AA):  {short_mask.sum()}/{n_targets}")
print(f"  GRAVY range:             {gravy_vals.min():.2f} to {gravy_vals.max():.2f}")
print(f"  Affinity Δmedian logKd:  {delta_aff:+.3f}")
print(f"  Dev.    Δmedian logKd:   {delta_dev:+.3f}")
print(f"  Affinity % above diag:   {100*np.mean(aff['rejected_logkd']>aff['chosen_logkd']):.1f}%")
print(f"  Dev.    % above diag:    {100*np.mean(dev['rejected_logkd']>dev['chosen_logkd']):.1f}%")
