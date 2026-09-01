"""Generate the two result charts for the write-up. Not part of the
pipeline - a one-off script, rerun manually if the underlying numbers change.

Numbers are copied from experiments/*/results.md (paired bootstrap means/CIs)
and experiments/model_sweep/results.md (lambda sweep) - not recomputed here.

Usage:
    python docs/figures/make_charts.py
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent

# --- palette (dataviz skill reference instance, light mode) ---
SURFACE      = "#fcfcfb"
INK_PRIMARY  = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED    = "#898781"
GRID         = "#e1e0d9"
BASELINE_AX  = "#c3c2b7"
BLUE         = "#2a78d6"   # categorical slot 1 / diverging "up" pole
ORANGE       = "#eb6834"   # categorical slot 2
RED          = "#e34948"   # diverging "down" pole

plt.rcParams.update({
    "font.family": "Segoe UI",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": BASELINE_AX,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


# ============================================================
# Chart 1 - paired-bootstrap forest plot
# ============================================================
# (label, mean diff, ci_low, ci_high, pct favoring, significant)
rows = [
    ("Population prior, sign-corrected\n(lam=5, vs. 36-feat baseline)",         0.068, -0.010, 0.149, 92.3, False),
    ("+ thermal features\n(LR)",                                                -0.025, -0.135, 0.087, 31.4, False),
    ("+ thermal features\n(GBT)",                                               -0.029, -0.117, 0.108, 26.8, False),
    ("Buffer 7s -> 3s\n(single-window baseline, GBT)",                          0.089, -0.050, 0.162, 94.8, False),
    ("+ clean-segment HRV\n(single-window baseline, LR)",                       0.043, 0.0045, 0.117, 98.6, True),
    ("+ HRV, multi-horizon set B\n(3s buffer, LR)",                             0.021, 0.0055, 0.041, 98.0, True),
    ("+ HRV, multi-horizon set D\n(3s buffer, LR)",                             0.023, 0.0048, 0.048, 100.0, True),
]

fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)
fig.subplots_adjust(top=0.82, left=0.28, right=0.97, bottom=0.14)

y = np.arange(len(rows))
for i, (label, mean, lo, hi, pct, sig) in enumerate(rows):
    color = BLUE if mean >= 0 else RED
    ax.plot([lo, hi], [i, i], color=color, lw=2.4 if sig else 1.6,
            alpha=1.0 if sig else 0.45, solid_capstyle="round", zorder=2)
    ax.scatter([mean], [i], s=90 if sig else 55, color=color,
               edgecolor=SURFACE, linewidth=2, zorder=3,
               alpha=1.0 if sig else 0.55)
    tag = "significant at 95%" if sig else "CI crosses zero"
    tag_color = INK_PRIMARY if sig else INK_MUTED
    weight = "bold" if sig else "normal"
    ax.text(hi + 0.012, i, f"{mean:+.3f}  ({tag})", va="center", ha="left",
            fontsize=9, color=tag_color, fontweight=weight)

ax.axvline(0, color=BASELINE_AX, lw=1.2, zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK_PRIMARY)
ax.set_xlabel("Paired bootstrap: mean AUC difference vs. that comparison's own baseline", fontsize=10)
ax.set_xlim(-0.16, 0.30)

fig.text(0.03, 0.955, "What actually moved AUC, and what didn't",
          fontsize=15, fontweight="bold", color=INK_PRIMARY, ha="left", va="top")
fig.text(0.03, 0.905,
          "Day-block paired bootstrap on each candidate's own comparison.\n"
          "Dot = mean difference, line = 95% CI. Only the two HRV additions exclude zero.",
          fontsize=9.5, color=INK_SECONDARY, ha="left", va="top", linespacing=1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.xaxis.grid(True, color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)

plt.savefig(OUT_DIR / "forest_plot.png", dpi=200)
plt.close(fig)


# ============================================================
# Chart 2 - population-prior lambda sweep
# ============================================================
lam = [1, 5, 20, 50, 100, 300, 1000, 5000]
original = [0.580, 0.541, 0.472, 0.454, 0.445, 0.444, 0.443, 0.446]
flipped  = [0.628, 0.651, 0.646, 0.631, 0.616, 0.607, 0.606, 0.608]
baseline_36feat = 0.571

fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=200)

ax.axhline(baseline_36feat, color=BASELINE_AX, lw=1.5, ls=(0, (4, 3)), zorder=1)
ax.text(lam[-1], baseline_36feat + 0.004, "personal baseline, no prior (0.571)",
        fontsize=9, color=INK_MUTED, ha="right", va="bottom")

ax.plot(lam, original, color=ORANGE, lw=2, marker="o", ms=6,
        markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
ax.plot(lam, flipped, color=BLUE, lw=2, marker="o", ms=6,
        markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)

ax.text(lam[-1] * 1.15, original[-1], "as directly fit\n(degrades)", fontsize=9.5,
        color=ORANGE, fontweight="bold", va="center")
ax.text(lam[-1] * 1.15, flipped[-1], "sign-corrected for\nmounting orientation\n(flat)", fontsize=9.5,
        color=BLUE, fontweight="bold", va="center")

ax.set_xscale("log")
ax.set_xlabel("lambda (shrinkage strength toward the population prior)", fontsize=10)
ax.set_ylabel("CV AUC", fontsize=10)
ax.set_title("Borrowing from a public dataset: raw prior vs. sign-corrected", fontsize=13.5,
             fontweight="bold", pad=14, loc="left", color=INK_PRIMARY)
ax.set_ylim(0.40, 0.70)
ax.set_xlim(0.8, 11000)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color=GRID, lw=1, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT_DIR / "lambda_sweep.png", dpi=200)
plt.close(fig)

print("Wrote:")
print(" ", OUT_DIR / "forest_plot.png")
print(" ", OUT_DIR / "lambda_sweep.png")
