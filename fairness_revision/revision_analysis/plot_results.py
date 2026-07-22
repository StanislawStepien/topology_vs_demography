"""
Results figure for the revision (Fig. 2).

Reads general_pop_metrics.csv (written by `revision_analysis.py stats`) and draws a
grouped bar chart of misprediction-class F1 for the three feature sets on each
question, with bootstrap 95% CI error bars and the most-frequent-class baseline
marked. Kept separate from the analysis so the figure can be restyled without
re-running the models.

Usage:
    python plot_results.py                                 # ./revision_outputs/stats/general_pop_metrics.csv
    python plot_results.py path/to/metrics.csv out.png
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # drop this line for interactive backends
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ----------------------------- STYLE KNOBS -----------------------------
QORDER = ["euthanasia", "fssocsec", "fswelfare", "jobguar", "marijuana", "toomucheqrights"]
QLABEL = {"euthanasia": "Euthanasia", "fssocsec": "Social\nsecurity", "fswelfare": "Welfare",
          "jobguar": "Job\nguarantee", "marijuana": "Marijuana", "toomucheqrights": "Equal\nrights"}
AORDER = ["demography", "topology", "hybrid"]
ALABEL = {"demography": "Demography", "topology": "Topology", "hybrid": "Hybrid"}
COLORS = {"demography": "#2563eb", "topology": "#d97706", "hybrid": "#16a34a"}
FIGSIZE = (9, 5)
DPI = 150
BAR_W = 0.26
TITLE = "Predicting CoDiNG misclassification: F1 by feature set (95% CI)"
YLABEL = "F1 (misprediction class)"


def main(src=None, out=None):
    src = Path(src or "revision_outputs/stats/general_pop_metrics.csv")
    out = Path(out or "revision_outputs/results_ci.png")
    df = pd.read_csv(src)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = list(range(len(QORDER)))
    for j, a in enumerate(AORDER):
        xs = [i + (j - 1) * BAR_W for i in x]
        sub = {q: df[(df.question == q) & (df.approach == a)] for q in QORDER}
        f1 = [sub[q].f1.values[0] for q in QORDER]
        lo = [sub[q].ci_low.values[0] for q in QORDER]
        hi = [sub[q].ci_high.values[0] for q in QORDER]
        yerr = [[f - l for f, l in zip(f1, lo)], [h - f for f, h in zip(f1, hi)]]
        ax.bar(xs, f1, width=BAR_W, color=COLORS[a], label=ALABEL[a], zorder=2)
        ax.errorbar(xs, f1, yerr=yerr, fmt="none", ecolor="#333333", elinewidth=1, capsize=2.5, zorder=3)

    # most-frequent-class baseline: short dashed line spanning each question's group
    for i, q in enumerate(QORDER):
        d = df[df.question == q].dummy_f1.values[0]
        ax.plot([i - 1.6 * BAR_W, i + 1.6 * BAR_W], [d, d], color="black", lw=1.4,
                ls=(0, (4, 2)), zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([QLABEL[q] for q in QORDER])
    ax.set_ylabel(YLABEL)
    ax.set_ylim(0, 1.0)
    ax.set_title(TITLE)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [Patch(facecolor=COLORS[a], label=ALABEL[a]) for a in AORDER] + \
              [Line2D([0], [0], color="black", lw=1.4, ls=(0, (4, 2)), label="Most-frequent baseline")]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:3])