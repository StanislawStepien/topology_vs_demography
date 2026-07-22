"""
Feature-importance figure for the revision (item C).

Reads feature_importance.csv (written by `revision_analysis.py shap`) and draws a
horizontal bar chart of the top-N features, coloured by family (demographic vs
topological). Kept separate from the analysis so you can restyle freely without
re-running SHAP.

Usage:
    python plot_feature_importance.py                      # uses ./revision_outputs/feature_importance.csv
    python plot_feature_importance.py path/to/importance.csv out.png
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # drop this line for interactive backends
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------- STYLE KNOBS -----------------------------
TOP_N       = 15
COLORS      = {"demographic": "#2563eb", "topological": "#d97706"}
FIGSIZE     = (8.5, 6)
DPI         = 150
LABEL_FONT  = 8
TITLE       = "What drives predicted CoDiNG misclassification (hybrid model)"
XLABEL      = "Mean |SHAP| (share of attribution, averaged across the 6 questions)"
LABEL_TRUNC = 40                          # truncate long feature names to this many chars
BAR_HEIGHT  = 0.8
# -----------------------------------------------------------------------

def main(csv_path, out_path):
    df = pd.read_csv(csv_path).head(TOP_N).iloc[::-1]      # top-N, ascending for barh
    colors = [COLORS.get(fam, "#888888") for fam in df["family"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.barh(range(len(df)), df["shap_share"], color=colors, height=BAR_HEIGHT)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f[:LABEL_TRUNC] for f in df["feature"]], fontsize=LABEL_FONT)
    ax.set_xlabel(XLABEL)
    ax.set_title(TITLE)
    ax.legend(handles=[Patch(color=c, label=k.capitalize()) for k, c in COLORS.items()],
              loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    print("saved", out_path)

if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("revision_outputs/feature_importance.csv")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("feature_importance.png")
    main(csv, out)
