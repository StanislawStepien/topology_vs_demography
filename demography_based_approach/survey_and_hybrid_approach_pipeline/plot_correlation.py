"""
Trimmed minority-correlation figure (revised Fig. 1).

Rebuilds the minority correlation matrix as a clean 7x7 heatmap of the pairwise
(phi) correlation between *membership* in each minority group - dropping the node
degree, question number, and model-performance entries of the original, which the
text never used (Reviewer 2's relevance concern). `gaymarriage_6` is excluded, so
this shows the seven groups the paper actually defines.

Correlation for binary membership is the phi coefficient, i.e. the Pearson
correlation of the 0/1 membership vectors over the full student population.

Usage:
    python plot_correlation.py                         # capsule defaults
    python plot_correlation.py minorities.pkl feats.csv out.png
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # drop for an interactive backend
import matplotlib.pyplot as plt

# ------------------------------- STYLE KNOBS -------------------------------
# order + display labels (edit labels freely; keys must match the pickle)
GROUPS = [
    ("gender_1",                "Gender"),
    ("ethnicity_1",             "Ethnicity"),
    ("fbprivacy_1",             "FB privacy"),
    ("engnative_1",             "English native"),
    ("pincome_1",               "Parents' income"),
    ("momed_1, daded_1",        "Parents' education"),
    ("momrelig_1, dadrelig_1",  "Parents' religion"),
]
FIGSIZE   = (7.2, 6.0)
DPI       = 200
CMAP      = "RdBu_r"     # diverging, centred at 0
VMIN, VMAX = -1.0, 1.0
ANNOT     = True         # write the coefficient in each cell
ANNOT_FMT = "%.2f"
ANNOT_SIZE = 9
LABEL_SIZE = 10
TITLE      = ""          # set a string to add a title; caption carries it in the paper
GRID_COLOR = "white"


def _find_id_col(df):
    for c in df.columns:
        if c.lower() in ("egoid", "studentid", "student_id", "id"):
            return c
    return df.columns[0]


def main(pkl=None, feats=None, out=None):
    base = Path("survey_and_hybrid_pipeline.ipynb").resolve().parent
    pkl   = Path(pkl   or base / "dictionary_of_dfs_with_minorities.pkl")
    feats = Path(feats or base / "all_features_plus_neighbours.csv")
    out   = Path(out   or "revision_outputs/correlation_matrix_trimmed.png")

    d = pickle.load(open(pkl, "rb"))
    # full student population from the feature table
    fdf = pd.read_csv(feats, sep=None, engine="python")
    idc = _find_id_col(fdf)
    pop = pd.Index(sorted(fdf[idc].dropna().astype(int).unique()), name="id")

    # binary membership matrix: population x groups
    M = pd.DataFrame(index=pop)
    for key, label in GROUPS:
        members = set(int(x) for x in d[key].values)
        M[label] = pop.isin(members).astype(int)

    corr = M.corr()                          # phi = Pearson on 0/1
    labels = [lbl for _, lbl in GROUPS]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(corr.values, cmap=CMAP, vmin=VMIN, vmax=VMAX)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=LABEL_SIZE)
    ax.set_yticklabels(labels, fontsize=LABEL_SIZE)
    # thin grid between cells
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    if ANNOT:
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = corr.values[i, j]
                ax.text(j, i, ANNOT_FMT % v, ha="center", va="center",
                        fontsize=ANNOT_SIZE,
                        color="white" if abs(v) > 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (phi)", fontsize=LABEL_SIZE)
    if TITLE:
        ax.set_title(TITLE, fontsize=LABEL_SIZE + 2)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print("population size:", len(pop))
    print("group sizes:", {lbl: int(M[lbl].sum()) for lbl in labels})
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])