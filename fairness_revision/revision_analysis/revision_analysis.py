"""
Revision analysis for "Topology or Demography?" (SNAM major revision).

Three subcommands, matching the three code-side reviewer asks:
    rebuild   -- item A: rebuild the hybrid so it fuses the real phone-network
                 battery with the demography features, and test (paired bootstrap)
                 whether it changes results vs the demography-only hybrid.
    stats     -- item B (R2.3/R2.8): per-approach F1 + precision + recall, class
                 balance, dummy baselines, bootstrap 95% CIs, cross-approach McNemar
                 significance and effect sizes.
    shap      -- item C (R2.6): SHAP + permutation importance + an interpretable
                 depth-3 decision tree; writes feature_importance.csv for plotting.

Usage:
    python revision_analysis.py rebuild
    python revision_analysis.py stats
    python revision_analysis.py shap
    python revision_analysis.py all

Data lives in the Code Ocean capsule. Set CODE_DIR / DATA_DIR below (or as env vars)
to the capsule's code/ and data/ directories.
"""
import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score
from sklearn.dummy import DummyClassifier
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

# --------------------------------- CONFIG ---------------------------------
# Locations of the Code Ocean capsule's code (the "demography_based_approach"
# folder) and data (the "data_zip" folder). These are auto-detected relative to
# this script; if your layout differs and detection fails, set the CODE_DIR and
# DATA_DIR environment variables. On Windows, e.g.:
#     set CODE_DIR=C:\path\to\capsule\code\demography_based_approach
#     set DATA_DIR=C:\path\to\capsule\data\data_zip
_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent


def _first_existing(cands, marker):
    for c in cands:
        if c and (c / marker).exists():
            return c
    return None


def _glob_base(root, marker):
    if root == root.parent:  # never scan a filesystem / drive root
        return None
    for hit in root.glob("**/" + Path(marker).name):
        base = hit
        for _ in Path(marker).parts:
            base = base.parent
        if (base / marker).exists():
            return base
    return None


_CODE_MARKER = "survey_and_hybrid_approach_pipeline/pipeline_lib.py"
_env = os.environ.get("CODE_DIR")
CODE_DIR = Path(_env) if _env else (_first_existing(
    [_HERE.parent / "demography_based_approach",
     _HERE / "demography_based_approach",
     _HERE.parent.parent / "demography_based_approach"], _CODE_MARKER)
                                    or _glob_base(_PROJECT, _CODE_MARKER))
if not CODE_DIR or not (CODE_DIR / _CODE_MARKER).exists():
    sys.exit("ERROR: could not locate the capsule code (survey_and_hybrid_approach_pipeline/"
             "pipeline_lib.py).\nSet the CODE_DIR environment variable to the capsule's "
             "demography_based_approach folder.")

_DATA_MARKER = "dicts/dict_of_training_dfs.pkl"
_env = os.environ.get("DATA_DIR")
DATA_DIR = Path(_env) if _env else (_first_existing(
    [CODE_DIR.parent.parent / "data_zip", CODE_DIR.parent / "data_zip", CODE_DIR / "data_zip",
     _HERE.parent / "data_zip", _HERE.parent.parent / "data_zip"], _DATA_MARKER)
                                    or _glob_base(_PROJECT, _DATA_MARKER) or _glob_base(CODE_DIR.parent, _DATA_MARKER))
if not DATA_DIR or not (DATA_DIR / _DATA_MARKER).exists():
    sys.exit("ERROR: could not locate the capsule data (dicts/dict_of_training_dfs.pkl).\n"
             "Set the DATA_DIR environment variable to the capsule's data_zip folder.")

OUT_DIR = Path(os.environ.get("OUT_DIR", str(_HERE / "revision_outputs")))
BAYES, CV = int(os.environ.get("BAYES_ITERS", 30)), int(os.environ.get("CV_SPLITS", 10))
# --------------------------------------------------------------------------
PIPE = CODE_DIR / "survey_and_hybrid_approach_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the capsule's pipeline_lib.py DIRECTLY from its file path. This is
# deliberate: importing it by name (`import pipeline_lib`) can silently pick up
# an unrelated 'pipeline_lib' package installed in the environment, which
# hard-imports a Unix-only multiprocessing symbol and crashes on Windows.
import importlib.util as _ilu

sys.path.insert(0, str(PIPE))  # so pipeline_lib can find any sibling module
_spec = _ilu.spec_from_file_location("capsule_pipeline_lib", str(PIPE / "pipeline_lib.py"))
PL = _ilu.module_from_spec(_spec)
sys.modules[_spec.name] = PL  # register before exec so @dataclass / type hints resolve
_spec.loader.exec_module(PL)



TOPICS = ["euthanasia", "fssocsec", "fswelfare", "jobguar", "marijuana", "toomucheqrights"]
# phone-network battery (topology approach) + survey-network centralities
PHONE = ['Node degree', 'Sum of CogsNet', "Avg Neighbour's CogsNetSum", 'Avg Neighbour Degree',
         "Avg Neighbour's InteractionSum", 'Degree-centrality', 'Betweenness-centrality', 'Pagerank',
         'Eigenvector centrality', 'Closeness centrality', 'Current flow closeness centrality',
         'Information centrality', 'Load', 'Subgraph centrality exp', 'Laplacian']
SURVEY_NET = ['Degree Centrality', 'Closeness Centrality', 'Betweenness Centrality',
              'Eigenvector Centrality', 'Community']
TOPO = set(PHONE + SURVEY_NET)


def family(f): return "topological" if f in TOPO else "demographic"


_rng = np.random.default_rng(42)


def _cfg(out):
    c = {"base_dir": str(CODE_DIR), "demog_features_path": str(PIPE / "all_features_plus_neighbours.csv"),
         "pure_demog_path": str(PIPE / "pure_demographical_data.csv"),
         "topology_features_path": str(CODE_DIR / "netsense/topology_based_features.csv"),
         "simulation_results_dir": str(CODE_DIR / "simulation_result_per_topic/best_sim_result"),
         "minorities_pickle_path": str(DATA_DIR / "dicts/dictionary_of_dfs_with_minorities.pkl"),
         "models_output_dir": str(out), "evaluation_output_dir": str(out), "artifacts_output_dir": str(out),
         "topics": TOPICS, "test_size": 0.2, "random_seed": 42,
         "bayes_search_iterations": BAYES, "cv_splits": CV,
         "top_feature_counts": [15, 20, 30, 40, 50, 60, 70, 85]}
    p = out / "_cfg.json";
    out.mkdir(parents=True, exist_ok=True);
    p.write_text(json.dumps(c))
    return PL.load_config(p)


def _phone_battery():
    tr = pickle.load(open(DATA_DIR / "dicts/dict_of_training_dfs.pkl", "rb"))
    ph = pd.concat(tr.values(), ignore_index=True).rename(columns={"EgoID": "egoid", "Semester": "SurveyNr"})
    ph["egoid"] = ph["egoid"].astype("int64");
    ph["SurveyNr"] = ph["SurveyNr"].astype("int64")
    return ph


def load_data(cfg):
    """demography table (duplicate centrality columns collapsed) + phone battery, fused."""
    demog = PL.load_feature_tables(cfg)
    demog = demog.drop(columns=[c for c in demog.columns if c.endswith(("_Survey", "_x", "_y"))])  # de-dupe bug fix
    demog_only = demog.copy()
    fused = demog.merge(_phone_battery(), how="left", on=["egoid", "SurveyNr"])
    data = PL.load_simulation_results(cfg);
    mino = PL.load_minorities(cfg.minorities_pickle_path)
    return demog_only, fused, data, mino


# ---- metric helpers ----
def bin_f1(y, p):
    y = np.asarray(y);
    p = np.asarray(p)
    tp = np.sum((p == 1) & (y == 1));
    fp = np.sum((p == 1) & (y == 0));
    fn = np.sum((p == 0) & (y == 1))
    d = 2 * tp + fp + fn;
    return (2 * tp / d) if d > 0 else 0.0


def macro_f1(y, p):
    y = np.asarray(y);
    p = np.asarray(p);
    s = 0.0
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c));
        fp = np.sum((p == c) & (y != c));
        fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn;
        s += (2 * tp / d) if d > 0 else 0.0
    return s / 2


def ci(vals): return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def boot_ci(y, p, metric=bin_f1, B=1500):
    y = np.asarray(y);
    p = np.asarray(p);
    k = len(y)
    if k == 0: return (float("nan"), float("nan"))
    o = np.array([metric(y[s], p[s]) for s in (_rng.integers(0, k, k) for _ in range(B))]);
    return ci(o)


def paired(y, pa, pb, metric=bin_f1, B=1500):
    """McNemar (pb vs pa) + paired bootstrap CI of metric(pb)-metric(pa)."""
    y = np.asarray(y);
    pa = np.asarray(pa);
    pb = np.asarray(pb)
    aok = (pa == y);
    bok = (pb == y);
    B_ = int(np.sum(aok & ~bok));
    C_ = int(np.sum(~aok & bok));
    n = B_ + C_
    pval = float(binomtest(min(B_, C_), n, 0.5).pvalue) if n > 0 else 1.0
    k = len(y);
    D = np.array([metric(y[s], pb[s]) - metric(y[s], pa[s]) for s in (_rng.integers(0, k, k) for _ in range(B))])
    return dict(p=pval, b=B_, c=C_, delta=metric(y, pb) - metric(y, pa), dci=ci(D))


def rf():  # common classifier for the approach comparison; isolates the feature family
    return RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)


# ========================= item A: rebuild =========================
def cmd_rebuild():
    cfgD, cfgH = _cfg(OUT_DIR / "demography"), _cfg(OUT_DIR / "hybrid")
    demog, fused, data, mino = load_data(cfgD)
    print("demography feature matrix:", demog.shape, "| honest hybrid (+phone):", fused.shape)
    rD = PL.train_and_evaluate(cfgD, demog, data, mino)  # demography-only hybrid (baseline)
    rH = PL.train_and_evaluate(cfgH, fused, data, mino)  # honest hybrid (+ phone battery)
    print("\ntopic             demog  hybrid   dF1 [95% CI]        McNemar p")
    for t in TOPICS:
        Xtr, Xte, ytr, yte, _, _ = PL.build_topic_dataset(t, fused, data[t], mino, cfgH)
        fD = rD[t]["best_features_subset"];
        fH = rH[t]["best_features_subset"]
        mD = joblib.load(OUT_DIR / f"demography/{t}_best_model_for_selected_features.joblib")
        mH = joblib.load(OUT_DIR / f"hybrid/{t}_best_model_for_selected_features.joblib")
        y = np.asarray(yte);
        pr = paired(y, mD.predict(Xte[fD]), mH.predict(Xte[fH]))
        sig = " *" if (pr["dci"][0] > 0 or pr["dci"][1] < 0) else ""
        print("%-16s %.3f  %.3f  %+.3f [%+.2f,%+.2f]   %.3f%s"
              % (t, rD[t]["best_f1"], rH[t]["best_f1"], pr["delta"], *pr["dci"], pr["p"], sig))
    print("\n( * = 95% bootstrap CI on the difference excludes 0 )")


# ========================= item B: stats =========================
def cmd_stats():
    cfg = _cfg(OUT_DIR / "stats");
    _, fused, data, mino = load_data(cfg)
    out = {}
    for t in TOPICS:
        Xtr, Xte, ytr, yte, _, _ = PL.build_topic_dataset(t, fused, data[t], mino, cfg)
        y = np.asarray(yte);
        n = len(y);
        bal = float(np.mean(y))
        fT = [c for c in PHONE if c in Xte.columns];
        fD = [c for c in Xte.columns if c not in fT];
        fH = list(Xte.columns)
        pD = rf().fit(Xtr[fD], ytr).predict(Xte[fD]);
        pT = rf().fit(Xtr[fT], ytr).predict(Xte[fT]);
        pH = rf().fit(Xtr[fH], ytr).predict(Xte[fH])
        dmf = DummyClassifier(strategy="most_frequent").fit(Xtr[fH], ytr).predict(Xte[fH])
        print("### %s  (n_test=%d, P(mispred)=%.2f, dummy-most-freq F1=%.3f)" % (t, n, bal, bin_f1(y, dmf)))
        for lab, p in [("demography", pD), ("topology", pT), ("hybrid", pH)]:
            print("   %-11s F1=%.3f %s  precision=%.2f  recall=%.2f"
                  % (lab, bin_f1(y, p), "[%.2f,%.2f]" % boot_ci(y, p),
                     precision_score(y, p, pos_label=1, zero_division=0),
                     recall_score(y, p, pos_label=1, zero_division=0)))
        for lab, a, b in [("hybrid vs demography", pD, pH), ("hybrid vs topology", pT, pH),
                          ("topology vs demography", pD, pT)]:
            r = paired(y, a, b);
            print("     %-24s dF1=%+.3f [%+.2f,%+.2f]  McNemar p=%.3f (b=%d,c=%d)%s"
                  % (lab, r["delta"], *r["dci"], r["p"], r["b"], r["c"], "  *SIG" if r["p"] < 0.05 else ""))
        out[t] = dict(n=n, balance=bal)
    json.dump(out, open(OUT_DIR / "stats_summary.json", "w"), indent=2)


# ========================= item C: explainability =========================
def cmd_shap():
    import shap
    cfg = _cfg(OUT_DIR / "shap");
    _, fused, data, mino = load_data(cfg)
    agg = {};
    topo_share = {};
    nfeat_topo = nfeat = 0;
    dt_rules = None
    for t in TOPICS:
        Xtr, Xte, ytr, yte, _, _ = PL.build_topic_dataset(t, fused, data[t], mino, cfg)
        feats = list(Xte.columns);
        nfeat = len(feats);
        nfeat_topo = sum(f in TOPO for f in feats)
        m = rf().fit(Xtr[feats], ytr)
        sv = shap.TreeExplainer(m).shap_values(Xte[feats], check_additivity=False)
        if isinstance(sv, list): sv = sv[1]
        sv = np.asarray(sv);
        sv = sv[:, :, 1] if sv.ndim == 3 else sv
        s = pd.Series(np.abs(sv).mean(0), index=feats);
        s = s / s.sum()
        topo_share[t] = float(s[[f for f in feats if f in TOPO]].sum())
        for f in feats: agg[f] = agg.get(f, 0) + s[f] / len(TOPICS)
        print("%-16s topological SHAP share = %.1f%%" % (t, 100 * topo_share[t]))
        if t == "euthanasia":
            dt_rules = export_text(
                DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(Xtr[feats], ytr),
                feature_names=list(feats))
    S = pd.Series(agg).sort_values(ascending=False)
    overall = sum(v for f, v in agg.items() if f in TOPO)
    print("\nTopological features: %d/%d (%.0f%% of set) | SHAP attribution share: %.1f%%" % (nfeat_topo, nfeat,
                                                                                              100 * nfeat_topo / nfeat,
                                                                                              100 * overall))
    df = pd.DataFrame({"feature": S.index, "shap_share": S.values, "family": [family(f) for f in S.index]})
    df.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    (OUT_DIR / "decision_tree_euthanasia.txt").write_text(dt_rules or "")
    print("wrote", OUT_DIR / "feature_importance.csv", "and decision_tree_euthanasia.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser();
    ap.add_argument("cmd", choices=["rebuild", "stats", "shap", "all"])
    c = ap.parse_args().cmd
    if c in ("rebuild", "all"): cmd_rebuild()
    if c in ("stats", "all"): cmd_stats()
    if c in ("shap", "all"): cmd_shap()
