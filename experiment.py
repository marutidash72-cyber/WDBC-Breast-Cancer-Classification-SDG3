"""
WDBC breast cancer classifier comparison
Leakage-free nested CV pipeline (supports SDG 3)

Run:  python wdbc_experiment.py            (full run, as in the paper: 5x10 outer folds)
      python wdbc_experiment.py --quick    (fast test run: 5x2 outer folds)

Outputs (folder ./results):
  table1_performance.csv      -> Table 1 (mean ± std over outer folds)
  table2_wilcoxon.csv         -> Table 2
  table3_time.csv             -> Table 3
  best_hyperparameters.csv    -> most frequently chosen grid values (for Sec 5.4)
  fig2_class_distribution.png, fig3_correlation.png, fig4_confusion_matrix.png,
  fig5_roc_curves.png, fig6_permutation_importance.png
  environment.txt             -> versions for Section 5.9
  summary.txt                 -> numbers/sentences to copy into Results text
  WDBC_filled.docx            -> your Word file with the three tables filled in
                                 (only if the .docx is found, see DOCX_PATH below)
"""

import argparse
import os
import platform
import sys
import time
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEED = 42
OUT = "results"
# Put your Word file next to this script (or change the path) to auto-fill tables
DOCX_PATH = "WDBC_Breast_Cancer_Classifiers_SDG3.docx"

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="5x2 outer folds for a fast test")
parser.add_argument("--jobs", type=int, default=-1, help="parallel workers for outer folds (-1 = all cores)")
args = parser.parse_args()

N_REPEATS = 2 if args.quick else 10
N_SPLITS = 5
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# Data  (malignant = 1 = positive class, benign = 0)
# ----------------------------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(1 - data.target, name="malignant")  # sklearn has malignant=0, so flip
print(f"Samples: {len(X)}, features: {X.shape[1]}")
print(f"Benign: {(y == 0).sum()}  Malignant: {(y == 1).sum()}")
print(f"Missing values: {int(X.isna().sum().sum())}, duplicate rows: {int(X.duplicated().sum())}")

# ----------------------------------------------------------------------------
# Models and (deliberately small) grids.  Scaler is INSIDE the pipeline -> no leakage
# ----------------------------------------------------------------------------
MODELS = {
    "Logistic regression": (
        LogisticRegression(max_iter=5000, random_state=SEED),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "k-NN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7, 9, 11]},
    ),
    "SVM (RBF)": (
        SVC(kernel="rbf", random_state=SEED),
        {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1]},
    ),
    "Random forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10]},
    ),
    "Gradient boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]},
    ),
}


def make_search(name):
    est, grid = MODELS[name]
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    return GridSearchCV(pipe, grid, scoring="f1", cv=inner, n_jobs=1, refit=True)


def scores_of(model, Xte):
    """Continuous score for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xte)[:, 1]
    return model.decision_function(Xte)


def run_fold(name, fold_id, tr, te):
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]

    search = make_search(name)
    t0 = time.perf_counter()
    search.fit(Xtr, ytr)                    # inner CV tuning + refit (training time)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    pred = search.predict(Xte)
    pred_time_ms = (time.perf_counter() - t0) / len(te) * 1000
    score = scores_of(search, Xte)

    tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
    return {
        "model": name, "fold": fold_id,
        "Accuracy": accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred, zero_division=0),
        "Recall": recall_score(yte, pred),
        "F1": f1_score(yte, pred),
        "ROC-AUC": roc_auc_score(yte, score),
        "MCC": matthews_corrcoef(yte, pred),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "fit_time_s": fit_time, "pred_time_ms": pred_time_ms,
        "best_params": str(search.best_params_),
        # keep out-of-fold scores for ROC curve (first repeat only)
        "te_idx": te if fold_id < N_SPLITS else None,
        "score": score if fold_id < N_SPLITS else None,
    }


# ----------------------------------------------------------------------------
# Nested CV
# ----------------------------------------------------------------------------
outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
splits = list(outer.split(X, y))
print(f"\nOuter folds: {len(splits)}  (inner: 3-fold grid search, F1 criterion)")

all_rows = []
for name in MODELS:
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.jobs)(
        delayed(run_fold)(name, i, tr, te) for i, (tr, te) in enumerate(splits)
    )
    all_rows.extend(rows)
    wall = time.perf_counter() - t0
    f1m = np.mean([r["F1"] for r in rows])
    print(f"  {name:<20s} done in {wall:7.1f}s wall-clock | mean F1 = {f1m:.4f}")

df = pd.DataFrame(all_rows)
df.drop(columns=["te_idx", "score"]).to_csv(f"{OUT}/per_fold_results.csv", index=False)
order = list(MODELS)

# ----------------------------------------------------------------------------
# Table 1: mean ± std
# ----------------------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MCC"]
mean = df.groupby("model")[metrics].mean().loc[order]
std = df.groupby("model")[metrics].std().loc[order]
table1 = pd.DataFrame(
    {m: [f"{mean.loc[n, m]:.4f} ± {std.loc[n, m]:.4f}" for n in order] for m in metrics},
    index=order,
)
table1.index.name = "Model"
table1.to_csv(f"{OUT}/table1_performance.csv")
print("\nTable 1 (mean ± std over outer folds)")
print(table1.to_string())

# ----------------------------------------------------------------------------
# Table 2: Wilcoxon signed-rank on per-fold F1 (best vs others) + Holm correction
# ----------------------------------------------------------------------------
best = mean["F1"].idxmax()
f1_by_model = {n: df[df.model == n].sort_values("fold")["F1"].values for n in order}
others = [n for n in mean["F1"].sort_values(ascending=False).index if n != best]

t2 = []
for n in others:
    d = f1_by_model[best] - f1_by_model[n]
    try:
        p = wilcoxon(f1_by_model[best], f1_by_model[n], zero_method="wilcox").pvalue
    except ValueError:      # all differences are zero
        p = 1.0
    t2.append({"Comparison": f"{best} vs. {n}", "Median difference in F1": np.median(d), "p-value": p})

# Holm correction
ps = np.array([r["p-value"] for r in t2])
idx = np.argsort(ps)
m = len(ps)
holm = np.empty(m)
running = 0.0
for rank, i in enumerate(idx):
    running = max(running, (m - rank) * ps[i])
    holm[i] = min(1.0, running)
for r, ph in zip(t2, holm):
    r["Holm-adjusted p"] = ph
    r["Significant at 0.05?"] = "Yes" if ph < 0.05 else "No"
table2 = pd.DataFrame(t2)
table2.to_csv(f"{OUT}/table2_wilcoxon.csv", index=False)
print(f"\nBest model by mean F1: {best}")
print("Table 2 (Wilcoxon, Holm-corrected)")
print(table2.to_string(index=False))

# ----------------------------------------------------------------------------
# Table 3: computational cost
# ----------------------------------------------------------------------------
tot_time = df.groupby("model")["fit_time_s"].sum().loc[order]       # summed over all outer folds
pred_ms = df.groupby("model")["pred_time_ms"].mean().loc[order]
table3 = pd.DataFrame({
    "Total nested-CV time (s)": tot_time.round(2),
    "Mean prediction time per sample (ms)": pred_ms.round(4),
})
table3.index.name = "Model"
table3.to_csv(f"{OUT}/table3_time.csv")
print("\nTable 3 (time summed across folds, single-core equivalent)")
print(table3.to_string())

# ----------------------------------------------------------------------------
# Selected hyperparameters (for the grid table in Sec 5.4)
# ----------------------------------------------------------------------------
hp = []
for n in order:
    c = Counter(df[df.model == n]["best_params"])
    for params, cnt in c.most_common(3):
        hp.append({"Model": n, "Chosen hyperparameters": params, "Times chosen": cnt})
pd.DataFrame(hp).to_csv(f"{OUT}/best_hyperparameters.csv", index=False)

# ----------------------------------------------------------------------------
# Error analysis: confusion matrix of the best model (summed over folds)
# ----------------------------------------------------------------------------
b = df[df.model == best]
cm = np.array([[b.TN.sum(), b.FP.sum()], [b.FN.sum(), b.TP.sum()]])
cm_avg = cm / N_REPEATS      # per full pass over the 569 samples
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]}\n({cm_avg[i, j]:.1f} per pass)", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Benign", "Malignant"]); ax.set_yticklabels(["Benign", "Malignant"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion matrix: {best}\n(summed over {len(splits)} outer folds)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_confusion_matrix.png", dpi=300); plt.close()

# ----------------------------------------------------------------------------
# Fig 5: ROC curves (out-of-fold predictions from the first repeat, all models)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
for n in order:
    sub = df[(df.model == n) & (df.fold < N_SPLITS)]
    idx_all = np.concatenate(sub["te_idx"].values)
    sc_all = np.concatenate(sub["score"].values)
    fpr, tpr, _ = roc_curve(y.iloc[idx_all], sc_all)
    ax.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y.iloc[idx_all], sc_all):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
ax.set_title("ROC curves (out-of-fold, first CV repeat)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig5_roc_curves.png", dpi=300); plt.close()

# ----------------------------------------------------------------------------
# Fig 2 and Fig 3: exploration
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4, 3.5))
counts = y.map({0: "Benign", 1: "Malignant"}).value_counts()[["Benign", "Malignant"]]
bars = ax.bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 4, f"{v} ({v / len(y) * 100:.1f}%)", ha="center")
ax.set_ylabel("Number of samples"); ax.set_title("Class distribution (WDBC)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_class_distribution.png", dpi=300); plt.close()

mean_cols = [c for c in X.columns if c.startswith("mean ")]
corr = X[mean_cols].corr()
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(mean_cols))); ax.set_yticks(range(len(mean_cols)))
ax.set_xticklabels([c.replace("mean ", "") for c in mean_cols], rotation=60, ha="right", fontsize=8)
ax.set_yticklabels([c.replace("mean ", "") for c in mean_cols], fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.046); ax.set_title("Correlation of the 10 mean features")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_correlation.png", dpi=300); plt.close()

# ----------------------------------------------------------------------------
# Fig 6: permutation importance on a held-out 20% test set (best model)
# ----------------------------------------------------------------------------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
final = make_search(best).fit(Xtr, ytr)
pi = permutation_importance(final, Xte, yte, scoring="f1", n_repeats=30, random_state=SEED, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values(ascending=False)
top = imp.head(10)[::-1]
err = pd.Series(pi.importances_std, index=X.columns)[top.index]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.barh(top.index, top.values, xerr=err.values, color="#4C72B0")
ax.set_xlabel("Drop in F1 when feature is permuted")
ax.set_title(f"Permutation importance, top 10 ({best}, 20% hold-out)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_permutation_importance.png", dpi=300); plt.close()
imp.head(10).to_csv(f"{OUT}/top10_permutation_importance.csv", header=["importance"])

# Logistic regression coefficients (Sec 5.8)
lr_final = make_search("Logistic regression").fit(X, y)
coef = pd.Series(lr_final.best_estimator_.named_steps["clf"].coef_[0], index=X.columns)
coef.reindex(coef.abs().sort_values(ascending=False).index).head(10).to_csv(
    f"{OUT}/logreg_top10_coefficients.csv", header=["standardised_coef"])

# ----------------------------------------------------------------------------
# Environment (Sec 5.9) and summary text
# ----------------------------------------------------------------------------
env = (
    f"OS: {platform.platform()}\nProcessor: {platform.processor()}\n"
    f"CPU cores: {os.cpu_count()}\nPython: {sys.version.split()[0]}\n"
    f"scikit-learn: {sklearn.__version__}\nNumPy: {np.__version__}\n"
    f"pandas: {pd.__version__}\nSciPy: {scipy.__version__}\n"
    f"Matplotlib: {matplotlib.__version__}\nSeed: {SEED}\n"
    "(Add RAM manually, e.g. 16 GB.)\n"
)
open(f"{OUT}/environment.txt", "w").write(env)

total_min = df["fit_time_s"].sum() / 60
rank = mean["F1"].sort_values(ascending=False)
lines = [
    f"Best model by mean F1: {best}",
    f"  Accuracy {mean.loc[best, 'Accuracy'] * 100:.2f}% | Recall {mean.loc[best, 'Recall']:.4f} | "
    f"MCC {mean.loc[best, 'MCC']:.4f} | ROC-AUC {mean.loc[best, 'ROC-AUC']:.4f}",
    f"  Confusion matrix summed over {len(splits)} folds: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}",
    f"  Average per full pass over 569 samples: FN={cm_avg[1,0]:.1f}, FP={cm_avg[0,1]:.1f}",
    "",
    "Best model per metric: " + "; ".join(f"{m}: {mean[m].idxmax()} ({mean[m].max():.4f})" for m in metrics),
    f"Gap between 1st and 2nd (F1): {rank.iloc[0] - rank.iloc[1]:.4f}",
    "",
    f"Total single-core compute for all nested CV: {total_min:.1f} min",
    "Top 10 permutation-importance features:",
    *[f"  {i + 1}. {n} ({v:.4f})" for i, (n, v) in enumerate(imp.head(10).items())],
]
open(f"{OUT}/summary.txt", "w").write("\n".join(lines))
print("\n" + "\n".join(lines))

# ----------------------------------------------------------------------------
# Fill the three tables in the Word file (needs: pip install python-docx)
# ----------------------------------------------------------------------------
def fill_docx():
    try:
        from docx import Document
    except ImportError:
        print("\npython-docx not installed -> skipped Word filling (pip install python-docx)")
        return
    if not os.path.exists(DOCX_PATH):
        print(f"\n{DOCX_PATH} not found next to the script -> skipped Word filling")
        return

    def set_text(cell, text):
        p = cell.paragraphs[0]
        if p.runs:
            p.runs[0].text = text
            for r in p.runs[1:]:
                r.text = ""
        else:
            p.add_run(text)

    doc = Document(DOCX_PATH)
    for t in doc.tables:
        header = [c.text.strip() for c in t.rows[0].cells]
        if header[:2] == ["Model", "Accuracy"]:                       # Table 1
            for r in t.rows[1:]:
                n = r.cells[0].text.strip()
                if n in table1.index:
                    for j, mname in enumerate(metrics, start=1):
                        set_text(r.cells[j], table1.loc[n, mname])
        elif header[0] == "Comparison":                              # Table 2
            for r, row in zip(t.rows[1:], t2):
                set_text(r.cells[0], row["Comparison"])
                set_text(r.cells[1], f"{row['Median difference in F1']:.4f}")
                set_text(r.cells[2], f"{row['Holm-adjusted p']:.4f}" if row['Holm-adjusted p'] >= 0.0001 else "<0.0001")
                set_text(r.cells[3], row["Significant at 0.05?"])
        elif header[:2] == ["Model", "Total nested-CV time (s)"]:    # Table 3
            for r in t.rows[1:]:
                n = r.cells[0].text.strip()
                if n in table3.index:
                    set_text(r.cells[1], f"{table3.loc[n].iloc[0]:.2f}")
                    set_text(r.cells[2], f"{table3.loc[n].iloc[1]:.4f}")
    out = f"{OUT}/WDBC_filled.docx"
    doc.save(out)
    print(f"\nWord file with filled tables saved to {out}")


fill_docx()
print(f"\nAll outputs are in the '{OUT}' folder.")