"""
WDBC Breast Cancer Classification — reproducible experiment
Run: python experiment.py
The dataset is loaded from scikit-learn; no manual CSV download is required.
Malignant is encoded as the positive class (1), benign as 0.
"""
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_curve,
    auc
)
from sklearn.inspection import permutation_importance
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
RANDOM_STATE = 42

def build_models():
    return {
        "Logistic Regression": (
            Pipeline([("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))]),
            {"model__C": [0.1, 1, 10]}
        ),
        "K-NN": (
            Pipeline([("scale", StandardScaler()),
                      ("model", KNeighborsClassifier())]),
            {"model__n_neighbors": [3, 5, 7, 9]}
        ),
        "SVM (RBF)": (
            Pipeline([("scale", StandardScaler()),
                      ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))]),
            {"model__C": [0.1, 1, 10], "model__gamma": ["scale", "auto"]}
        ),
        "Random Forest": (
            Pipeline([("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10]}
        ),
        "Gradient Boosting": (
            Pipeline([("model", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            {"model__n_estimators": [100, 150], "model__learning_rate": [0.05, 0.1]}
        )
    }

def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    # sklearn's original target: 0=malignant, 1=benign. Convert malignant to positive=1.
    y = pd.Series((data.target == 0).astype(int), name="target")

    print(f"Samples: {len(X)}, features: {X.shape[1]}")
    print(f"Benign: {(y == 0).sum()}, Malignant: {(y == 1).sum()}")
    print(f"Missing values: {int(X.isna().sum().sum())}, duplicate rows: {int(X.duplicated().sum())}")

    # Save the exact dataset used, with malignant=1 and benign=0.
    export = X.copy()
    export.insert(0, "diagnosis", y.map({0: "benign", 1: "malignant"}))
    export.to_csv(ROOT / "dataset" / "wdbc_used.csv", index=False)

    outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    model_specs = build_models()
    fold_rows, fold_predictions, fitted_models = [], {}, {}
    start_all = time.perf_counter()

    for name, (pipeline, grid) in model_specs.items():
        start = time.perf_counter()
        fold_predictions[name] = []
        for fold_no, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
            search = GridSearchCV(pipeline, grid, scoring="f1", cv=inner, n_jobs=-1, refit=True)
            search.fit(X.iloc[train_idx], y.iloc[train_idx])
            best = search.best_estimator_
            pred = best.predict(X.iloc[test_idx])
            prob = best.predict_proba(X.iloc[test_idx])[:, 1]
            yt = y.iloc[test_idx]
            metrics = {
                "model": name, "fold": fold_no,
                "accuracy": accuracy_score(yt, pred),
                "precision": precision_score(yt, pred, zero_division=0),
                "recall": recall_score(yt, pred, zero_division=0),
                "f1": f1_score(yt, pred, zero_division=0),
                "roc_auc": roc_auc_score(yt, prob),
                "mcc": matthews_corrcoef(yt, pred)
            }
            fold_rows.append(metrics)
            fold_predictions[name].append((yt.to_numpy(), pred, prob))
            if fold_no == 50:
                fitted_models[name] = best
        print(f"{name}: completed in {time.perf_counter()-start:.1f}s")

    total_seconds = time.perf_counter() - start_all
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "fold_metrics.csv", index=False)
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    summary = folds.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(OUT / "model_comparison.csv", index=False)

    # Plot mean accuracy and F1.
    means = folds.groupby("model")[["accuracy", "f1"]].mean().sort_values("f1", ascending=False)
    ax = means.plot(kind="bar", figsize=(10, 6), ylim=(0, 1), rot=20)
    ax.set_title("Classifier comparison (mean over 50 outer folds)")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=.25)
    plt.tight_layout()
    plt.savefig(OUT / "accuracy_f1_comparison.png", dpi=200)
    plt.close()

    # Confusion matrix from the final outer fold's held-out predictions (one 5-fold test split).
    best_name = folds.groupby("model")["f1"].mean().idxmax()
    yt, pred, prob = fold_predictions[best_name][-1]
    cm = confusion_matrix(yt, pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    plt.title(f"Confusion matrix — {best_name} (one held-out fold)")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=200)
    plt.close()

    # ROC curve for the same held-out fold.
    fpr, tpr, _ = roc_curve(yt, prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Chance")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve — final held-out fold")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=200)
    plt.close()

    # Refit best model on all available data for a descriptive feature-importance analysis.
    pipe, grid = model_specs[best_name]
    final_search = GridSearchCV(pipe, grid, scoring="f1", cv=inner, n_jobs=-1)
    final_search.fit(X, y)
    perm = permutation_importance(final_search.best_estimator_, X, y,
                                  scoring="f1", n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    imp = pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    top = imp.head(10).sort_values("importance")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Mean decrease in F1 after permutation")
    plt.title(f"Permutation feature importance — {best_name}")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=200)
    plt.close()

    # Pairwise Wilcoxon tests on fold F1; Holm correction.
    best_scores = folds[folds.model == best_name].sort_values("fold")["f1"].to_numpy()
    tests = []
    raw_p = []
    for name in model_specs:
        if name == best_name:
            continue
        other = folds[folds.model == name].sort_values("fold")["f1"].to_numpy()
        stat, p = wilcoxon(best_scores, other, zero_method="wilcox", alternative="two-sided")
        tests.append({"comparison": f"{best_name} vs {name}",
                      "median_f1_difference": float(np.median(best_scores-other)),
                      "p_value": float(p)})
        raw_p.append(p)
    # Holm-Bonferroni adjusted p-values.
    order = np.argsort(raw_p)
    adjusted = np.empty(len(raw_p))
    running = 0.0
    m = len(raw_p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank)*raw_p[idx])
        running = max(running, val)
        adjusted[idx] = running
    for i, row in enumerate(tests):
        row["holm_adjusted_p"] = float(adjusted[i])
        row["significant_at_0_05"] = bool(adjusted[i] < 0.05)
    pd.DataFrame(tests).to_csv(OUT / "wilcoxon_comparison.csv", index=False)

    # Classification report for the selected model's final held-out fold.
    (OUT / "classification_report.txt").write_text(
        f"Model: {best_name}\n" +
        classification_report(yt, pred, target_names=["Benign", "Malignant"], zero_division=0),
        encoding="utf-8"
    )
    (OUT / "run_summary.txt").write_text(
        f"Best model by mean F1: {best_name}\n"
        f"Total nested cross-validation runtime: {total_seconds:.2f} seconds\n"
        f"Outer folds: 50; inner CV: 3 folds; random state: {RANDOM_STATE}\n"
        "Confusion matrix and classification report are for the final held-out outer fold only.\n",
        encoding="utf-8"
    )
    print("\nMean ± standard deviation over outer folds:")
    print(summary.to_string(index=False))
    print(f"\nBest model by mean F1: {best_name}")
    print(f"Total runtime: {total_seconds:.1f} seconds")
    print(f"Saved all result files in: {OUT}")

if __name__ == "__main__":
    main()
