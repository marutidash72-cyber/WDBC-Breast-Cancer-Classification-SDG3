# WDBC Breast Cancer Classification — SDG 3

A reproducible machine-learning comparison using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Models
- Logistic Regression
- K-Nearest Neighbours (K-NN)
- Support Vector Machine with RBF kernel
- Random Forest
- Gradient Boosting

## Dataset
The code loads the 569-row, 30-feature Wisconsin Diagnostic Breast Cancer dataset directly from `scikit-learn`. It exports the exact data used to `dataset/wdbc_used.csv`. The target is encoded as **malignant = 1 (positive)** and **benign = 0**.

## Run locally (Windows)
```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py experiment.py
```

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py
```

The experiment uses repeated stratified 5-fold outer cross-validation (10 repeats = 50 outer folds), with 3-fold grid search inside each training fold. Scaling is included in the relevant model pipelines to avoid preprocessing leakage.

## Generated outputs
After running `experiment.py`, the `results/` folder will contain:
- `model_comparison.csv` — mean and standard deviation of six metrics
- `fold_metrics.csv` — metrics for every outer fold
- `confusion_matrix.png` — final outer test fold for the best model
- `roc_curve.png` — ROC curve for that same held-out fold
- `accuracy_f1_comparison.png` — comparison plot
- `feature_importance.png` and `feature_importance.csv` — permutation importance
- `classification_report.txt`
- `wilcoxon_comparison.csv` — paired Wilcoxon tests with Holm adjustment
- `run_summary.txt`

**Important:** Results generated on another computer or software version can differ slightly. Use the files produced by your own run in your final report. The confusion matrix is for one held-out fold, not a sum across all 50 folds. The repeated-fold Wilcoxon p-values are indicative because repeated cross-validation folds are not fully independent.

## Repository structure
```text
.
├── experiment.py
├── requirements.txt
├── README.md
├── dataset/
├── results/
├── screenshots/
└── report/
```

## Research and safety note
This is an educational research project, not a medical device. It must not be used to diagnose patients or guide treatment. External clinical validation is required before any clinical use.

## Citation
Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

Scikit-learn dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
