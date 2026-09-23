# Bank Customer Term-Deposit Subscription Prediction Using Machine Learning

## Problem Statement

Banks conduct direct marketing campaigns (phone calls) to encourage customers to subscribe to term deposits. Contacting every customer is inefficient and costly. This project analyses customer demographics and campaign-related data from a Portuguese bank and develops a machine-learning classification model to **predict whether a customer is likely to subscribe to a term deposit**.

---

## Objectives

1. Perform thorough exploratory data analysis (EDA) to understand customer and campaign patterns.
2. Build a reproducible preprocessing pipeline addressing class imbalance and data leakage.
3. Train and compare three classification models: Logistic Regression, Decision Tree, Random Forest.
4. Evaluate models using multiple metrics with careful attention to class imbalance.
5. Select and justify the final model based on the project objective.
6. Save the final trained model for reuse.

---

## Dataset

**Source:** UCI Machine Learning Repository — Bank Marketing Dataset  
**Citation:** Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems.  
**URL:** https://archive.ics.uci.edu/ml/datasets/bank+marketing  
**File:** `dataset/bank-full.csv`  
**Rows:** 45,211  
**Columns:** 17 (16 features + 1 target)  
**Delimiter:** semicolon (`;`)

---

## Dataset Description

The dataset describes customers of a Portuguese bank and their responses to a direct marketing campaign between 2008–2010.

### Features

| Column | Type | Description |
|--------|------|-------------|
| `age` | numeric | Client age (years) |
| `job` | categorical | Type of job (12 categories incl. `"unknown"`) |
| `marital` | categorical | Marital status: `married`, `single`, `divorced` |
| `education` | categorical | Education level: `primary`, `secondary`, `tertiary`, `unknown` |
| `default` | categorical | Has credit in default? `yes` / `no` |
| `balance` | numeric | Average yearly balance (€); can be negative |
| `housing` | categorical | Has housing loan? `yes` / `no` |
| `loan` | categorical | Has personal loan? `yes` / `no` |
| `contact` | categorical | Contact type: `cellular`, `telephone`, `unknown` |
| `day` | numeric | Last contact day of month |
| `month` | categorical | Last contact month (abbreviated lowercase, e.g. `may`) |
| `duration` | numeric | **Excluded** — last call duration (seconds); causes data leakage |
| `campaign` | numeric | Number of contacts during this campaign |
| `pdays` | numeric | Days since last contact in previous campaign; **-1 = never contacted** |
| `previous` | numeric | Number of contacts before this campaign |
| `poutcome` | categorical | Outcome of previous campaign: `success`, `failure`, `other`, `unknown` |
| `was_contacted` | numeric | **Engineered**: 1 if `pdays != -1`, else 0 |

### Target Variable

| Column | Values | Description |
|--------|--------|-------------|
| `y` | `0` (no) / `1` (yes) | Did the customer subscribe to a term deposit? |

**Class distribution:** 39,922 (88.3%) No — 5,289 (11.7%) Yes ← significant class imbalance

---

## Data Preprocessing

### Key Decisions

1. **`duration` dropped (data leakage):** This column records the duration of the last phone call and is only known after the call ends. Using it as a predictor would inflate performance for historical data but make the model useless for prospective prediction. It is dropped immediately on data load.

2. **`pdays = -1` preserved as-is:** The value -1 is a meaningful sentinel indicating a customer was never previously contacted. Original `pdays` values are preserved exactly. A new binary feature `was_contacted` (1 if `pdays != -1`, else 0) is added to make this distinction explicit.

3. **`"unknown"` values retained:** Several columns contain the literal string `"unknown"`. These are not NaN values — `df.isna()` returns False for them. They are kept as valid category levels and produce their own One-Hot Encoded indicator columns (e.g., `poutcome_unknown`). They carry predictive signal, especially for `poutcome` (~81.7% unknown) and `contact` (~28.8% unknown).

4. **Class imbalance addressed:** `class_weight="balanced"` is applied to all three models, causing the loss function to weight the minority class (subscribers) proportionally more during training.

### Pipeline Architecture

```
Numeric features (7):
  age, balance, day, campaign, pdays, previous, was_contacted
  → SimpleImputer(strategy="median")
  → StandardScaler()

Categorical features (9):
  job, marital, education, default, housing, loan, contact, month, poutcome
  → SimpleImputer(strategy="most_frequent")
  → OneHotEncoder(handle_unknown="ignore")

ColumnTransformer assembles both paths → 51 features total
Wrapped in Pipeline(preprocessor + classifier) → no leakage
```

---

## Exploratory Data Analysis

Key findings from EDA:

- **Class imbalance:** 88.3% No vs 11.7% Yes — accuracy is not a reliable metric.
- **`poutcome = success`** strongly predicts subscription (~64% subscription rate).
- **`contact = cellular`** has a higher subscription rate than `telephone` or `unknown`.
- **Months:** March, September, October, December show elevated per-contact success rates (these are smaller campaigns).
- **`was_contacted = 1`** customers have a noticeably higher subscription rate — prior campaign contact is a positive signal.
- **Demographics:** Students and retired customers have higher subscription rates; tertiary education is also associated with higher rates.
- **`campaign`:** Fewer contacts per campaign is associated with higher subscription probability — over-contacting reduces conversion.
- **`balance`** tends to be slightly higher for subscribers.

EDA figures are saved to `outputs/figures/`.

---

## Models

Three classifiers were trained, each embedded in a `Pipeline(ColumnTransformer + Estimator)` with `class_weight="balanced"`:

| Model | Key Parameters |
|-------|----------------|
| Logistic Regression | `class_weight="balanced"`, `max_iter=1000`, `solver="lbfgs"` |
| Decision Tree | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"`, `n_estimators=100` |

All models use a **stratified 80/20 train/test split** (`random_state=42`).

---

## Evaluation

### Metrics

Given the class imbalance (~88/12%), the following metrics were used:

- **Accuracy** — reported but not used for model selection (misleadingly high for majority-class models)
- **Precision** — fraction of predicted subscribers who actually subscribed (reduces wasted contacts)
- **Recall** — fraction of actual subscribers correctly identified (reduces missed opportunities)
- **F1-score** — harmonic mean of Precision and Recall (primary selection criterion)
- **ROC-AUC** — threshold-independent discriminative ability (secondary selection criterion)
- **Confusion Matrix** — visualises True/False Positives and Negatives

---

## Results

Results on held-out test set (9,043 rows, stratified):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.7553 | 0.2667 | 0.6238 | 0.3736 | 0.7722 |
| Decision Tree | 0.8394 | 0.3072 | 0.2968 | 0.3019 | 0.6041 |
| **Random Forest** | **0.8800** | **0.4849** | **0.4102** | **0.4444** | **0.7935** |

### Final Model: Random Forest

**Random Forest** was selected as the final model based on the project objective and the full results picture:

- **Highest F1-score (0.4444):** Best balance between Precision and Recall — the most relevant metric for a phone-based campaign where both subscriber capture and contact efficiency matter.
- **Highest ROC-AUC (0.7935):** Best overall discriminative ability across all classification thresholds.
- **Highest Precision (0.4849):** Nearly half of predicted subscribers actually subscribe — almost double Logistic Regression's precision, directly reducing wasted calls.
- **Decision Tree was rejected:** Poor ROC-AUC (0.6041) and F1 (0.3019) indicate overfitting and poor generalisation.
- **Logistic Regression trade-off:** Higher Recall (0.6238) but very low Precision (0.2667) — would flood the campaign with unlikely subscribers. Acceptable if per-contact cost is low, but suboptimal for standard phone campaigns.

The saved model is at `models/final_model.pkl`.

---

## How to Install

```bash
# Clone or download the project
cd Bank_Term_Deposit_Prediction

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, jupyter, ipykernel

---

## How to Run

### Run the Notebook (recommended)

```bash
jupyter notebook notebooks/analysis.ipynb
```

Then use **Kernel → Restart & Run All** to execute all cells.

### Run individual source modules (for verification)

```bash
# Test data loading
python src/data_loader.py

# Test EDA plots
python src/eda.py

# Test preprocessing
python src/preprocessing.py

# Train all models and evaluate
python src/train.py

# Test model save/load
python src/utils.py
```

---

## Project Structure

```
Bank_Term_Deposit_Prediction/
├── dataset/
│   └── bank-full.csv              ← original dataset (never modified)
├── src/
│   ├── __init__.py
│   ├── config.py                  ← paths, constants, RANDOM_STATE
│   ├── data_loader.py             ← load_data() + pdays + unknown audit
│   ├── eda.py                     ← reusable EDA plot functions
│   ├── preprocessing.py           ← ColumnTransformer pipeline
│   ├── train.py                   ← 3 models, evaluation, ROC curves
│   └── utils.py                   ← save_model() / load_model()
├── notebooks/
│   └── analysis.ipynb             ← full analysis notebook
├── models/
│   └── final_model.pkl            ← saved Random Forest pipeline
├── outputs/
│   └── figures/                   ← EDA and evaluation plots (PNG)
│       ├── 01_target_distribution.png
│       ├── 02_numeric_distributions.png
│       ├── 03_categorical_distributions.png
│       ├── 04_correlation_heatmap.png
│       ├── 05_pdays_analysis.png
│       ├── 06_sub_rate_*.png
│       ├── 07_unknown_audit.png
│       ├── 08_numeric_boxplots.png
│       ├── cm_*.png               ← confusion matrices
│       ├── roc_curves.png
│       └── metrics_comparison.png
├── requirements.txt
├── README.md
├── AGENTS.md
└── bank-term-deposit-plan.md
```

---

## Notes

- All paths are relative — the project runs on any machine without modification.
- The original `dataset/bank-full.csv` is never modified.
- `duration` is excluded from all models to prevent data leakage.
- `pdays = -1` values are preserved exactly in the dataset; `was_contacted` encodes the sentinel separately.
- `"unknown"` strings are retained and treated as valid category levels.
