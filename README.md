# Financial Transactions: Data Cleaning, EDA & Predictive Modeling

**Author:** Gokul Balachander
**Course project:** ACM @ UCR — Data Analysis & Statistical Computing (Market Data Analysis Project)
**Dataset:** `dirty_financial_transactions.csv` — 100,000 raw e-commerce transaction records

## 1. Project Question

How did COVID-19 affect certain markets and sectors within the US economy?

## 2. Data Cleaning

The raw file had five classic dirty-data problems, all handled in `scripts/01_clean.py`:

| Issue | Example | Fix |
|---|---|---|
| Truncated product names | `"Smar"`, `"Coff"`, `"Lapt"` | Matched against 6 canonical products (`Headphones`, `Coffee`, `Coffee Machine`, `Tablet`, `Smartphone`, `Laptop`) using prefix matching |
| Inconsistent payment method labels | `"pay pal"`, `"creditcard"`, `"Credit Card"` | Normalized to 3 categories: `PayPal`, `Credit Card`, `Cash` |
| Inconsistent status labels | `"complete"`, `"Completed"`, `"completed"` | Normalized to `Completed`, `Pending`, `Failed` |
| Price stored as string with `$`, includes negative values | `"$420.21"`, `-445.34` | Stripped `$`, cast to float, took absolute value (negative price = entry error), **rounded to 2 decimals** |
| Negative / corrupted quantities | `-5.0`, `608.0` | Took absolute value; capped outliers above 1000 |
| Invalid calendar dates | `2025-02-30`, `2023-13-01` | Parsed with `errors='coerce'` — invalid dates become missing (`NaT`) rather than dropping the whole row, since ~68% of rows had this issue and dropping them all would have destroyed most of the dataset |
| Missing core identifiers | blank `Transaction_ID`, `Customer_ID` | Rows dropped — a transaction we can't identify or attribute to a customer isn't usable |
| Duplicate rows | exact row duplicates | Removed |

**Result:** 100,000 raw rows → **89,442 cleaned rows** (a `cleaning_log.txt` in `/data` has the full breakdown).

**Key judgment call:** Rather than deleting the ~68% of rows with unparseable dates (which would have
left only ~28k usable rows), invalid dates were nulled but the row was kept. This means date-based charts
and stats only use the ~32% of rows with a valid date, while revenue/product/payment analysis uses the
full cleaned set.

## 3. Exploratory Data Analysis

See `dashboard/dashboard.png` for the full 6-panel dashboard. Headline numbers:

- **Total revenue (cleaned data):** ~$5.57B
- **Average transaction value:** ~$98,427
- **Top product by revenue:** Tablet (~$1.13B)
- **Transaction status split:** ~50% Completed, ~17% Pending, ~17% Failed (~17% unknown/missing status, excluded from modeling)
- **Most common payment method:** Credit Card
- **Date coverage:** only 31.8% of cleaned rows have a valid, parseable date — this should be flagged as a data quality issue upstream if this were a real production dataset

## 4. Machine Learning — Predicting Transaction Status

Following the standard supervised classification workflow (drop non-predictive ID columns → encode
categoricals → train/test split (80/20, stratified) → scale for logistic regression → fit → evaluate),
three models were trained on 47,231 rows (rows with a known status, price, and quantity):

| Model | Accuracy |
|---|---|
| Logistic Regression | 59.96% |
| Decision Tree (max_depth=5) | 59.91% |
| Random Forest (100 trees) | 50.73% |

**Honest finding:** All three models land close to the "always predict Completed" baseline (~60%,
since Completed is ~60% of the modeling subset), and per-class recall for Failed/Pending is close to
zero across models (see `models/model_results.txt` for full classification reports and
`dashboard/cm_*.png` for confusion matrices). This is a meaningful result, not a failure of the
pipeline: it suggests that, in this dataset, transaction outcome (Completed/Pending/Failed) carries
little to no signal from price, quantity, product, payment method, or date — consistent with the
outcome being close to randomly assigned in this synthetic dataset. Feature importances
(`dashboard/feature_importance.png`) back this up — no single feature dominates.

## 5. Repo Structure

```
├── README.md                  <- this file
├── data/
│   ├── cleaned_transactions.csv
│   ├── cleaning_log.txt
│   └── eda_summary.json
├── dashboard/
│   ├── dashboard.png           <- main 6-panel EDA dashboard
│   ├── model_comparison.png
│   ├── feature_importance.png
│   └── cm_*.png                <- confusion matrices per model
├── models/
│   └── model_results.txt       <- full accuracy + classification reports
└── scripts/
    ├── 01_clean.py
    ├── 02_eda_dashboard.py
    └── 03_ml_models.py
```

## 6. How to Reproduce

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python scripts/01_clean.py
python scripts/02_eda_dashboard.py
python scripts/03_ml_models.py
```
