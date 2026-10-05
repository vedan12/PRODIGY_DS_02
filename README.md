# PRODIGY_DS_02 — Data Cleaning & Exploratory Data Analysis (Titanic)

**Prodigy InfoTech — Data Science Internship, Task 02**

## Task
Perform data cleaning and exploratory data analysis (EDA) on a dataset of
your choice, such as the Titanic dataset from Kaggle. Explore the
relationships between variables and identify patterns and trends in the data.

## Dataset
`titanic.csv` — 891 passengers from the Titanic, with columns for survival,
class, sex, age, fare, family aboard, embarkation point, and more.

## Data Cleaning
| Issue | Column | Action | Reasoning |
|---|---|---|---|
| 688/891 missing (~77%) | `deck` | Dropped the column | Too sparse to impute reliably |
| 177/891 missing (~20%) | `age` | Imputed with **median age within each passenger class** | More realistic than a single global median, since age varies meaningfully by class |
| 2/891 missing | `embarked`, `embark_town` | Dropped those 2 rows | Negligible fraction of the data |
| 116 duplicate rows (no unique ID column) | — | **Kept**, reported only | Likely coincidental matches (e.g. two 3rd-class men of the same age & fare), not data-entry errors — removing them risks deleting real passengers |

Final cleaned dataset: **889 rows, 14 columns, 0 missing values.**

## Exploratory Analysis — Charts
1. `survival_by_class.png` — survival rate by passenger class
2. `survival_by_gender.png` — survival rate by gender
3. `age_distribution_by_survival.png` — age histogram split by survival outcome
4. `fare_by_class.png` — fare distribution (boxplot) by class
5. `survival_by_class_and_gender.png` — combined class × gender survival pattern
6. `correlation_heatmap.png` — correlation matrix of numeric variables

## Key Findings
- **Overall survival rate: 38.2%**
- **Survival by class:** First 62.6% → Second 47.3% → Third 24.2% — a clear class gradient
- **Survival by gender:** Female 74.0% vs. Male 18.9% — the single strongest pattern in the data
- **Combined effect:** First-class women survived at ~97%, while third-class men survived at just ~14% — class and gender compound rather than act independently
- **Correlations:** `pclass` correlates negatively with survival (-0.34), `fare` positively (+0.26) — consistent with the class-based pattern above
- These findings reflect the "women and children first" evacuation policy combined with unequal access to lifeboats across passenger classes

## Files
- `task2_eda.py` — full cleaning + EDA pipeline
- `titanic.csv` — dataset used
- `output/` — all 6 generated charts
- `PRODIGY_DS_02.ipynb` — Colab-ready notebook version

## How to run
```bash
pip install pandas matplotlib seaborn
python task2_eda.py
```

---
*Part of the Data Science Internship @ Prodigy InfoTech (Oct 2026)*
#ProdigyInfoTech
