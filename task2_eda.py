"""
PRODIGY_DS_02
--------------
Task: Perform data cleaning and exploratory data analysis (EDA) on a dataset
of your choice, such as the Titanic dataset from Kaggle. Explore the
relationships between variables and identify patterns and trends in the data.

Dataset: titanic.csv (891 passengers) — the classic Titanic dataset.
Output : output/*.png — six EDA visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ===========================================================
# 1. LOAD DATA
# ===========================================================
df = pd.read_csv("titanic.csv")
print("Shape:", df.shape)
print("\nColumn info:")
print(df.info())

# ===========================================================
# 2. DATA CLEANING
# ===========================================================
print("\n" + "=" * 50)
print("DATA CLEANING")
print("=" * 50)

print("\nMissing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 'deck' is missing for 688 of 891 rows (~77%) — too sparse to impute
# meaningfully, so we drop the column rather than guess most values.
df = df.drop(columns=["deck"])

# 'age' has 177 missing values (~20%). Instead of a single global median
# (which would ignore that age varies a lot by passenger class), we impute
# using the median age WITHIN each passenger class — a more realistic fill.
df["age"] = df.groupby("class")["age"].transform(lambda x: x.fillna(x.median()))

# 'embarked'/'embark_town' are missing for just 2 rows — safe to drop these
# rows entirely since it's a negligible fraction of the data.
df = df.dropna(subset=["embarked", "embark_town"])

# Check for duplicate rows. Note: this dataset has no passenger ID column,
# so "duplicates" just means identical values across all columns — some of
# these could be coincidental (e.g. two 3rd class men, same age & fare) not
# true data-entry errors. We report the count but keep them, since removing
# rows based on coincidental attribute matches could delete real passengers.
dupes = df.duplicated().sum()
print(f"\nDuplicate rows found: {dupes} (kept — see comment in script)")

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum(), "total missing values remain")
print("Final shape:", df.shape)

# ===========================================================
# 3. EXPLORATORY DATA ANALYSIS
# ===========================================================
print("\n" + "=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

overall_rate = df["survived"].mean()
print(f"\nOverall survival rate: {overall_rate:.1%}")

# -----------------------------------------------------------
# 3a. Survival rate by passenger class
# -----------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x="class", y="survived", hue="class",
            order=["First", "Second", "Third"], palette="Blues_d", legend=False)
plt.title("Survival Rate by Passenger Class", fontsize=14, fontweight="bold")
plt.ylabel("Survival Rate")
plt.xlabel("Class")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("output/survival_by_class.png", dpi=150)
plt.close()
print("Saved output/survival_by_class.png")

# -----------------------------------------------------------
# 3b. Survival rate by gender
# -----------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.barplot(data=df, x="sex", y="survived", hue="sex", palette="Set2", legend=False)
plt.title("Survival Rate by Gender", fontsize=14, fontweight="bold")
plt.ylabel("Survival Rate")
plt.xlabel("Gender")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("output/survival_by_gender.png", dpi=150)
plt.close()
print("Saved output/survival_by_gender.png")

# -----------------------------------------------------------
# 3c. Age distribution split by survival outcome
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="alive", multiple="stack", bins=30, palette="Set1")
plt.title("Age Distribution by Survival Outcome", fontsize=14, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("output/age_distribution_by_survival.png", dpi=150)
plt.close()
print("Saved output/age_distribution_by_survival.png")

# -----------------------------------------------------------
# 3d. Fare distribution by class (boxplot — shows spread + outliers)
# -----------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="class", y="fare", hue="class",
            order=["First", "Second", "Third"], palette="Blues_d", legend=False)
plt.title("Fare Distribution by Passenger Class", fontsize=14, fontweight="bold")
plt.ylabel("Fare")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig("output/fare_by_class.png", dpi=150)
plt.close()
print("Saved output/fare_by_class.png")

# -----------------------------------------------------------
# 3e. Survival rate by class AND gender combined (the classic Titanic pattern)
# -----------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x="class", y="survived", hue="sex",
            order=["First", "Second", "Third"], palette="Set2")
plt.title("Survival Rate by Class and Gender", fontsize=14, fontweight="bold")
plt.ylabel("Survival Rate")
plt.xlabel("Class")
plt.ylim(0, 1)
plt.legend(title="Gender")
plt.tight_layout()
plt.savefig("output/survival_by_class_and_gender.png", dpi=150)
plt.close()
print("Saved output/survival_by_class_and_gender.png")

# -----------------------------------------------------------
# 3f. Correlation heatmap of numeric variables
# -----------------------------------------------------------
numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[numeric_cols].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Between Numeric Variables", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("output/correlation_heatmap.png", dpi=150)
plt.close()
print("Saved output/correlation_heatmap.png")

# ===========================================================
# 4. KEY FINDINGS SUMMARY
# ===========================================================
print("\n" + "=" * 50)
print("KEY FINDINGS")
print("=" * 50)

survival_by_class = df.groupby("class", observed=True)["survived"].mean()
survival_by_gender = df.groupby("sex")["survived"].mean()

print(f"\nSurvival by class:\n{survival_by_class}")
print(f"\nSurvival by gender:\n{survival_by_gender}")
print(f"\nFare vs Survival correlation: {corr.loc['fare', 'survived']:.3f}")
print(f"Pclass vs Survival correlation: {corr.loc['pclass', 'survived']:.3f}")
print("\nDone!")
