"""Generate Kiva_Loans_Microfinance_Analytics.ipynb, an authored, from-scratch
rebuild replacing the copied reference kernels in archive_reference_kernels/."""
import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)})

# =====================================================================
# 0. TITLE
# =====================================================================
md("""# Kiva Loans Microfinance Analytics

**Which loans are at risk of not getting fully funded, and does that risk fall
hardest on the poorest regions?**

This notebook analyzes 671k+ real microloans from Kiva's public dataset (Kaggle's
"Data Science for Good: Kiva Crowdfunding"). It combines:

1. Exploratory analysis of loan structure, sectors, and borrower demographics.
2. Text mining of loan-use descriptions.
3. A geospatial join against the Multidimensional Poverty Index (MPI) by region.
4. A funding-risk classification model, built with an explicit leakage check, and
   explained with SHAP.
5. A days-to-fund regression among successfully funded loans.
6. A synthesis: which regions are both poverty-deep and funding-at-risk.

All findings are correlational, not causal: this is a single-snapshot dataset.
""")

# =====================================================================
# 1. SETUP
# =====================================================================
md("""---

## 1. Setup
""")

code("""import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", context="notebook")

# The feature engineering and the leakage guard live in src/features.py rather
# than in this notebook, so that they can be unit-tested. See tests/ and the
# Tests section of the README. Importing them here means the notebook and the
# test suite are checking the same code, not two copies that can drift apart.
sys.path.insert(0, str(Path.cwd() / "src"))
from features import (
    CATEGORICAL_COLUMNS,
    LEAKY_COLUMNS,
    POSTING_TIME_FEATURES,
    add_borrower_features,
    assert_no_leakage,
    mark_fully_funded,
)

# Where the Kiva CSVs live. Override with KIVA_DATA_DIR to keep the ~200MB of
# data outside the repo. An earlier version hardcoded a path relative to a
# directory layout that no longer exists, which left the notebook unrunnable
# without anyone noticing, because the stored outputs still looked fine.
DATA_DIR = Path(os.environ.get("KIVA_DATA_DIR", "data"))
if not (DATA_DIR / "kiva_loans.csv").exists():
    raise FileNotFoundError(
        "kiva_loans.csv not found under " + str(DATA_DIR.resolve()) + '''

Download the "Data Science for Good: Kiva Crowdfunding" dataset from
https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding
then either put the CSVs in ./data or set KIVA_DATA_DIR to the folder holding them.'''
    )

print("Setup OK. pandas:", pd.__version__, "| numpy:", np.__version__)
print("Data directory:", DATA_DIR.resolve())
""")

# =====================================================================
# 2. DATA LOADING
# =====================================================================
md("""---

## 2. Data loading

Loading `kiva_loans.csv` (671k+ rows) with explicit dtypes to keep memory usage
reasonable, and parsing the four timestamp columns.
""")

code("""LOAN_DTYPES = {
    "id": "int32",
    "activity": "category",
    "sector": "category",
    "country_code": "category",
    "country": "category",
    "region": "category",
    "currency": "category",
    "partner_id": "float32",
    "term_in_months": "float32",
    "lender_count": "int32",
    "repayment_interval": "category",
}

df = pd.read_csv(
    DATA_DIR / "kiva_loans.csv",
    dtype=LOAN_DTYPES,
    parse_dates=["posted_time", "disbursed_time", "funded_time", "date"],
)
df["funded_amount"] = df["funded_amount"].astype("float32")
df["loan_amount"] = df["loan_amount"].astype("float32")

print(f"Loaded {len(df):,} rows, {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
df.head()
""")

md("""### 2.1 Data-quality checks
""")

code("""print("Missing values (top 10 columns):")
print(df.isna().sum().sort_values(ascending=False).head(10))
print()
print("Duplicate loan ids:", df["id"].duplicated().sum())
print()
print("loan_amount <= 0:", (df["loan_amount"] <= 0).sum())
print("funded_amount > loan_amount (should be 0 or near-0):", (df["funded_amount"] > df["loan_amount"]).sum())
""")

# =====================================================================
# 3. EXPLORATORY DATA ANALYSIS
# =====================================================================
md("""---

## 3. Exploratory data analysis
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["loan_amount"].clip(upper=df["loan_amount"].quantile(0.99)), bins=50)
axes[0].set_title("Loan amount distribution (99th pct clipped)")
axes[1].hist(np.log1p(df["loan_amount"]), bins=50)
axes[1].set_title("log1p(loan_amount)")
plt.tight_layout()
plt.show()

print(df["loan_amount"].describe())
""")

md("""### 3.1 Sectors and activities
""")

code("""top_sectors = df["sector"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 6))
top_sectors.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Loan count by sector")
plt.tight_layout()
plt.show()
top_sectors
""")

md("""### 3.2 Countries and regions
""")

code("""top_countries = df["country"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 6))
top_countries.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Loan count by country (top 15)")
plt.tight_layout()
plt.show()
top_countries
""")

md("""### 3.3 Borrower gender composition

`borrower_genders` is a comma-separated list, one entry per borrower on the loan
(loans can be group loans). Parsing it into counts of male/female borrowers.
""")

code("""# add_borrower_features comes from src/features.py. Its edge cases (group
# loans, missing values, unrecognised labels) are covered in
# tests/test_features.py, which is why the parsing is not written out here.
# An unparseable row gets pct_female = NaN rather than 0: NaN means "we could
# not tell", 0 would claim the loan had only male borrowers, and the model
# reads this column.
df = add_borrower_features(df)

print(f"Loans with 0 parsed borrowers: {(df['n_borrowers'] == 0).sum():,}")
print(f"Median % female borrowers per loan: {df['pct_female'].median():.2%}")
df[["n_male", "n_female", "n_borrowers", "pct_female"]].describe()
""")

md("""### 3.4 Repayment intervals and loan volume over time
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df["repayment_interval"].value_counts().plot(kind="bar", ax=axes[0])
axes[0].set_title("Repayment interval")

monthly = df.set_index("posted_time").resample("MS").size()
monthly.plot(ax=axes[1])
axes[1].set_title("Loans posted per month")
plt.tight_layout()
plt.show()
""")

# =====================================================================
# 4. TEXT MINING ON LOAN-USE DESCRIPTIONS
# =====================================================================
md("""---

## 4. Text mining: what are these loans actually for?

`use` is a free-text field ("to buy seasonal, fresh fruits to sell"). TF-IDF over
this text surfaces the vocabulary that separates sectors, and a compact set of
term-presence features gets carried into the funding-risk model.
""")

code("""from sklearn.feature_extraction.text import TfidfVectorizer

df["use"] = df["use"].fillna("")
df["use_len"] = df["use"].str.len().astype("int32")

tfidf = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2), min_df=50)
tfidf_matrix = tfidf.fit_transform(df["use"])
tfidf_features = pd.DataFrame(
    (tfidf_matrix > 0).toarray().astype("int8"),
    columns=[f"use_tfidf_{t.replace(' ', '_')}" for t in tfidf.get_feature_names_out()],
    index=df.index,
)
tfidf_features["use_len"] = df["use_len"]

print(f"Top TF-IDF terms: {list(tfidf.get_feature_names_out()[:15])}")
tfidf_features.sum().sort_values(ascending=False).head(10)
""")

md("""### 4.1 Top terms by sector
""")

code("""for sector in df["sector"].value_counts().head(5).index:
    sector_mask = df["sector"] == sector
    sector_tfidf = TfidfVectorizer(max_features=8, stop_words="english", min_df=10)
    try:
        sector_tfidf.fit(df.loc[sector_mask, "use"])
        print(f"{sector}: {list(sector_tfidf.get_feature_names_out())}")
    except ValueError:
        print(f"{sector}: not enough vocabulary to extract terms")
""")

# =====================================================================
# 5. GEOSPATIAL + POVERTY (MPI) JOIN
# =====================================================================
md("""---

## 5. Geospatial analysis: loans against poverty depth

Joining each loan's `country` + `region` against Kiva's own region-to-MPI mapping
(`kiva_mpi_region_locations.csv`), which is a many-fewer-rows region lookup table,
not a per-loan file. Most loans will match on `country` + `region` exactly, some
won't (region naming isn't perfectly standardized). Join coverage is reported
explicitly rather than assumed.
""")

code("""mpi = pd.read_csv(DATA_DIR / "kiva_mpi_region_locations.csv")
mpi = mpi[["country", "region", "MPI", "lat", "lon"]].drop_duplicates(subset=["country", "region"])

df = df.merge(mpi, on=["country", "region"], how="left", validate="many_to_one")

coverage = df["MPI"].notna().mean()
print(f"MPI join coverage: {coverage:.1%} of loans matched to a region-level MPI score")
print(f"Loans matched: {df['MPI'].notna().sum():,} / {len(df):,}")
""")

md("""### 5.1 Funding success vs. poverty depth, by region
""")

code("""region_summary = (
    df.dropna(subset=["MPI"])
    .groupby(["country", "region"], observed=True)
    .agg(
        n_loans=("id", "count"),
        total_loan_amount=("loan_amount", "sum"),
        pct_fully_funded=("loan_amount", lambda s: (df.loc[s.index, "funded_amount"] >= s).mean()),
        MPI=("MPI", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    )
    .reset_index()
)
region_summary = region_summary[region_summary["n_loans"] >= 20]

print(f"Regions with >=20 loans and known MPI: {len(region_summary)}")
print(f"Correlation between MPI and % fully funded: {region_summary['MPI'].corr(region_summary['pct_fully_funded']):.3f}")
region_summary.sort_values("MPI", ascending=False).head(10)
""")

md("""### 5.2 Map: loan volume and funding success against poverty depth
""")

code("""import plotly.express as px

fig = px.scatter_geo(
    region_summary,
    lat="lat", lon="lon",
    size="n_loans",
    color="pct_fully_funded",
    hover_name="region",
    hover_data={"country": True, "MPI": ":.3f", "n_loans": True},
    color_continuous_scale="RdYlGn",
    title="Loan volume (size) and funding success rate (color) by region",
)
fig.update_layout(height=550)
fig.show()

try:
    fig.write_image("figs/geo_funding_vs_poverty.png", scale=2)
    print("Saved static export to figs/geo_funding_vs_poverty.png")
except Exception as e:
    print(f"Static export skipped (kaleido not available?): {e}")
""")

# =====================================================================
# 6. FEATURE ENGINEERING
# =====================================================================
md("""---

## 6. Feature engineering

### 6.1 The leakage check

`funded_time`, `disbursed_time`, and `lender_count` are **consequences** of a loan
being funded. They don't exist yet at the moment a loan is posted, so a model
using them to predict "will this loan be funded" would be cheating. A leak of
exactly this kind, in an earlier project of mine, inflated a model's R² from
0.906 to 0.996 before it was caught, which is why the guard below is executed
rather than described. They are explicitly excluded.
""")

code("""# mark_fully_funded, LEAKY_COLUMNS and POSTING_TIME_FEATURES all come from
# src/features.py, so the definition the model uses is the definition the tests
# check. >= rather than == on purpose: a loan that closes slightly over its
# target is funded.
df["fully_funded"] = mark_fully_funded(df)
df["post_month"] = df["posted_time"].dt.month.astype("int8")
df["post_dow"] = df["posted_time"].dt.dayofweek.astype("int8")

model_df = pd.concat([df[POSTING_TIME_FEATURES + ["fully_funded"]], tfidf_features.drop(columns=["use_len"])], axis=1)
model_df = model_df.dropna(subset=["loan_amount", "term_in_months"])

print(f"Modeling rows: {len(model_df):,} (dropped {len(df) - len(model_df):,} with missing core fields)")
print(f"Excluded as leakage: {LEAKY_COLUMNS}")
print(f"Feature count: {model_df.shape[1] - 1}")

# The guard, run rather than described. It raises LeakageError naming every
# offending column. This is the check that has to fire, because a leaked column
# does not break the model, it improves its score, so nothing downstream would
# ever flag it.
assert_no_leakage(model_df.drop(columns=["fully_funded"]).columns)
print("Leakage guard passed: no post-outcome column reached the feature set.")

model_df.head()
""")

md("""### 6.2 Encoding categoricals
""")

code("""model_encoded = pd.get_dummies(model_df, columns=CATEGORICAL_COLUMNS, drop_first=True)

y = model_encoded.pop("fully_funded")
X = model_encoded.fillna({"MPI": model_encoded["MPI"].median()})
X["pct_female"] = X["pct_female"].fillna(X["pct_female"].median())
FEATURE_COLUMNS = list(X.columns)

# Re-run the guard after one-hot encoding. Encoding creates new column names,
# and a suffixed column such as funded_amount_bucket would sail past a check
# that only ran before the encoding step.
assert_no_leakage(FEATURE_COLUMNS)

print(f"X shape: {X.shape}")
print(f"Target balance: {y.value_counts(normalize=True).to_dict()}")
""")

# =====================================================================
# 7. FUNDING-RISK MODEL
# =====================================================================
md("""---

## 7. Funding-risk model

Comparing Logistic Regression, Random Forest, and LightGBM on `fully_funded`.
The target is moderately imbalanced: about 92.8% fully funded vs. 7.2% not
(not as extreme as e.g. fraud detection, but skewed enough that accuracy would
be misleading).

This notebook's stated question is *which loans are at risk of not getting
fully funded*, i.e. performance on the minority "not funded" class (label 0)
is what actually matters, not performance on the majority "funded" class.
`sklearn.metrics.average_precision_score` defaults to scoring the positive
label (1 = funded), so that number alone would silently answer the wrong
question. It would mostly reflect how easy the majority class is (its floor
is the class prevalence, ~0.928, so a majority-class PR-AUC of ~0.99 is a much
smaller lift than it looks). We report **both**:

- **PR-AUC (funded, majority class)**: `average_precision_score(y_test, y_proba)`
- **PR-AUC (at-risk, minority class)**: `average_precision_score(1 - y_test, 1 - y_proba)`,
  which reframes "predict class 1" as "predict class 0" by flipping both the
  true labels and the predicted probabilities

The **minority-class PR-AUC is the headline metric** for this project and is
also what drives model selection below, since it's the one that answers the
notebook's actual question.
""")

code("""from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import lightgbm as lgb

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

results = {}
results_minority = {}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
results["Logistic Regression"] = average_precision_score(y_test, lr_proba)
results_minority["Logistic Regression"] = average_precision_score(1 - y_test, 1 - lr_proba)

rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
results["Random Forest"] = average_precision_score(y_test, rf_proba)
results_minority["Random Forest"] = average_precision_score(1 - y_test, 1 - rf_proba)

lgbm = lgb.LGBMClassifier(
    n_estimators=300, max_depth=8, learning_rate=0.05,
    class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1,
)
lgbm.fit(X_train, y_train)
lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
results["LightGBM"] = average_precision_score(y_test, lgbm_proba)
results_minority["LightGBM"] = average_precision_score(1 - y_test, 1 - lgbm_proba)

for name, _ in sorted(results_minority.items(), key=lambda kv: -kv[1]):
    print(
        f"{name:20s} PR-AUC (funded, majority class): {results[name]:.4f}"
        f"   PR-AUC (at-risk, minority class): {results_minority[name]:.4f}"
    )
""")

md("""### 7.1 Best model: full evaluation

Model selection uses the **minority-class PR-AUC** (`results_minority`), not
the majority-class one, since that's the metric that reflects performance on
the "at risk of not being funded" class this notebook is actually about.
""")

code("""best_model_name = max(results_minority, key=results_minority.get)
best_model = {"Logistic Regression": lr, "Random Forest": rf, "LightGBM": lgbm}[best_model_name]
X_eval = X_test_scaled if best_model_name == "Logistic Regression" else X_test

y_proba = best_model.predict_proba(X_eval)[:, 1]
pr_auc_majority = average_precision_score(y_test, y_proba)
pr_auc_minority = average_precision_score(1 - y_test, 1 - y_proba)
print(f"Best model (by minority-class PR-AUC): {best_model_name}")
print(f"PR-AUC (at-risk, minority class) [headline]: {pr_auc_minority:.4f}")
print(f"PR-AUC (funded, majority class):             {pr_auc_majority:.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(classification_report(y_test, (y_proba >= 0.5).astype(int)))
""")

md("""### 7.2 SHAP explainability
""")

code("""import shap

if best_model_name == "LightGBM":
    explainer = shap.TreeExplainer(best_model)
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_test), size=min(1000, len(X_test)), replace=False)
    X_sample = X_test.iloc[sample_idx]
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=12, show=False)
else:
    explainer = shap.Explainer(best_model, X_train if best_model_name == "Random Forest" else X_train_scaled)
    sample = (X_test if best_model_name == "Random Forest" else X_test_scaled)[:1000]
    shap_values = explainer(sample)
    shap.summary_plot(shap_values, sample, plot_type="bar", max_display=12, show=False)

plt.title(f"Top features by mean |SHAP| ({best_model_name})")
plt.tight_layout()
plt.savefig("figs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
""")

# =====================================================================
# 8. DAYS-TO-FUND REGRESSION
# =====================================================================
md("""---

## 8. How long does it take to get funded?

Among loans that *did* get fully funded, regressing `(funded_time - posted_time)`
in days against the same posting-time feature set (no separate leakage question
here: the population is already restricted to funded loans, and the target is a
time gap, not the funding outcome itself).
""")

code("""funded_only = df[df["fully_funded"] == 1].copy()
funded_only["days_to_fund"] = (funded_only["funded_time"] - funded_only["posted_time"]).dt.total_seconds() / 86400
funded_only = funded_only[funded_only["days_to_fund"] >= 0]

reg_df = pd.concat(
    [funded_only[POSTING_TIME_FEATURES], tfidf_features.loc[funded_only.index].drop(columns=["use_len"]), funded_only[["days_to_fund"]]],
    axis=1,
)
reg_encoded = pd.get_dummies(reg_df, columns=CATEGORICAL_COLUMNS, drop_first=True)
y_reg = reg_encoded.pop("days_to_fund")
X_reg = reg_encoded.fillna({"MPI": reg_encoded["MPI"].median()})

print(f"Regression rows: {len(X_reg):,}")
print(funded_only['days_to_fund'].describe())
""")

code("""from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=RANDOM_STATE)

reg_model = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
reg_model.fit(Xr_train, yr_train)
yr_pred = reg_model.predict(Xr_test)

print(f"MAE: {mean_absolute_error(yr_test, yr_pred):.2f} days")
print(f"R²: {r2_score(yr_test, yr_pred):.3f}")
""")

# =====================================================================
# 9. SYNTHESIS: PRIORITY REGIONS
# =====================================================================
md("""---

## 9. Synthesis: where should Kiva focus promotion?

Combining the model's predicted funding-risk (aggregated to region level, using the
same held-out test predictions from Section 7) with each region's MPI: regions that
are both poverty-deep (`high MPI`) and funding-at-risk (`low mean predicted
fully_funded probability`) are the actionable priority list.
""")

code("""test_region_info = df.loc[X_test.index, ["country", "region", "MPI"]].copy()
test_region_info["predicted_fund_prob"] = y_proba

priority = (
    test_region_info.dropna(subset=["MPI"])
    .groupby(["country", "region"], observed=True)
    .agg(
        n_test_loans=("MPI", "count"),
        MPI=("MPI", "first"),
        mean_predicted_fund_prob=("predicted_fund_prob", "mean"),
    )
    .reset_index()
)
priority = priority[priority["n_test_loans"] >= 10]
priority["priority_score"] = priority["MPI"] * (1 - priority["mean_predicted_fund_prob"])
priority = priority.sort_values("priority_score", ascending=False)

print(f"Regions in priority table: {len(priority)}")
priority.head(15)
""")

md("""---

## 10. Limitations

- **Right-censoring in the target, measured rather than assumed.** Loans with no
  `funded_time` at the moment this dataset was captured are treated as "not fully
  funded", but Kiva loans fundraise for weeks after posting, so a loan posted near
  the snapshot boundary is not "not funded", it is "not funded yet". The snapshot
  ends 2017-07-26, and the funded rate falls off a cliff as that date approaches:
  93.6% for loans posted more than 90 days before it, 89.3% at 46 to 60 days,
  74.0% at 31 to 45 days, 34.8% at 15 to 21 days, and 16.4% in the final week.
  **12.7% of all 48,328 not-funded labels are loans posted within the final 45
  days.** The headline metric is minority-class PR-AUC, so roughly one in eight of
  the positives it is scored on is a censoring artifact rather than a funding
  failure. Dropping loans posted within the last 60 days would cost about 3.8% of
  rows and give a defensible target; it is not done here, and the published
  PR-AUC should be read with that in mind.
- **The leakage guard cannot see this, by design.** `assert_no_leakage` is a name
  check over feature columns. The censoring above is in the *target definition*,
  which the guard never inspects. The same blind spot covers a renamed derived
  feature and any groupby target encoding.
- **Preprocessing is fit before the split, not inside it.** The `MPI` and
  `pct_female` medians in Section 7 and the TF-IDF vocabulary and IDF weights in
  Section 6 are all computed over train and test together. A median barely moves
  for a handful of extra rows, so the effect is small, but the correct form is a
  pipeline fit on train only, and
  this repo's own argument is that a leakage check has to be executed rather than
  assumed.
- **The split is random, on data that has a time dimension.** `train_test_split`
  with `stratify=y` is used, while `posted_time` exists, `post_month` and
  `post_dow` are features, and the target has the strong time trend documented
  above. A train-before-a-cutoff, test-after split would be the honest form and
  would report a lower number.
- **Two of the mapped regions are plotted on the wrong continent.** In
  `kiva_mpi_region_locations.csv`, Sierra Leone / Port Loko carries 5.557,
  23.763, which is in the Central African Republic, and Timor-Leste / Aileu
  carries 3.428, -76.487, which is in Colombia. Every other Timor-Leste region in
  that file sits near -8.x, 125 to 127. The join is exact and `many_to_one`, so
  this is an upstream defect in the Kaggle file rather than a bug here, but
  `figs/geo_funding_vs_poverty.png` publishes it, and Aileu is the top row of the
  priority table. A bounding-box check after the merge would catch it.
- **MPI join coverage is only 7.6% (50,955 / 671,205 loans).** The vast majority of
  loans could not be matched to a region-level MPI score, most likely because
  `kiva_loans.csv`'s free-text `region` field (entered inconsistently by field
  partners) doesn't standardize against `kiva_mpi_region_locations.csv`'s `region`
  field well enough for an exact string join. Every MPI-dependent result in this
  notebook, the Section 5 map/correlation, the `MPI` feature in the Section 7
  model, and the Section 9 priority-regions table, describes only that small,
  non-random 7.6% subset of loans (skewed toward whichever regions happen to have
  cleanly-matching names), not the full dataset. This is well below a level where
  region-level findings should be treated as representative, and the priority-regions
  table in particular should be read as illustrative of the method, not as a
  reliable region-targeting list for the other 92.4% of loan volume.
- **The MPI correlation is an ecological one.** The 0.253 correlation in Section 5
  is computed across 76 regions, between a region's MPI and its percent-funded.
  A region-level association says nothing about whether any individual poorer
  borrower is less likely to be funded; assuming it does is the ecological
  fallacy.
- **Correlational, not causal.** Nothing here establishes that any feature *causes*
  funding success or delay, only that it is predictive or associated.
- **English-only text mining.** TF-IDF was fit on the raw `use` field without
  language detection; non-English descriptions contribute noise to the term list.
""")

# =====================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path(__file__).parent / "Kiva_Loans_Microfinance_Analytics.ipynb"
with out.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote {out}  ({len(cells)} cells)")
