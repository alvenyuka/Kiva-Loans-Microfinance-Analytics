"""Generate Kiva_Loans_Microfinance_Analytics.ipynb.

This script is the source of the notebook: edit it, regenerate, and execute the
notebook. Never edit the .ipynb by hand."""
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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RUN_STARTED = time.time()
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
# data outside the repo.
DATA_DIR = Path(os.environ.get("KIVA_DATA_DIR", "data"))
if not (DATA_DIR / "kiva_loans.csv").exists():
    raise FileNotFoundError(
        "kiva_loans.csv not found under " + str(DATA_DIR.resolve()) + '''

Download the "Data Science for Good: Kiva Crowdfunding" dataset from
https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding
then either put the CSVs in ./data or set KIVA_DATA_DIR to the folder holding them.'''
    )

print("Setup OK. pandas:", pd.__version__, "| numpy:", np.__version__)
print("Data directory:", "KIVA_DATA_DIR" if "KIVA_DATA_DIR" in os.environ else "./data")
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
axes[0].set_xlabel("Loan amount (USD)")
axes[0].set_ylabel("Number of loans")
axes[1].hist(np.log1p(df["loan_amount"]), bins=50)
axes[1].set_title("log1p(loan_amount)")
axes[1].set_xlabel("log(1 + loan amount in USD)")
axes[1].set_ylabel("Number of loans")
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
ax.set_xlabel("Number of loans")
ax.set_ylabel("Sector")
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
ax.set_xlabel("Number of loans")
ax.set_ylabel("Country")
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
axes[0].set_xlabel("Repayment interval")
axes[0].set_ylabel("Number of loans")

monthly = df.set_index("posted_time").resample("MS").size()
monthly.plot(ax=axes[1])
axes[1].set_title("Loans posted per month")
axes[1].set_xlabel("Month posted")
axes[1].set_ylabel("Loans posted per month")
plt.tight_layout()
plt.show()
""")

# =====================================================================
# 4. TEXT MINING ON LOAN-USE DESCRIPTIONS
# =====================================================================
md("""---

## 4. Text mining: what are these loans actually for?

`use` is a free-text field ("to buy seasonal, fresh fruits to sell"). TF-IDF over
this text surfaces the vocabulary that separates sectors. This section is
exploratory only: the vocabulary fitted here, on every loan, is not used by the
model. The model's term-presence features are refitted on the training period
alone in Section 6.2.
""")

code("""from sklearn.feature_extraction.text import TfidfVectorizer

df["use"] = df["use"].fillna("")
df["use_len"] = df["use"].str.len().astype("int32")

# Exploratory vocabulary over all loans. Not a model input (see 6.2).
tfidf_eda = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2), min_df=50)
eda_matrix = tfidf_eda.fit_transform(df["use"])
loans_per_term = pd.Series(
    np.asarray((eda_matrix > 0).sum(axis=0)).ravel(),
    index=tfidf_eda.get_feature_names_out(),
    name="loans_using_term",
)
del eda_matrix

print(f"Top TF-IDF terms: {list(tfidf_eda.get_feature_names_out()[:15])}")
loans_per_term.sort_values(ascending=False).head(10)
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

Some region names in the MPI file are mis-encoded upstream: the file is valid
UTF-8, but it stores "Maranhðo" and "Rondðnia" where the Brazilian states are
Maranhão and Rondônia. The names print as stored rather than being repaired by
hand.
""")

code("""mpi = pd.read_csv(DATA_DIR / "kiva_mpi_region_locations.csv")
mpi = mpi[["country", "region", "MPI", "lat", "lon"]].drop_duplicates(subset=["country", "region"])

# Coordinate sanity check. A region far from its country's median position, by more
# than five robust standard deviations of that country's spread and at least 10
# degrees, is mis-keyed in the upstream file, so its point is dropped from the map
# (its MPI value, which is what the model uses, is kept).
by_country = mpi.groupby("country")[["lat", "lon"]]
offset = (mpi[["lat", "lon"]] - by_country.transform("median")).abs()
spread = by_country.transform(lambda s: (s - s.median()).abs().median()) * 1.4826
limit = np.maximum(10, 5 * spread)
misplaced = (offset["lat"] > limit["lat"]) | (offset["lon"] > limit["lon"])
n_misplaced, n_located = int(misplaced.sum()), int(mpi["lat"].notna().sum())
print(f"Coordinates dropped as misplaced: {n_misplaced} of {n_located} located regions, for example:")
print(mpi.loc[misplaced, ["country", "region", "lat", "lon"]].head(8).to_string(index=False))
mpi.loc[misplaced, ["lat", "lon"]] = np.nan

df = df.merge(mpi, on=["country", "region"], how="left", validate="many_to_one")

coverage = df["MPI"].notna().mean()
print(f"MPI join coverage: {coverage:.1%} of loans matched to a region-level MPI score")
print(f"Loans matched: {df['MPI'].notna().sum():,} / {len(df):,}")
""")

md("""### 5.1 Funding success vs. poverty depth, by region
""")

code("""# The funding target and the settled-outcome rule are the ones the model uses
# (Section 6.1 measures why loans posted within 60 days of the snapshot are left
# out: many of them were still fundraising).
CENSOR_DAYS = 60
df["fully_funded"] = mark_fully_funded(df)
snapshot_end = df["posted_time"].max()
age_days = (snapshot_end - df["posted_time"]).dt.days
settled = age_days > CENSOR_DAYS

region_summary = (
    df[settled].dropna(subset=["MPI"])
    .groupby(["country", "region"], observed=True)
    .agg(
        n_loans=("id", "count"),
        total_loan_amount=("loan_amount", "sum"),
        pct_fully_funded=("fully_funded", "mean"),
        MPI=("MPI", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    )
    .reset_index()
)
region_summary = region_summary[region_summary["n_loans"] >= 20]

mpi_corr = region_summary["MPI"].corr(region_summary["pct_fully_funded"])
mpi_corr_rank = region_summary["MPI"].corr(region_summary["pct_fully_funded"], method="spearman")
region_loan_share = region_summary["n_loans"].sum() / settled.sum()
print(f"Regions with >=20 settled loans and known MPI: {len(region_summary)}")
print(f"Settled loans in those regions: {region_summary['n_loans'].sum():,} ({region_loan_share:.1%} of settled loans)")
print(f"Correlation between MPI and share fully funded (Pearson):  {mpi_corr:.3f}")
print(f"Rank correlation between MPI and share fully funded (Spearman): {mpi_corr_rank:.3f}")
region_summary.sort_values("MPI", ascending=False).head(10)
""")

md("""**Does funding risk fall hardest on the poorest regions?** Not in the data that
can answer it. Across the regions printed above (at least 20 settled loans and a
known MPI), both correlations are positive: in this matched subset, poorer regions
were funded slightly more often, not less. The subset holds only the share of
settled loans printed above, and the correlation is ecological, measured across
regions rather than borrowers (see Limitations), so it does not show that poorer
borrowers are favoured.
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
    labels={"pct_fully_funded": "Share fully funded", "n_loans": "Settled loans"},
    title="Loan volume (size) and funding success rate (color) by region",
)
fig.update_layout(height=550)

# A static PNG is shown inline so the map renders on GitHub, which does not
# display interactive Plotly output. Without kaleido, fall back to the
# interactive figure.
from IPython.display import Image, display
try:
    fig.write_image("figs/geo_funding_vs_poverty.png", scale=2)
    display(Image("figs/geo_funding_vs_poverty.png", width=900))
except Exception as e:
    print(f"Static export skipped (kaleido not available?): {e}")
    fig.show()
""")

# =====================================================================
# 6. FEATURE ENGINEERING
# =====================================================================
md("""---

## 6. Feature engineering

### 6.1 The leakage check

`funded_time`, `disbursed_time`, and `lender_count` are **consequences** of a loan
being funded. They don't exist yet at the moment a loan is posted, so a model
using them to predict "will this loan be funded" would be cheating. A leaked
column does not break a model; it improves the score, so nothing downstream
would flag it. The guard below is therefore executed, not described. They are
explicitly excluded.
""")

code("""# mark_fully_funded (applied in 5.1), LEAKY_COLUMNS and POSTING_TIME_FEATURES all
# come from src/features.py, so the definition the model uses is the definition
# the tests check. >= rather than == on purpose: a loan that closes slightly over
# its target is funded.
df["post_month"] = df["posted_time"].dt.month.astype("int8")
df["post_dow"] = df["posted_time"].dt.dayofweek.astype("int8")
# Time-trend control: months since the first loan in the data. Known at posting
# time. It is offered to the model selection in Section 7 as a separate LightGBM
# candidate, so that post_month cannot stand in for a platform-wide trend unseen.
TREND_COLUMN = "months_since_start"
df[TREND_COLUMN] = ((df["posted_time"] - df["posted_time"].min()).dt.days / 30.4375).astype("float32")

# Right-censoring. A loan with no funded_time when the snapshot was taken may still
# have been fundraising, so "not funded" is only known for loans posted well before
# the snapshot. Measure it; `settled` (defined in 5.1) keeps loans whose outcome was known.
age_band = pd.cut(age_days, [-1, 7, 14, 21, 30, 45, 60, 90, 10**6],
                  labels=["0-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61-90", ">90"])
censoring = df.groupby(age_band, observed=True)["fully_funded"].agg(funded_rate="mean", loans="size")
print(f"Snapshot ends {snapshot_end.date()}. Funded rate by days between posting and snapshot:")
print(censoring.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"Excluded {(~settled).sum():,} loans ({(~settled).mean():.1%}) posted within {CENSOR_DAYS} days of the snapshot.")
print(f"Settled loans: {settled.sum():,}, of which {1 - df.loc[settled, 'fully_funded'].mean():.1%} not fully funded")

model_df = df.loc[settled, POSTING_TIME_FEATURES + [TREND_COLUMN, "fully_funded", "use"]]
model_df = model_df.dropna(subset=["loan_amount", "term_in_months"])

print(f"Modeling rows: {len(model_df):,} (dropped {settled.sum() - len(model_df):,} settled loans with missing core fields)")
print(f"Excluded as leakage: {LEAKY_COLUMNS}")
print(f"Posting-time inputs: {len(POSTING_TIME_FEATURES)} columns and the use text, plus the optional time trend")

# The guard, run rather than described. It raises LeakageError naming every
# offending column. This is the check that has to fire, because a leaked column
# does not break the model, it improves its score, so nothing downstream would
# ever flag it.
assert_no_leakage(model_df.drop(columns=["fully_funded", "use"]).columns)
print("Leakage guard passed: no post-outcome column reached the feature set.")

model_df.head()
""")

md("""### 6.2 Time-based split, and preprocessing fitted on earlier loans only

Loans are split by posting date, the way a platform would use a model on loans it
has not yet seen:

- **Training period:** the earliest 80% of settled loans. Within it, the earliest
  80% form the **fit slice** and the latest 20% the **validation slice**, which is
  used to choose the model.
- **Test period:** the most recent 20%, scored once by the chosen model after it
  is refitted on the whole training period.

The TF-IDF vocabulary and the medians used to fill missing values are fitted on
the fit slice for model selection, and on the whole training period for the final
model, so no stage sees text or medians from the loans it is scored on. A
missing-poverty flag (`MPI_missing`) is added before the fill, so the model can
tell "median poverty" from "no MPI match".
""")

code("""posted = df.loc[model_df.index, "posted_time"]
TRAIN_SHARE = 0.8
split_date = posted.quantile(TRAIN_SHARE)
is_train = posted <= split_date
val_cut = posted[is_train].quantile(TRAIN_SHARE)
is_fit = is_train & (posted <= val_cut)
is_val = is_train & (posted > val_cut)
print(f"Fit slice:        posted up to {val_cut.date()} ({is_fit.sum():,} loans)")
print(f"Validation slice: posted after it, up to {split_date.date()} ({is_val.sum():,} loans)")
print(f"Test period:      posted after {split_date.date()} ({(~is_train).sum():,} loans)")


def build_matrix(fit_rows):
    # Encode model_df, with the text vocabulary and missing-value medians fitted on fit_rows only.
    vec = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2), min_df=50)
    vec.fit(model_df.loc[fit_rows, "use"])
    terms = pd.DataFrame(
        (vec.transform(model_df["use"]) > 0).toarray().astype("int8"),
        columns=[f"use_tfidf_{t.replace(' ', '_')}" for t in vec.get_feature_names_out()],
        index=model_df.index,
    )
    encoded = pd.get_dummies(pd.concat([model_df.drop(columns=["use", "fully_funded"]), terms], axis=1),
                             columns=CATEGORICAL_COLUMNS, drop_first=True)
    encoded["MPI_missing"] = encoded["MPI"].isna().astype("int8")
    medians = {c: encoded.loc[fit_rows, c].median() for c in ("MPI", "pct_female")}
    return encoded.fillna(medians), vec, medians


y = model_df["fully_funded"]
X_sel, _, _ = build_matrix(is_fit)                      # for model selection
X, tfidf_train, train_medians = build_matrix(is_train)  # for the final model
ALL_COLUMNS = list(X.columns)


def model_columns(frame, with_trend):
    # The two matrices can hold different text terms, so columns are chosen per matrix.
    return [c for c in frame.columns if with_trend or c != TREND_COLUMN]

# Re-run the guard after one-hot encoding. Encoding creates new column names,
# and a suffixed column such as funded_amount_bucket would sail past a check
# that only ran before the encoding step.
assert_no_leakage(ALL_COLUMNS)

print(f"X shape: {X.shape} ({len(model_columns(X, False))} base features, plus {TREND_COLUMN})")
for label, rows in (("fit", is_fit), ("validation", is_val), ("test", ~is_train)):
    print(f"Not fully funded, {label}: {1 - y[rows].mean():.1%}")
""")

# =====================================================================
# 7. FUNDING-RISK MODEL
# =====================================================================
md("""---

## 7. Funding-risk model

Comparing Logistic Regression, Random Forest and LightGBM on `fully_funded`, plus
a fourth candidate: the same LightGBM given the time-trend column. The target is
imbalanced (the shares are printed in 6.2), skewed enough that accuracy would be
misleading.

This notebook's stated question is *which loans are at risk of not getting
fully funded*, i.e. performance on the minority "not funded" class (label 0)
is what actually matters, not performance on the majority "funded" class.
`sklearn.metrics.average_precision_score` defaults to scoring the positive
label (1 = funded), so that number alone would silently answer the wrong
question. It would mostly reflect how easy the majority class is: its floor is
the funded share of the test set, printed below, so a majority-class PR-AUC
close to 1 is a much smaller lift than it looks. We report **both**:

- **PR-AUC (funded, majority class)**: `average_precision_score(y_test, y_proba)`
- **PR-AUC (at-risk, minority class)**: `average_precision_score(1 - y_test, 1 - y_proba)`,
  which reframes "predict class 1" as "predict class 0" by flipping both the
  true labels and the model scores

The **minority-class PR-AUC is the headline metric**. The model is chosen on it
using the validation slice only. The winner is then refitted on the whole
training period and scored on the test period. The other candidates are refitted
and scored too, for reference; their test scores play no part in the choice.

All four models use balanced class weights, so their outputs rank loans by risk
but are not calibrated funding probabilities. They are called scores below.
""")

code("""from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report
import lightgbm as lgb


def at_risk_pr_auc(y_true, funded_score):
    return average_precision_score(1 - y_true, 1 - funded_score)


def make_lgbm():
    return lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                              class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1)


CANDIDATES = {
    "Logistic Regression": (lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)), False),
    "Random Forest": (lambda: RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE), False),
    "LightGBM": (make_lgbm, False),
    "LightGBM + time trend": (make_lgbm, True),
}

# Stage 1: fit on the fit slice, score the validation slice, choose.
validation_pr_auc = {}
for name, (make, with_trend) in CANDIDATES.items():
    cols = model_columns(X_sel, with_trend)
    m = make().fit(X_sel.loc[is_fit, cols], y[is_fit])
    validation_pr_auc[name] = at_risk_pr_auc(y[is_val], m.predict_proba(X_sel.loc[is_val, cols])[:, 1])
    print(f"{name:22s} validation PR-AUC (at-risk): {validation_pr_auc[name]:.4f}")
del X_sel, m

best_model_name = max(validation_pr_auc, key=validation_pr_auc.get)
print(f"Chosen on the validation slice: {best_model_name}")
""")

code("""# Stage 2: refit every candidate on the whole training period and score the test
# period. The choice above is already fixed; the non-chosen rows are for reference.
X_train, X_test = X[is_train], X[~is_train]
y_train, y_test = y[is_train], y[~is_train]
test_pr_auc, test_pr_auc_majority = {}, {}
for name, (make, with_trend) in CANDIDATES.items():
    cols = model_columns(X, with_trend)
    m = make().fit(X_train[cols], y_train)
    proba = m.predict_proba(X_test[cols])[:, 1]
    test_pr_auc[name] = at_risk_pr_auc(y_test, proba)
    test_pr_auc_majority[name] = average_precision_score(y_test, proba)
    if name == best_model_name:
        best_model, y_proba = m, proba
del m

model_table = pd.DataFrame({
    "validation PR-AUC (at-risk)": validation_pr_auc,
    "test PR-AUC (at-risk)": test_pr_auc,
    "test PR-AUC (funded)": test_pr_auc_majority,
}).sort_values("validation PR-AUC (at-risk)", ascending=False)
print(model_table.to_string(float_format=lambda v: f"{v:.4f}"))
""")

md("""### 7.1 Chosen model: full evaluation on the test period
""")

code("""pr_auc_majority = average_precision_score(y_test, y_proba)
pr_auc_minority = at_risk_pr_auc(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"Chosen model (on validation PR-AUC): {best_model_name}")
print(f"PR-AUC (at-risk, minority class) [headline]: {pr_auc_minority:.4f}")
print(f"PR-AUC (funded, majority class):             {pr_auc_majority:.4f}")
print(f"Test-set funded share (majority-class PR-AUC floor): {y_test.mean():.3f}")
print(f"Test-set not-funded share (minority-class PR-AUC floor): {1 - y_test.mean():.3f}")
print(f"ROC-AUC: {roc_auc:.4f}")
report = classification_report(y_test, (y_proba >= 0.5).astype(int), output_dict=True)
print(classification_report(y_test, (y_proba >= 0.5).astype(int)))
""")

md("""### 7.2 SHAP explainability
""")

code("""import warnings

# SHAP pulls in tqdm, which warns when ipywidgets is absent, and warns that
# LightGBM's binary output format changed. Neither affects the values below.
warnings.filterwarnings("ignore", message="IProgress not found")
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded")
import shap

best_cols = model_columns(X, CANDIDATES[best_model_name][1])
sample = X_test[best_cols].sample(n=min(1000, len(X_test)), random_state=RANDOM_STATE)
if best_model_name == "Logistic Regression":
    scaler, linear = best_model[0], best_model[-1]
    background = scaler.transform(X_train[best_cols].sample(n=5000, random_state=RANDOM_STATE))
    shap_values = shap.LinearExplainer(linear, background).shap_values(scaler.transform(sample))
else:
    shap_values = shap.TreeExplainer(best_model).shap_values(sample)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[..., 1]

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=best_cols).sort_values(ascending=False)
print(f"Top 12 features by mean |SHAP| ({best_model_name}, 1,000 test loans, log-odds of being funded):")
print(mean_abs_shap.head(12).to_string(float_format=lambda v: f"{v:.4f}"))

shap.summary_plot(shap_values, sample, plot_type="bar", max_display=12, show=False)
plt.title(f"Top features by mean |SHAP| ({best_model_name})")
plt.xlabel("Mean |SHAP value| (impact on log-odds of being funded)")
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
time gap, not the funding outcome itself). The comparator is a naive forecast:
the median days-to-fund of the training period, given to every test loan.
""")

code("""funded_only = df[(df["fully_funded"] == 1) & settled].copy()
funded_only["days_to_fund"] = (funded_only["funded_time"] - funded_only["posted_time"]).dt.total_seconds() / 86400
funded_only = funded_only[funded_only["days_to_fund"] >= 0]

reg_terms = pd.DataFrame(
    (tfidf_train.transform(funded_only["use"]) > 0).toarray().astype("int8"),
    columns=[f"use_tfidf_{t.replace(' ', '_')}" for t in tfidf_train.get_feature_names_out()],
    index=funded_only.index,
)
reg_df = pd.concat([funded_only[POSTING_TIME_FEATURES], reg_terms, funded_only[["days_to_fund"]]], axis=1)
reg_encoded = pd.get_dummies(reg_df, columns=CATEGORICAL_COLUMNS, drop_first=True)
reg_encoded["MPI_missing"] = reg_encoded["MPI"].isna().astype("int8")
y_reg = reg_encoded.pop("days_to_fund")
reg_train = funded_only["posted_time"] <= split_date
X_reg = reg_encoded.fillna({c: reg_encoded.loc[reg_train, c].median() for c in ("MPI", "pct_female")})

print(f"Regression rows: {len(X_reg):,}")
print(funded_only['days_to_fund'].describe())
""")

code("""from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

Xr_train, Xr_test = X_reg[reg_train], X_reg[~reg_train]
yr_train, yr_test = y_reg[reg_train], y_reg[~reg_train]

reg_model = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
reg_model.fit(Xr_train, yr_train)
yr_pred = reg_model.predict(Xr_test)

reg_mae = mean_absolute_error(yr_test, yr_pred)
reg_r2 = r2_score(yr_test, yr_pred)
baseline_days = float(yr_train.median())
baseline_mae = mean_absolute_error(yr_test, np.full(len(yr_test), baseline_days))
print(f"Test loans: {len(yr_test):,}")
print(f"MAE, random forest:                        {reg_mae:.2f} days")
print(f"MAE, naive baseline (train median {baseline_days:.1f} days): {baseline_mae:.2f} days")
print(f"R-squared, random forest: {reg_r2:.3f}")
""")

# =====================================================================
# 9. SYNTHESIS: PRIORITY REGIONS
# =====================================================================
md("""---

## 9. Synthesis: where should Kiva focus promotion?

Combining the model's funding-risk score (aggregated to region level, using the
same test-period scores from Section 7) with each region's MPI. The scores are not
calibrated probabilities (balanced class weights move them towards 0.5), so they
are not multiplied by MPI. Instead each region is ranked twice among the regions
in the table, on poverty (higher MPI ranks higher) and on funding risk (lower mean
score ranks higher), and the **relative priority index** is the product of the two
percentile ranks: 1 means the poorest and riskiest region in the table. It orders
regions; its scale carries no other meaning.
""")

code("""test_region_info = df.loc[X_test.index, ["country", "region", "MPI"]].copy()
test_region_info["model_score"] = y_proba  # higher = more likely to be funded; a ranking score

priority = (
    test_region_info.dropna(subset=["MPI"])
    .groupby(["country", "region"], observed=True)
    .agg(
        n_test_loans=("MPI", "count"),
        MPI=("MPI", "first"),
        mean_model_score=("model_score", "mean"),
    )
    .reset_index()
)
priority = priority[priority["n_test_loans"] >= 10].copy()
priority["poverty_rank"] = priority["MPI"].rank(pct=True)
priority["risk_rank"] = priority["mean_model_score"].rank(pct=True, ascending=False)
priority["priority_index"] = priority["poverty_rank"] * priority["risk_rank"]
priority = priority.sort_values(["priority_index", "MPI"], ascending=False)

print(f"Regions in priority table (at least 10 test loans with a known MPI): {len(priority)}")
priority.head(15)
""")

md("""### 9.1 Business impact: how much of the funding shortfall does the score point at?

A loan's shortfall is the part of its requested amount that lenders never funded, in US
dollars. If Kiva reviewed only the loans the model ranks riskiest in the test period (for
example to feature them or add matching funds), what share of the total shortfall would
those reviews cover? `src/impact.py` does the arithmetic and is covered by the tests.
""")

code("""from impact import funding_shortfall, shortfall_capture

test_loans = df.loc[X_test.index, ["loan_amount", "funded_amount"]]
risk = 1 - y_proba  # higher = more likely to go unfunded
capture = shortfall_capture(risk, test_loans["loan_amount"], test_loans["funded_amount"])
shortfall = funding_shortfall(test_loans["loan_amount"], test_loans["funded_amount"])
total_shortfall = shortfall.sum()
flagged = y_proba < 0.5  # the same default threshold as the classification report in 7.1
flagged_cover = shortfall[flagged].sum()

print(f"Test period: {len(test_loans):,} loans, total funding shortfall ${total_shortfall:,.0f}")
print(f"Loans flagged at-risk at the default threshold: {flagged.sum():,} "
      f"({flagged.mean():.1%}), covering ${flagged_cover:,.0f} ({flagged_cover / total_shortfall:.1%}) of the shortfall")
print(capture.to_string(index=False, formatters={
    "review_share": "{:.0%}".format, "shortfall_covered_usd": "${:,.0f}".format,
    "shortfall_covered_pct": "{:.1%}".format, "unfunded_loans_covered_pct": "{:.1%}".format}))

shares = np.linspace(0, 1, 101)
curve = shortfall_capture(risk, test_loans["loan_amount"], test_loans["funded_amount"], review_shares=shares[1:])
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot([0] + list(curve["review_share"] * 100), [0] + list(curve["shortfall_covered_pct"] * 100),
        color="#2b6cb0", lw=2, label=f"{best_model_name} ranking")
ax.plot([0, 100], [0, 100], color="#a0aec0", ls="--", lw=1, label="Random review")
for _, r in capture.iterrows():
    ax.annotate(f"{r['shortfall_covered_pct']:.0%}", (r["review_share"] * 100, r["shortfall_covered_pct"] * 100),
                textcoords="offset points", xytext=(6, -12), fontsize=9)
ax.set_xlabel("Share of test-period loans reviewed, riskiest first (%)")
ax.set_ylabel("Share of funding shortfall covered (%)")
ax.set_title(f"Funding shortfall covered by reviewing the riskiest loans (total ${total_shortfall / 1e6:.1f}M)",
             loc="left", fontsize=11)
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("figs/shortfall_capture.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""---

## 10. Limitations

- **Censoring is handled, not eliminated.** Loans posted within 60 days of the
  snapshot are excluded because their outcome was not yet known (the table in 6.1
  shows the funded rate by posting age). A few loans older than that may still have
  been fundraising, so the at-risk class can contain a small residue of them.
- **One out-of-time test window.** The model is chosen on a validation slice at the
  end of the training period and scored once on the most recent 20% of settled
  loans. Walk-forward windows would show how stable the test score is.
- **Trend and season.** Month of posting can stand in for a platform-wide trend
  under a time split. The time-trend candidate in Section 7 tests this on the
  validation slice; a tree model cannot extrapolate a trend beyond the training
  period, so the trend column can only hold the latest level steady.
- **Scores are not probabilities.** Balanced class weights move every score towards
  0.5. The scores rank loans and regions, which is how they are used here; a
  probability reading would need calibration on later data.
- **Days-to-fund is truncated in the test period.** Only loans that were funded by
  the snapshot have a days-to-fund value, and the most recent loans had the least
  time to fund slowly, so the test target is biased towards short fundraising times.
  The regression and its baseline share that bias.
- **Misplaced map points are dropped.** Regions far outside their country's spread
  in the upstream MPI file (counted in Section 5) are left off the map; their MPI
  values are still used. Some region names in that file are also mis-encoded
  upstream (Section 5).
- **MPI join coverage is low.** Section 5 prints the share of loans that match a
  region-level MPI score; most do not, most likely because `kiva_loans.csv`'s
  free-text `region` field (entered inconsistently by field partners) does not
  standardize against `kiva_mpi_region_locations.csv`'s `region` field well enough
  for an exact string join. Every MPI-dependent result in this notebook, the
  Section 5 map and correlation, the `MPI` feature in the Section 7 model, and the
  Section 9 priority table, describes only that small, non-random subset (skewed
  towards regions with cleanly matching names). The priority table illustrates the
  method; it is not a region-targeting list for the rest of the loan volume.
- **The MPI correlation is an ecological one.** The Section 5 correlation is
  computed across regions, between a region's MPI and its share of loans fully
  funded. A region-level association says nothing about whether any individual
  poorer borrower is more or less likely to be funded; assuming it does is the
  ecological fallacy.
- **Correlational, not causal.** Nothing here establishes that any feature *causes*
  funding success or delay, only that it is predictive or associated.
- **English-only text mining.** TF-IDF was fit on the raw `use` field without
  language detection; non-English descriptions contribute noise to the term list.
""")

# =====================================================================
# 11. RESULTS FILE
# =====================================================================
md("""---

## 11. Results file

Every headline number above is written to `outputs/results.json`, with the package
versions, the git commit of the code and the run time, so the README can be checked
against it mechanically (`tests/test_readme_numbers.py`).
""")

code("""import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def peak_memory_gb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak is None:
            import resource  # Linux and macOS
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return round(peak / 1e9, 1)
    except Exception:
        return None


# Changes to the code since the recorded commit; the generated notebook, figures and
# this file are outputs of the run and are left out of the check.
uncommitted = git("status", "--porcelain", "--", ".", ":(exclude)*.ipynb", ":(exclude)figs", ":(exclude)outputs")
chosen = report["0"]

results = {
    "provenance": {
        "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "runtime_minutes": round((time.time() - RUN_STARTED) / 60),
        "peak_memory_gb": peak_memory_gb(),
        "git_commit": git("rev-parse", "HEAD"),
        "uncommitted_code_changes": None if uncommitted is None else bool(uncommitted),
        "python": platform.python_version(),
        "packages": {p: package_version(p) for p in (
            "numpy", "pandas", "scikit-learn", "lightgbm", "shap", "matplotlib", "seaborn", "plotly", "kaleido")},
    },
    "data": {
        "loans": len(df),
        "snapshot_end": str(snapshot_end.date()),
        "censor_days": CENSOR_DAYS,
        "recent_loans_excluded": int((~settled).sum()),
        "recent_loans_excluded_share": float((~settled).mean()),
        "settled_loans": int(settled.sum()),
        "settled_not_funded_share": float(1 - df.loc[settled, "fully_funded"].mean()),
        "funded_rate_by_age_days": {str(k): float(v) for k, v in censoring["funded_rate"].items()},
        "modeling_rows": len(model_df),
        "mpi_matched_loans": int(df["MPI"].notna().sum()),
        "mpi_coverage": float(coverage),
        "mpi_located_regions": n_located,
        "mpi_misplaced_coordinates": n_misplaced,
    },
    "split": {
        "train_share": TRAIN_SHARE,
        "fit_slice_end": str(val_cut.date()),
        "train_end": str(split_date.date()),
        "fit_loans": int(is_fit.sum()),
        "validation_loans": int(is_val.sum()),
        "train_loans": int(is_train.sum()),
        "test_loans": int((~is_train).sum()),
        "test_not_funded_loans": int((y_test == 0).sum()),
        "not_funded_share": {"fit": float(1 - y[is_fit].mean()), "validation": float(1 - y[is_val].mean()),
                             "test": float(1 - y_test.mean())},
        "feature_columns": len(ALL_COLUMNS),
    },
    "classification": {
        "candidates": {name: {"validation_pr_auc_at_risk": float(validation_pr_auc[name]),
                              "test_pr_auc_at_risk": float(test_pr_auc[name]),
                              "test_pr_auc_funded": float(test_pr_auc_majority[name])}
                       for name in CANDIDATES},
        "chosen_model": best_model_name,
        "chosen_on": "validation PR-AUC, at-risk class",
        "test_pr_auc_at_risk": float(pr_auc_minority),
        "test_pr_auc_funded": float(pr_auc_majority),
        "test_roc_auc": float(roc_auc),
        "test_funded_share": float(y_test.mean()),
        "at_risk_recall_at_0_5": float(chosen["recall"]),
        "at_risk_precision_at_0_5": float(chosen["precision"]),
        "top12_mean_abs_shap": {k: float(v) for k, v in mean_abs_shap.head(12).items()},
    },
    "regional_poverty": {
        "regions_min_20_settled_loans": len(region_summary),
        "settled_loans_in_regions": int(region_summary["n_loans"].sum()),
        "settled_loans_in_regions_share": float(region_loan_share),
        "pearson_mpi_vs_share_funded": float(mpi_corr),
        "spearman_mpi_vs_share_funded": float(mpi_corr_rank),
    },
    "days_to_fund": {
        "rows": len(X_reg),
        "test_rows": len(yr_test),
        "mean_days": float(funded_only["days_to_fund"].mean()),
        "std_days": float(funded_only["days_to_fund"].std()),
        "mae_model_days": float(reg_mae),
        "mae_baseline_days": float(baseline_mae),
        "baseline_train_median_days": baseline_days,
        "r2_model": float(reg_r2),
    },
    "priority_regions": {
        "regions": len(priority),
        "top5": [{"country": r.country, "region": r.region, "n_test_loans": int(r.n_test_loans),
                  "MPI": float(r.MPI), "mean_model_score": float(r.mean_model_score),
                  "priority_index": float(r.priority_index)} for r in priority.head(5).itertuples()],
    },
    "shortfall": {
        "test_total_usd": float(total_shortfall),
        "flagged_loans": int(flagged.sum()),
        "flagged_share": float(flagged.mean()),
        "flagged_shortfall_usd": float(flagged_cover),
        "flagged_shortfall_share": float(flagged_cover / total_shortfall),
        "capture": capture.to_dict(orient="records"),
    },
}

Path("outputs").mkdir(exist_ok=True)
with open("outputs/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Wrote outputs/results.json")
print(json.dumps(results["provenance"], indent=2))
""")

# =====================================================================
for i, cell in enumerate(cells):
    cell["id"] = f"cell-{i:02d}"  # stable ids (required by nbformat 4.5)

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
