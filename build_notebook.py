"""Generate Kiva_Loans_Microfinance_Analytics.ipynb.

This script is the source of the notebook: edit it, regenerate, and execute the
notebook. Never edit the .ipynb by hand.

The notebook is written to be followed and recreated step by step. It is
organised around the project's two questions (which loans are at risk, and does
that risk fall on the poorest regions), with each section broken into small
numbered steps of one short code cell each, and "Check" assertions stop the run
if a step goes wrong.
The section numbers 1 to 11 are referenced from docs/METHODOLOGY.md, so keep
them stable."""
import json
from pathlib import Path

cells = []

def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})

def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})

# =====================================================================
# TITLE
# =====================================================================
md("""
# Kiva Loans Microfinance Analytics

**Which loans are at risk of not getting fully funded, and does that risk fall
hardest on the poorest regions?**

This notebook analyses 671,205 real microloans from Kiva's public dataset
(Kaggle's "Data Science for Good: Kiva Crowdfunding"). Kiva posts each loan
request on its website and individual lenders fund it in small amounts; a loan
that does not reach its target in time is not fully funded.

**How this notebook is organised.** This is an analysis driven by two
business questions, so it is organised around them rather than around a single
model:

- **Part 1, what the data can support** (sections 1 to 6): load and check the
  loans, explore them, mine the loan descriptions, and join a poverty index by
  region. Two limits surface here that shape every later answer: recent loans
  were still fundraising when the data was taken, and only a small share of
  loans can be matched to a poverty score. Section 5.1 gives the first,
  descriptive answer to question 2, and section 6 builds leakage-free features
  with a time-based split.
- **Part 2, question 1: which loans are at risk?** (sections 7 and 8): a
  funding-risk model chosen against a baseline on a validation period, scored
  once on later loans and explained with SHAP; then how long funded loans take.
- **Part 3, question 2 and what to do about it** (section 9): where poverty and
  funding risk coincide, and how much of the funding shortfall a review list
  built from the model would cover.
- **Part 4, limits and record** (sections 10 and 11): what the results cannot
  show, and a results file every README number is checked against.

Each step is one short code cell with an explanation of what it does and what
its output shows. Cells containing **Check** assertions stop the notebook at the
step that went wrong instead of carrying a wrong number forward. All findings are correlational, not causal:
this is a single-snapshot dataset.
""")

# =====================================================================
# PART 1
# =====================================================================
md("""
---
# Part 1: What the data can support

## 1. Setup

**Step 1.1:** Import the libraries and fix the random seed, so every run gives
the same numbers.
""")

code("""
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RUN_STARTED = time.time()          # used for the runtime in the results file
RANDOM_STATE = 42                  # one seed for every model below
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", context="notebook")
""")

md("""
**Step 1.2:** Import the project's own functions from `src/features.py`. The
feature engineering and the leakage guard live there rather than in this
notebook so that `tests/` can check them; the notebook and the tests therefore
run the same code.
""")

code("""
sys.path.insert(0, str(Path.cwd() / "src"))
from features import (
    CATEGORICAL_COLUMNS,     # columns one-hot encoded before modelling
    LEAKY_COLUMNS,           # columns that only exist after a loan is funded
    POSTING_TIME_FEATURES,   # everything known when a loan is posted
    add_borrower_features,   # parses borrower_genders into counts
    assert_no_leakage,       # raises if a leaky column reaches the model
    mark_fully_funded,       # the target: funded_amount >= loan_amount
)
""")

md("""
**Step 1.3:** Point the notebook at the data. The Kiva CSVs (about 200 MB) are
not stored in the repository; put them in `./data` or set `KIVA_DATA_DIR`.
""")

code("""
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
md("""
---
## 2. Data loading

**Step 2.1:** Load `kiva_loans.csv`. Text columns with few distinct values
(sector, country and so on) are read as `category`, which stores each distinct
value once and keeps 671k rows small in memory. The four timestamps are parsed
as dates.
""")

code("""
LOAN_DTYPES = {
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

code("""
# Check: every loan in the file was read, once
assert len(df) == 671_205
""")

md("""
### 2.1 Data-quality checks

**Step 2.2:** Which columns have missing values?
""")

code("""
df.isna().sum().sort_values(ascending=False).head(10)
""")

md("""
`funded_time` is missing for loans that were never fully funded, which is
expected and is what the model predicts. `region` is missing for 56,800 loans,
which limits the poverty join in section 5.

**Step 2.3:** Look for impossible values: duplicate loans, non-positive amounts,
and loans funded above what they asked for.
""")

code("""
print("Duplicate loan ids:", df["id"].duplicated().sum())
print("loan_amount <= 0:", (df["loan_amount"] <= 0).sum())
print("funded_amount > loan_amount (should be 0 or near-0):", (df["funded_amount"] > df["loan_amount"]).sum())
""")

code("""
# Check: no duplicate loans and no zero or negative loan amounts
assert not df["id"].duplicated().any()
assert (df["loan_amount"] > 0).all()
""")

md("""
Two loans closed slightly above their target. They are funded loans, which is
why the target in section 5.1 uses `>=` rather than `==`.
""")

# =====================================================================
# 3. EDA
# =====================================================================
md("""
---
## 3. Exploratory data analysis

**Step 3.1:** How large are the loans? Plot the distribution of `loan_amount`,
once on the raw scale (cut at the 99th percentile so a few very large loans do
not squash the chart) and once on a log scale.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
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
""")

code("""
df["loan_amount"].describe()
""")

md("""
The amounts are right-skewed: the median loan is $500, the mean $842 is pulled
up by a long tail, and the largest is $100,000. On the log scale the shape is
close to symmetric.

### 3.1 Sectors and activities

**Step 3.2:** Which sectors receive the most loans?
""")

code("""
top_sectors = df["sector"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 6))
top_sectors.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Loan count by sector")
ax.set_xlabel("Number of loans")
ax.set_ylabel("Sector")
plt.tight_layout()
plt.show()
top_sectors
""")

md("""
### 3.2 Countries and regions

**Step 3.3:** Which countries receive the most loans?
""")

code("""
top_countries = df["country"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(9, 6))
top_countries.sort_values().plot(kind="barh", ax=ax)
ax.set_title("Loan count by country (top 15)")
ax.set_xlabel("Number of loans")
ax.set_ylabel("Country")
plt.tight_layout()
plt.show()
top_countries
""")

md("""
### 3.3 Borrower gender composition

`borrower_genders` is a comma-separated list with one entry per borrower, since
many Kiva loans are group loans (for example "female, female, male").

**Step 3.4:** Parse it into counts of male and female borrowers per loan, and
the female share. `add_borrower_features` (in `src/features.py`, with its edge
cases tested in `tests/test_features.py`) does this. A loan whose genders cannot
be parsed gets `pct_female = NaN`, not 0: NaN means "we could not tell", while 0
would claim the loan had only male borrowers, and the model reads this column.
""")

code("""
df = add_borrower_features(df)

print(f"Loans with 0 parsed borrowers: {(df['n_borrowers'] == 0).sum():,}")
print(f"Median % female borrowers per loan: {df['pct_female'].median():.2%}")
df[["n_male", "n_female", "n_borrowers", "pct_female"]].describe()
""")

code("""
# Check: the female share is a share, and it is missing exactly where nobody was parsed
assert df["pct_female"].dropna().between(0, 1).all()
assert (df["pct_female"].isna() == (df["n_borrowers"] == 0)).all()
""")

md("""
The 4,221 loans with no parsed borrower are the ones whose `borrower_genders` is
missing (section 2.1). Most loans have a single female borrower, so the median
female share is 100%.

### 3.4 Repayment intervals and loan volume over time

**Step 3.5:** How are loans repaid, and how did volume change over time?
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df["repayment_interval"].value_counts().plot(kind="bar", ax=axes[0])
axes[0].set_title("Repayment interval")
axes[0].set_xlabel("Repayment interval")
axes[0].set_ylabel("Number of loans")

monthly = df.set_index("posted_time").resample("MS").size()   # loans posted per calendar month
monthly.plot(ax=axes[1])
axes[1].set_title("Loans posted per month")
axes[1].set_xlabel("Month posted")
axes[1].set_ylabel("Loans posted per month")
plt.tight_layout()
plt.show()
""")

# =====================================================================
# 4. TEXT MINING
# =====================================================================
md("""
---
## 4. Text mining: what are these loans actually for?

`use` is a free-text field ("to buy seasonal, fresh fruits to sell"). TF-IDF
surfaces the words that characterise these descriptions.

> **What's TF-IDF?** Term frequency times inverse document frequency. A word
> scores high in a description when it appears there but is rare across all
> descriptions, so filler words that appear everywhere score low.

This section is exploratory only: the vocabulary fitted here, on every loan, is
not used by the model. The model's text features are refitted on the training
period alone in section 6.2.

**Step 4.1:** Fill missing descriptions with an empty string and record each
description's length (a model input).
""")

code("""
df["use"] = df["use"].fillna("")
df["use_len"] = df["use"].str.len().astype("int32")
""")

md("""
**Step 4.2:** Fit a 30-term TF-IDF vocabulary (single words and two-word
phrases, English stop words removed, each term in at least 50 descriptions) and
count how many loans use each term.
""")

code("""
from sklearn.feature_extraction.text import TfidfVectorizer

# Exploratory vocabulary over all loans. Not a model input (see 6.2).
tfidf_eda = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2), min_df=50)
eda_matrix = tfidf_eda.fit_transform(df["use"])
loans_per_term = pd.Series(
    np.asarray((eda_matrix > 0).sum(axis=0)).ravel(),   # number of loans containing each term
    index=tfidf_eda.get_feature_names_out(),
    name="loans_using_term",
)
del eda_matrix

print(f"Top TF-IDF terms: {list(tfidf_eda.get_feature_names_out()[:15])}")
loans_per_term.sort_values(ascending=False).head(10)
""")

md("""
### 4.1 Top terms by sector

**Step 4.3:** Fit a small vocabulary within each of the five largest sectors.
""")

code("""
for sector in df["sector"].value_counts().head(5).index:
    sector_mask = df["sector"] == sector
    sector_tfidf = TfidfVectorizer(max_features=8, stop_words="english", min_df=10)
    try:
        sector_tfidf.fit(df.loc[sector_mask, "use"])
        print(f"{sector}: {list(sector_tfidf.get_feature_names_out())}")
    except ValueError:
        print(f"{sector}: not enough vocabulary to extract terms")
""")

md("""
Each sector's vocabulary is coherent (fertilizer and seeds in Agriculture,
canned goods in Retail, water filters and solar lights in Personal Use), which
suggests the text carries information a model can use.
""")

# =====================================================================
# 5. GEOSPATIAL
# =====================================================================
md("""
---
## 5. Geospatial analysis: loans against poverty depth

> **What's the MPI?** The Multidimensional Poverty Index scores a region from 0
> to 1 on deprivations in health, education and living standards. Higher means
> poorer.

Each loan's `country` + `region` is joined against Kiva's own region-to-MPI
lookup (`kiva_mpi_region_locations.csv`), a small table with one row per
region. Region names are not perfectly standardised, so the join coverage is
reported rather than assumed.

Some region names in the MPI file are mis-encoded upstream: the file is valid
UTF-8, but it stores "Maranh°o" and "Rond°nia" where the Brazilian states are
Maranhão and Rondônia. The names print as stored rather than being repaired by
hand.

**Step 5.1:** Load the lookup table, keeping one row per country and region.
""")

code("""
mpi = pd.read_csv(DATA_DIR / "kiva_mpi_region_locations.csv")
mpi = mpi[["country", "region", "MPI", "lat", "lon"]].drop_duplicates(subset=["country", "region"])
print(f"MPI lookup: {len(mpi):,} regions, {mpi['MPI'].notna().sum():,} with an MPI score")
""")

md("""
**Step 5.2:** Check the coordinates before drawing a map. A region whose point
lies far from the rest of its country (more than five robust standard
deviations of that country's spread, and at least 10 degrees) is mis-keyed in
the upstream file. Its map point is dropped; its MPI value, which is what the
model uses, is kept.
""")

code("""
by_country = mpi.groupby("country")[["lat", "lon"]]
offset = (mpi[["lat", "lon"]] - by_country.transform("median")).abs()                # distance from the country's median point
spread = by_country.transform(lambda s: (s - s.median()).abs().median()) * 1.4826     # robust standard deviation
limit = np.maximum(10, 5 * spread)
misplaced = (offset["lat"] > limit["lat"]) | (offset["lon"] > limit["lon"])
n_misplaced, n_located = int(misplaced.sum()), int(mpi["lat"].notna().sum())
print(f"Coordinates dropped as misplaced: {n_misplaced} of {n_located} located regions, for example:")
print(mpi.loc[misplaced, ["country", "region", "lat", "lon"]].head(8).to_string(index=False))
mpi.loc[misplaced, ["lat", "lon"]] = np.nan
""")

md("""
Herat in Afghanistan placed in South-East Asia, or Oruro in Bolivia placed in
the Himalayas, are clearly wrong, so dropping these points loses nothing real.

**Step 5.3:** Join the MPI onto the loans and report how many matched.
`validate="many_to_one"` makes pandas raise an error if a region appeared twice
in the lookup, which would silently duplicate loans.
""")

code("""
n_before = len(df)
df = df.merge(mpi, on=["country", "region"], how="left", validate="many_to_one")

coverage = df["MPI"].notna().mean()
print(f"MPI join coverage: {coverage:.1%} of loans matched to a region-level MPI score")
print(f"Loans matched: {df['MPI'].notna().sum():,} / {len(df):,}")
""")

code("""
# Check: the join added columns, not loans
assert len(df) == n_before
""")

md("""
Only 7.6% of loans match. The free-text `region` field in the loans file does
not line up with the lookup's region names well enough for an exact join, so
every MPI result below describes a small, non-random subset (see section 10).

### 5.1 Question 2, first look: funding success against poverty depth, by region

**Step 5.4:** Define the target and the loans whose outcome is known. A loan is
fully funded when it raised at least what it asked for. Loans posted in the
last 60 days before the data snapshot may still have been fundraising, so their
outcome is not yet known; section 6.1 measures this.
""")

code("""
CENSOR_DAYS = 60
df["fully_funded"] = mark_fully_funded(df)          # 1 if funded_amount >= loan_amount
snapshot_end = df["posted_time"].max()              # the last posting date in the data
age_days = (snapshot_end - df["posted_time"]).dt.days
settled = age_days > CENSOR_DAYS                    # outcome known: posted more than 60 days before the snapshot
print(f"Settled loans: {settled.sum():,} of {len(df):,}")
""")

md("""
**Step 5.5:** Summarise each region with a known MPI and at least 20 settled
loans: number of loans, total amount, share fully funded, and MPI.
""")

code("""
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
region_summary.sort_values("MPI", ascending=False).head(10)
""")

md("""
**Step 5.6:** Measure the association between poverty and funding success
across these regions, with both the Pearson correlation and the Spearman rank
correlation (which is not pulled around by a few extreme regions).
""")

code("""
mpi_corr = region_summary["MPI"].corr(region_summary["pct_fully_funded"])
mpi_corr_rank = region_summary["MPI"].corr(region_summary["pct_fully_funded"], method="spearman")
region_loan_share = region_summary["n_loans"].sum() / settled.sum()
print(f"Regions with >=20 settled loans and known MPI: {len(region_summary)}")
print(f"Settled loans in those regions: {region_summary['n_loans'].sum():,} ({region_loan_share:.1%} of settled loans)")
print(f"Correlation between MPI and share fully funded (Pearson):  {mpi_corr:.3f}")
print(f"Rank correlation between MPI and share fully funded (Spearman): {mpi_corr_rank:.3f}")
""")

md("""
**Does funding risk fall hardest on the poorest regions?** Not in the data that
can answer it. Across the regions printed above (at least 20 settled loans and a
known MPI), both correlations are positive: in this matched subset, poorer
regions were funded slightly more often, not less. The subset holds only the
share of settled loans printed above, and the correlation is ecological,
measured across regions rather than borrowers (see section 10), so it does not
show that poorer borrowers are favoured.

That answer rests on 7.5% of settled loans, so it is worth testing on more data.
The exact join fails mostly because the two files describe places at different
levels: a loan's `region` is usually a town ("Lahore", "Kisii", "Palo, Leyte"),
while the poverty index is published by province ("Punjab", "Nyanza"). Cleaning
the spelling cannot fix that.

**Step 5.7:** Kiva publishes its own link from loan regions to poverty-index
regions, in `loan_themes_by_region.csv` (column `mpi_region`). Use it to give
each loan the poverty score of its province. Where Kiva's field partners link one
loan region to more than one province, the most common link is used.
""")

code("""
themes = pd.read_csv(DATA_DIR / "loan_themes_by_region.csv", usecols=["country", "region", "mpi_region"])
link = (themes.dropna(subset=["mpi_region"])
        .groupby(["country", "region"])["mpi_region"]
        .agg(lambda s: s.mode().iloc[0])           # the most common province for each loan region
        .reset_index())
mpi_by_location = (pd.read_csv(DATA_DIR / "kiva_mpi_region_locations.csv")
                   .dropna(subset=["MPI", "LocationName"])
                   .drop_duplicates("LocationName")
                   .set_index("LocationName")["MPI"])
link["MPI_linked"] = link["mpi_region"].map(mpi_by_location)   # "Punjab, Pakistan" -> its MPI

n_before = len(df)
df = df.merge(link[["country", "region", "mpi_region", "MPI_linked"]], on=["country", "region"],
              how="left", validate="many_to_one")
assert len(df) == n_before
linked_coverage = df["MPI_linked"].notna().mean()
print(f"loans with a poverty score: exact join {coverage:.1%}, through Kiva's link {linked_coverage:.1%}")
""")

md("""
Coverage rises from about one loan in thirteen to about seven in ten. The link is
approximate: Kiva assigned it by nearest point, and a few loan regions land in a
neighbouring province (Battambang, in Cambodia, is linked to Banteay Mean Chey).
The model in section 7 keeps the exact-join score; this wider link is used to
re-test question 2 and, in section 7.3, as a challenger input.

**Step 5.8:** Re-test question 2 by province, with at least 20 settled loans each.
""")

code("""
province_summary = (
    df[settled].dropna(subset=["MPI_linked"])
    .groupby(["country", "mpi_region"], observed=True)
    .agg(n_loans=("id", "count"), pct_fully_funded=("fully_funded", "mean"), MPI=("MPI_linked", "first"))
    .reset_index()
)
province_summary = province_summary[province_summary["n_loans"] >= 20]
linked_corr = province_summary["MPI"].corr(province_summary["pct_fully_funded"])
linked_corr_rank = province_summary["MPI"].corr(province_summary["pct_fully_funded"], method="spearman")
linked_share = province_summary["n_loans"].sum() / settled.sum()
print(f"provinces: {len(province_summary)} in {province_summary['country'].nunique()} countries, "
      f"{linked_share:.1%} of settled loans")
print(f"correlation between poverty and share fully funded: Pearson {linked_corr:.3f}, Spearman {linked_corr_rank:.3f}")
""")

md("""
**Step 5.9:** A correlation summarises a straight line. Look at the shape instead:
split settled loans into five equal groups by the poverty score of their province,
from least to most poor, and compare funding rates.
""")

code("""
poverty_fifth = pd.qcut(df.loc[settled, "MPI_linked"], 5)
funded_by_fifth = df[settled].groupby(poverty_fifth, observed=True)["fully_funded"].agg(
    share_fully_funded="mean", loans="size")
funded_by_fifth
""")

md("""
**The wider answer to question 2.** On about 240 provinces holding about 71% of
settled loans, the straight-line relationship between poverty and funding is weak
(correlations of about 0.1). The shape is not a line: loans from the least poor
fifth and the poorest fifth are both funded less often (about 92% and 93%) than
loans from the middle three fifths (about 96% to 97%). So funding risk does not
fall hardest on the poorest regions alone; it is somewhat higher at both ends.
This is still measured by province, not by borrower, so the caveats in section 10
apply, but it rests on most of the data rather than a small, skewed slice.

### 5.2 Map: loan volume and funding success against poverty depth

**Step 5.10:** Map each region, sized by loan volume and coloured by funding
success.
""")

code("""
import plotly.express as px
from IPython.display import Image, display

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

# A static PNG is shown so the map renders on GitHub, which does not display
# interactive Plotly output. Without kaleido, fall back to the interactive figure.
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
md("""
---
## 6. Feature engineering

### 6.1 The leakage check

`funded_time`, `disbursed_time`, `lender_count` and `funded_amount` are
**consequences** of a loan being funded. They do not exist yet when a loan is
posted, so a model that used them to predict "will this loan be funded" would be
reading the answer. A leaked column does not break a model; it improves the
score, so nothing downstream would flag it. That is why the guard below is run,
not just described.

**Step 6.1:** Add three features that are known at posting time: the month and
weekday of posting, and the number of months since the first loan in the data
(a time trend). The trend is offered to the model only as a separate candidate
in section 7, so that the month of posting cannot quietly stand in for a
platform-wide trend.
""")

code("""
df["post_month"] = df["posted_time"].dt.month.astype("int8")
df["post_dow"] = df["posted_time"].dt.dayofweek.astype("int8")
TREND_COLUMN = "months_since_start"
df[TREND_COLUMN] = ((df["posted_time"] - df["posted_time"].min()).dt.days / 30.4375).astype("float32")
""")

md("""
**Step 6.2:** Measure right-censoring. A loan with no `funded_time` when the
snapshot was taken may simply still have been fundraising. Tabulate the funded
rate by how many days before the snapshot the loan was posted.
""")

code("""
age_band = pd.cut(age_days, [-1, 7, 14, 21, 30, 45, 60, 90, 10**6],
                  labels=["0-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61-90", ">90"])
censoring = df.groupby(age_band, observed=True)["fully_funded"].agg(funded_rate="mean", loans="size")
print(f"Snapshot ends {snapshot_end.date()}. Funded rate by days between posting and snapshot:")
print(censoring.to_string(float_format=lambda v: f"{v:.3f}"))
print(f"Excluded {(~settled).sum():,} loans ({(~settled).mean():.1%}) posted within {CENSOR_DAYS} days of the snapshot.")
print(f"Settled loans: {settled.sum():,}, of which {1 - df.loc[settled, 'fully_funded'].mean():.1%} not fully funded")
""")

md("""
Loans posted in the last three weeks show funded rates between 15% and 35%,
against more than 90% for loans older than 60 days. Counting them as "not funded" would
fill the at-risk class with loans that were still raising money, so they are
left out (the `settled` mask from Step 5.4).

**Step 6.3:** Build the modelling table from settled loans and posting-time
columns only, then run the leakage guard.
""")

code("""
model_df = df.loc[settled, POSTING_TIME_FEATURES + [TREND_COLUMN, "fully_funded", "use"]]
model_df = model_df.dropna(subset=["loan_amount", "term_in_months"])

print(f"Modeling rows: {len(model_df):,} (dropped {settled.sum() - len(model_df):,} settled loans with missing core fields)")
print(f"Excluded as leakage: {LEAKY_COLUMNS}")
print(f"Posting-time inputs: {len(POSTING_TIME_FEATURES)} columns and the use text, plus the optional time trend")

# The guard: raises LeakageError naming any post-outcome column in the feature set
assert_no_leakage(model_df.drop(columns=["fully_funded", "use"]).columns)
print("Leakage guard passed: no post-outcome column reached the feature set.")
model_df.head()
""")

md("""
### 6.2 Time-based split, and preprocessing fitted on earlier loans only

Loans are split by posting date, the way a platform would use a model on loans
it has not yet seen:

- **Training period:** the earliest 80% of settled loans. Within it, the
  earliest 80% form the **fit slice** and the latest 20% the **validation
  slice**, which is used to choose the model.
- **Test period:** the most recent 20%, scored once by the chosen model after it
  is refitted on the whole training period.

**Step 6.4:** Find the two cut-off dates and label every loan.
""")

code("""
posted = df.loc[model_df.index, "posted_time"]
TRAIN_SHARE = 0.8
split_date = posted.quantile(TRAIN_SHARE)             # end of the training period
is_train = posted <= split_date
val_cut = posted[is_train].quantile(TRAIN_SHARE)      # end of the fit slice
is_fit = is_train & (posted <= val_cut)
is_val = is_train & (posted > val_cut)
print(f"Fit slice:        posted up to {val_cut.date()} ({is_fit.sum():,} loans)")
print(f"Validation slice: posted after it, up to {split_date.date()} ({is_val.sum():,} loans)")
print(f"Test period:      posted after {split_date.date()} ({(~is_train).sum():,} loans)")
""")

code("""
# Check: every loan is in exactly one slice, and the slices follow each other in time
assert (is_fit.astype(int) + is_val.astype(int) + (~is_train).astype(int) == 1).all()
assert posted[is_fit].max() <= posted[is_val].min() and posted[is_val].max() <= posted[~is_train].min()
""")

md("""
**Step 6.5:** Write the function that turns `model_df` into a numeric feature
matrix. Anything learned from data (the text vocabulary, the medians used to
fill missing values) is learned from `fit_rows` only, so no stage sees text or
medians from the loans it is scored on. A missing-poverty flag (`MPI_missing`)
is added before the fill, so the model can tell "median poverty" from "no MPI
match".
""")

code("""
def build_matrix(fit_rows):
    # 1. Text: learn a 30-term vocabulary from fit_rows, mark which terms each loan's description contains
    vec = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2), min_df=50)
    vec.fit(model_df.loc[fit_rows, "use"])
    terms = pd.DataFrame(
        (vec.transform(model_df["use"]) > 0).toarray().astype("int8"),
        columns=[f"use_tfidf_{t.replace(' ', '_')}" for t in vec.get_feature_names_out()],
        index=model_df.index,
    )
    # 2. Categories: one 0/1 column per sector, activity, country and repayment interval
    encoded = pd.get_dummies(pd.concat([model_df.drop(columns=["use", "fully_funded"]), terms], axis=1),
                             columns=CATEGORICAL_COLUMNS, drop_first=True)
    # 3. Missing values: flag a missing MPI, then fill MPI and pct_female with fit_rows medians
    encoded["MPI_missing"] = encoded["MPI"].isna().astype("int8")
    medians = {c: encoded.loc[fit_rows, c].median() for c in ("MPI", "pct_female")}
    return encoded.fillna(medians), vec, medians
""")

md("""
**Step 6.6:** Build two matrices: one learned from the fit slice, for choosing
the model, and one learned from the whole training period, for the final model.
Then run the leakage guard again: one-hot encoding creates new column names, and
a suffixed column such as `funded_amount_bucket` would get past a check that ran
only before encoding.
""")

code("""
y = model_df["fully_funded"]
X_sel, _, _ = build_matrix(is_fit)                      # for model selection
X, tfidf_train, train_medians = build_matrix(is_train)  # for the final model
ALL_COLUMNS = list(X.columns)


def model_columns(frame, with_trend):
    # The two matrices can hold different text terms, so columns are chosen per matrix.
    return [c for c in frame.columns if with_trend or c != TREND_COLUMN]


assert_no_leakage(ALL_COLUMNS)

print(f"X shape: {X.shape} ({len(model_columns(X, False))} base features, plus {TREND_COLUMN})")
for label, rows in (("fit", is_fit), ("validation", is_val), ("test", ~is_train)):
    print(f"Not fully funded, {label}: {1 - y[rows].mean():.1%}")
""")

md("""
Between about 5% and 10% of loans go unfunded in each slice, so the at-risk
class is the minority. That shapes the choice of metric in section 7.
""")

# =====================================================================
# PART 2
# =====================================================================
md("""
---
# Part 2: Question 1, which loans are at risk?

## 7. Funding-risk model

The question is *which loans are at risk of not getting fully funded*, so what
matters is performance on the minority "not funded" class (label 0).

> **What's PR-AUC?** Rank the loans from most to least at risk and walk down the
> list. Precision is the share of loans flagged so far that really went
> unfunded; recall is the share of all unfunded loans flagged so far. PR-AUC
> (average precision) summarises that trade-off in one number. A model that
> ranks at random scores the share of unfunded loans, so that share is the
> floor.

`average_precision_score` scores the positive label (1 = funded) by default,
which would answer the wrong question and look good regardless (its floor is
about 95%). Both are reported:

- **PR-AUC (at-risk, minority class), the headline:**
  `average_precision_score(1 - y_test, 1 - y_proba)`, which flips both the
  labels and the scores so that "not funded" is the class being found
- **PR-AUC (funded, majority class):** `average_precision_score(y_test, y_proba)`

**Step 7.1:** Write the headline metric, and set the baseline: a score that
ranks every loan the same.
""")

code("""
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score


def at_risk_pr_auc(y_true, funded_score):
    # PR-AUC for the not-funded class: flip the labels and the scores
    return average_precision_score(1 - y_true, 1 - funded_score)


# Baseline: the same score for every validation loan, so no loan is ranked above another
baseline_val = at_risk_pr_auc(y[is_val], np.full(is_val.sum(), 0.5))
print(f"Baseline validation PR-AUC (at-risk): {baseline_val:.4f}  (= the share not funded)")
""")

md("""
Any model below has to beat that number to be worth anything.

**Step 7.2:** Define four candidate models: logistic regression, a random
forest, LightGBM (gradient-boosted trees), and the same LightGBM given the time
trend. All four use balanced class weights, so the rare unfunded loans count as
much as the funded ones during fitting. Their outputs therefore rank loans by
risk but are not calibrated probabilities; they are called scores below.
""")

code("""
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_lgbm():
    return lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                              class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1)


# name: (function that builds an unfitted model, whether it gets the time trend)
CANDIDATES = {
    "Logistic Regression": (lambda: make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)), False),
    "Random Forest": (lambda: RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE), False),
    "LightGBM": (make_lgbm, False),
    "LightGBM + time trend": (make_lgbm, True),
}
""")

md("""
**Step 7.3:** Choose the model on the validation slice only. Fit each candidate
on the fit slice, score the validation slice, and keep the highest at-risk
PR-AUC. The test period is not touched.
""")

code("""
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

code("""
# Check: every candidate beats the constant-score baseline on validation
assert min(validation_pr_auc.values()) > baseline_val
""")

md("""
**Step 7.4:** Refit every candidate on the whole training period and score the
test period once. The choice above is already fixed; the other rows are shown
for reference and play no part in it.
""")

code("""
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
        best_model, y_proba = m, proba        # keep the chosen model and its test scores
del m

model_table = pd.DataFrame({
    "validation PR-AUC (at-risk)": validation_pr_auc,
    "test PR-AUC (at-risk)": test_pr_auc,
    "test PR-AUC (funded)": test_pr_auc_majority,
}).sort_values("validation PR-AUC (at-risk)", ascending=False)
print(model_table.to_string(float_format=lambda v: f"{v:.4f}"))
""")

md("""
The time-trend LightGBM was chosen on validation. On the test period the plain
LightGBM scores slightly higher, but switching to it now would be choosing on
the test set, which would make the test score optimistic. The chosen model is
kept, and both numbers are reported.

### 7.1 Chosen model: full evaluation on the test period

**Step 7.5:** Score the chosen model on the test period, against both floors.
""")

code("""
pr_auc_majority = average_precision_score(y_test, y_proba)
pr_auc_minority = at_risk_pr_auc(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
print(f"Chosen model (on validation PR-AUC): {best_model_name}")
print(f"PR-AUC (at-risk, minority class) [headline]: {pr_auc_minority:.4f}")
print(f"PR-AUC (funded, majority class):             {pr_auc_majority:.4f}")
print(f"Test-set funded share (majority-class PR-AUC floor): {y_test.mean():.3f}")
print(f"Test-set not-funded share (minority-class PR-AUC floor): {1 - y_test.mean():.3f}")
print(f"ROC-AUC: {roc_auc:.4f}")
""")

md("""
The headline PR-AUC is about eight times its floor of 0.046. The funded-class
number looks far better but barely clears its own floor of 0.954, which is why
it is not the headline.

**Step 7.6:** Look at precision and recall at the default cut-off of 0.5.
""")

code("""
report = classification_report(y_test, (y_proba >= 0.5).astype(int), output_dict=True)
print(classification_report(y_test, (y_proba >= 0.5).astype(int)))
""")

md("""
For class 0 (not funded), recall is the share of unfunded test loans the model
flags, and precision the share of flagged loans that really went unfunded. A
high recall with a low precision fits a review queue: most at-risk loans are
caught, at the cost of reviewing many that would have been funded anyway.

### 7.2 SHAP explainability

> **What's SHAP?** For one loan, SHAP splits the model's score into a
> contribution from each feature, measured against the average loan. Averaging
> the size of those contributions over many loans ranks the features by how much
> they move the model.

**Step 7.7:** Compute SHAP values for 1,000 test loans.
""")

code("""
import warnings

# SHAP warns about missing progress-bar widgets and LightGBM's output format; neither affects the values.
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
# Different SHAP versions return the values in different shapes; keep the "funded" class as a 2-D array
if isinstance(shap_values, list):
    shap_values = shap_values[1]
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[..., 1]

mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=best_cols).sort_values(ascending=False)
print(f"Top 12 features by mean |SHAP| ({best_model_name}, 1,000 test loans, log-odds of being funded):")
print(mean_abs_shap.head(12).to_string(float_format=lambda v: f"{v:.4f}"))
""")

md("""
**Step 7.8:** Plot the ranking.
""")

code("""
shap.summary_plot(shap_values, sample, plot_type="bar", max_display=12, show=False)
plt.title(f"Top features by mean |SHAP| ({best_model_name})")
plt.xlabel("Mean |SHAP value| (impact on log-odds of being funded)")
plt.tight_layout()
plt.savefig("figs/shap_summary.png", dpi=150, bbox_inches="tight")
plt.show()
""")

md("""
### 7.3 Two follow-up tests

The published score (0.374) comes from one test window. Two questions follow.

**Step 7.9: Is the score stable over time?** Walk forward through the settled
loans: train on everything posted before a cut-off, score the next tenth of loans,
move the cut-off on, and repeat four times. The model specification is the one
chosen on the validation slice, and every step learns its text vocabulary and
medians from its own training loans only.
""")

code("""
cuts = posted.quantile([0.6, 0.7, 0.8, 0.9, 1.0])
make_model, with_trend = CANDIDATES[best_model_name]
wf_rows = []
for start, end in zip(cuts.iloc[:-1], cuts.iloc[1:]):
    train_rows = posted <= start
    test_rows = (posted > start) & (posted <= end)
    X_wf, _, _ = build_matrix(train_rows)                    # vocabulary and medians from train_rows only
    cols = model_columns(X_wf, with_trend)
    m = make_model().fit(X_wf.loc[train_rows, cols], y[train_rows])
    scores = m.predict_proba(X_wf.loc[test_rows, cols])[:, 1]
    floor = 1 - y[test_rows].mean()                          # PR-AUC of a model that ranks at random
    wf_rows.append({"test_window": f"{start.date()} to {end.date()}", "train_loans": int(train_rows.sum()),
                    "test_loans": int(test_rows.sum()), "not_funded_share": floor,
                    "pr_auc_at_risk": at_risk_pr_auc(y[test_rows], scores)})
del X_wf, m
walk_forward = pd.DataFrame(wf_rows)
walk_forward["lift_over_random"] = walk_forward["pr_auc_at_risk"] / walk_forward["not_funded_share"]
walk_forward.round(3)
""")

md("""
Across the four windows the at-risk PR-AUC runs from about 0.39 to 0.50, and in every window it is between
about 5 and 14 times what random ranking would score. The raw number moves partly because the share of
unfunded loans moves (from about 8% down to about 3%), which is why the lift over random is shown next to it.
The published 0.374 sits at the low end of this range: it comes from one model trained up to October 2016 and
scored on the last two windows together, whereas the walk-forward refits before each window. The headline is
therefore a cautious figure, and the model keeps its usefulness as time moves on.

**Step 7.10: Does the wider poverty score help the model?** Section 5 linked about
71% of loans to a province poverty score, against 7.6% for the exact join the
model uses. Swap the wider score in, refit the chosen model on the fit slice, and
compare on the validation slice. The rule, set before running it: adopt the wider
score only if validation PR-AUC (at-risk) improves by at least 0.005.
""")

code("""
def with_linked_mpi(frame, fit_rows):
    # Replace the exact-join MPI (and its missing flag) with the linked province score
    out = frame.copy()
    linked = df.loc[model_df.index, "MPI_linked"]
    out["MPI_missing"] = linked.isna().astype("int8")
    out["MPI"] = linked.fillna(linked[fit_rows].median())
    return out


X_linked = with_linked_mpi(build_matrix(is_fit)[0], is_fit)
cols = model_columns(X_linked, with_trend)
m = make_model().fit(X_linked.loc[is_fit, cols], y[is_fit])
linked_val_pr_auc = at_risk_pr_auc(y[is_val], m.predict_proba(X_linked.loc[is_val, cols])[:, 1])
linked_gain = linked_val_pr_auc - validation_pr_auc[best_model_name]
adopt_linked = linked_gain >= 0.005
del X_linked, m
print(f"validation PR-AUC (at-risk): exact-join MPI {validation_pr_auc[best_model_name]:.4f}, "
      f"linked MPI {linked_val_pr_auc:.4f} (change {linked_gain:+.4f})")
print("decision:", "adopt the linked score" if adopt_linked else "keep the exact-join score (gain below 0.005)")
""")

md("""
The wider poverty score does not help the model: validation PR-AUC moves from 0.4902 to 0.4880, below the
0.005 gain the rule required, so the model keeps the exact-join score and its published results are unchanged.
That fits the answer to question 2: across most of the data, poverty is only weakly related to whether a loan
gets funded, and the model's risk ranking rests mainly on the loan's own terms (Step 7.7).
""")

# =====================================================================
# 8. DAYS TO FUND
# =====================================================================
md("""
---
## 8. How long does it take to get funded?

Among loans that *did* get fully funded, predict the number of days from posting
to full funding from the same posting-time features. There is no separate
leakage question here: the population is already restricted to funded loans,
and the target is a time gap, not the funding outcome itself.

**Step 8.1:** Build the regression table: settled, funded loans, with the text
vocabulary and medians learned from the training period.
""")

code("""
funded_only = df[(df["fully_funded"] == 1) & settled].copy()
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
reg_train = funded_only["posted_time"] <= split_date          # same date split as section 6.2
X_reg = reg_encoded.fillna({c: reg_encoded.loc[reg_train, c].median() for c in ("MPI", "pct_female")})

print(f"Regression rows: {len(X_reg):,}")
print(funded_only['days_to_fund'].describe())
""")

md("""
**Step 8.2:** Set the baseline first: give every test loan the median
days-to-fund of the training period, and measure the mean absolute error (MAE),
the average number of days the prediction is off by.
""")

code("""
from sklearn.metrics import mean_absolute_error, r2_score

Xr_train, Xr_test = X_reg[reg_train], X_reg[~reg_train]
yr_train, yr_test = y_reg[reg_train], y_reg[~reg_train]

baseline_days = float(yr_train.median())
baseline_mae = mean_absolute_error(yr_test, np.full(len(yr_test), baseline_days))
print(f"Test loans: {len(yr_test):,}")
print(f"MAE, naive baseline (train median {baseline_days:.1f} days): {baseline_mae:.2f} days")
""")

md("""
**Step 8.3:** Fit a random forest and compare it with the baseline.
""")

code("""
from sklearn.ensemble import RandomForestRegressor

reg_model = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
reg_model.fit(Xr_train, yr_train)
yr_pred = reg_model.predict(Xr_test)

reg_mae = mean_absolute_error(yr_test, yr_pred)
reg_r2 = r2_score(yr_test, yr_pred)
print(f"MAE, random forest:                        {reg_mae:.2f} days")
print(f"MAE, naive baseline (train median {baseline_days:.1f} days): {baseline_mae:.2f} days")
print(f"R-squared, random forest: {reg_r2:.3f}")
""")

code("""
# Check: the model beats the naive baseline
assert reg_mae < baseline_mae
""")

md("""
The forest is about a day closer than the baseline on average, and explains
about a quarter of the variation (R-squared): most of what decides how fast a
loan funds is not in the information available when it is posted.
""")

# =====================================================================
# PART 3
# =====================================================================
md("""
---
# Part 3: Question 2, and what Kiva could do about it

## 9. Synthesis: where should Kiva focus promotion?

Combine the model's funding-risk score, averaged by region over the test period,
with each region's MPI. The scores are not calibrated probabilities (balanced
class weights move them towards 0.5), so they are not multiplied by MPI.
Instead each region is ranked twice among the regions in the table, on poverty
(higher MPI ranks higher) and on funding risk (lower mean score ranks higher),
and the **relative priority index** is the product of the two percentile ranks:
1 means the poorest and riskiest region in the table. It orders regions; its
scale carries no other meaning.

**Step 9.1:** Average the test-period scores by region, for regions with a known
MPI and at least 10 test loans.
""")

code("""
test_region_info = df.loc[X_test.index, ["country", "region", "MPI"]].copy()
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
print(f"Regions in priority table (at least 10 test loans with a known MPI): {len(priority)}")
""")

md("""
**Step 9.2:** Rank the regions and compute the priority index.
""")

code("""
priority["poverty_rank"] = priority["MPI"].rank(pct=True)                               # 1 = poorest
priority["risk_rank"] = priority["mean_model_score"].rank(pct=True, ascending=False)    # 1 = least likely funded
priority["priority_index"] = priority["poverty_rank"] * priority["risk_rank"]
priority = priority.sort_values(["priority_index", "MPI"], ascending=False)
priority.head(15)
""")

md("""
### 9.1 Business impact: how much of the funding shortfall does the score point at?

A loan's **shortfall** is the part of its requested amount that lenders never
funded, in US dollars. If Kiva reviewed only the loans the model ranks riskiest
in the test period (for example to feature them or add matching funds), what
share of the total shortfall would those reviews cover? `src/impact.py` does the
arithmetic and is covered by the tests.

**Step 9.3:** Compute the shortfall covered when the riskiest 5%, 10%, 20% and
30% of test loans are reviewed, and by the loans flagged at the 0.5 cut-off.
""")

code("""
from impact import funding_shortfall, shortfall_capture

test_loans = df.loc[X_test.index, ["loan_amount", "funded_amount"]]
risk = 1 - y_proba  # higher = more likely to go unfunded
capture = shortfall_capture(risk, test_loans["loan_amount"], test_loans["funded_amount"])
shortfall = funding_shortfall(test_loans["loan_amount"], test_loans["funded_amount"])   # loan_amount - funded_amount, never below 0
total_shortfall = shortfall.sum()
flagged = y_proba < 0.5  # the same default threshold as the classification report in 7.1
flagged_cover = shortfall[flagged].sum()

print(f"Test period: {len(test_loans):,} loans, total funding shortfall ${total_shortfall:,.0f}")
print(f"Loans flagged at-risk at the default threshold: {flagged.sum():,} "
      f"({flagged.mean():.1%}), covering ${flagged_cover:,.0f} ({flagged_cover / total_shortfall:.1%}) of the shortfall")
print(capture.to_string(index=False, formatters={
    "review_share": "{:.0%}".format, "shortfall_covered_usd": "${:,.0f}".format,
    "shortfall_covered_pct": "{:.1%}".format, "unfunded_loans_covered_pct": "{:.1%}".format}))
""")

md("""
**Step 9.4:** Plot the shortfall covered against the share of loans reviewed,
next to what a random review would cover.
""")

code("""
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

md("""
Reviewing the riskiest 10% of loans covers about three quarters of the
shortfall, where a random 10% would cover about 10%, because the model
concentrates the unfunded dollars at the top of its ranking. That is the practical use of the score: a short, ordered review list.
""")

# =====================================================================
# 10. LIMITATIONS
# =====================================================================
md("""
---
# Part 4: Limits and record

## 10. Limitations

- **Censoring is handled, not eliminated.** Loans posted within 60 days of the
  snapshot are excluded because their outcome was not yet known (the table in 6.1
  shows the funded rate by posting age). A few loans older than that may still have
  been fundraising, so the at-risk class can contain a small residue of them.
- **One out-of-time test window.** The model is chosen on a validation slice at the
  end of the training period and scored once on the most recent 20% of settled
  loans. Walk-forward windows would show how stable the test score is.
- **Trend and season.** Month of posting can stand in for a platform-wide trend
  under a time split. The time-trend candidate in section 7 tests this on the
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
  in the upstream MPI file (counted in section 5) are left off the map; their MPI
  values are still used. Some region names in that file are also mis-encoded
  upstream (section 5).
- **MPI join coverage is low.** Section 5 prints the share of loans that match a
  region-level MPI score; most do not, most likely because `kiva_loans.csv`'s
  free-text `region` field (entered inconsistently by field partners) does not
  standardize against `kiva_mpi_region_locations.csv`'s `region` field well enough
  for an exact string join. Every MPI-dependent result in this notebook, the
  section 5 map and correlation, the `MPI` feature in the section 7 model, and the
  section 9 priority table, describes only that small, non-random subset (skewed
  towards regions with cleanly matching names). The priority table illustrates the
  method; it is not a region-targeting list for the rest of the loan volume.
- **The MPI correlation is an ecological one.** The section 5 correlation is
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
md("""
---
## 11. Results file

Every headline number above is written to `outputs/results.json`, with the
package versions, the git commit of the code and the run time, so the README can
be checked against it mechanically (`tests/test_readme_numbers.py`).

**Step 11.1:** Small helpers for the provenance record: a package's version, a
git query, and the peak memory used by this run.
""")

code("""
import json
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
""")

md("""
**Step 11.2:** Record where the numbers came from: the run date, runtime, code
version and library versions. Changes to the code since the recorded commit are
flagged; the notebook, figures and results file are outputs of the run and are
left out of that check.
""")

code("""
uncommitted = git("status", "--porcelain", "--", ".", ":(exclude)*.ipynb", ":(exclude)figs", ":(exclude)outputs")

results = {}
results["provenance"] = {
    "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "runtime_minutes": round((time.time() - RUN_STARTED) / 60),
    "peak_memory_gb": peak_memory_gb(),
    "git_commit": git("rev-parse", "HEAD"),
    "uncommitted_code_changes": None if uncommitted is None else bool(uncommitted),
    "python": platform.python_version(),
    "packages": {p: package_version(p) for p in (
        "numpy", "pandas", "scikit-learn", "lightgbm", "shap", "matplotlib", "seaborn", "plotly", "kaleido")},
}
""")

md("""
**Step 11.3:** Record the data, censoring and split figures (sections 2, 5 and 6).
""")

code("""
results["data"] = {
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
}
results["split"] = {
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
}
""")

md("""
**Step 11.4:** Record the model results (sections 5.1, 7 and 8).
""")

code("""
chosen = report["0"]   # precision and recall for the not-funded class at the 0.5 cut-off
results["classification"] = {
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
}
results["regional_poverty"] = {
    "regions_min_20_settled_loans": len(region_summary),
    "settled_loans_in_regions": int(region_summary["n_loans"].sum()),
    "settled_loans_in_regions_share": float(region_loan_share),
    "pearson_mpi_vs_share_funded": float(mpi_corr),
    "spearman_mpi_vs_share_funded": float(mpi_corr_rank),
}
results["regional_poverty_linked"] = {
    "loans_with_linked_mpi_share": float(linked_coverage),
    "provinces_min_20_settled_loans": len(province_summary),
    "countries": int(province_summary["country"].nunique()),
    "settled_loans_in_provinces_share": float(linked_share),
    "pearson_mpi_vs_share_funded": float(linked_corr),
    "spearman_mpi_vs_share_funded": float(linked_corr_rank),
    "share_fully_funded_by_poverty_fifth": [float(v) for v in funded_by_fifth["share_fully_funded"]],
}
results["walk_forward"] = {
    "model": best_model_name,
    "windows": [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
                for r in walk_forward.to_dict(orient="records")],
    "pr_auc_at_risk_min": float(walk_forward["pr_auc_at_risk"].min()),
    "pr_auc_at_risk_max": float(walk_forward["pr_auc_at_risk"].max()),
    "lift_over_random_min": float(walk_forward["lift_over_random"].min()),
    "lift_over_random_max": float(walk_forward["lift_over_random"].max()),
}
results["linked_mpi_challenger"] = {
    "rule": "adopt if validation PR-AUC (at-risk) improves by at least 0.005",
    "validation_pr_auc_exact": float(validation_pr_auc[best_model_name]),
    "validation_pr_auc_linked": float(linked_val_pr_auc),
    "gain": float(linked_gain),
    "adopted": bool(adopt_linked),
}
results["days_to_fund"] = {
    "rows": len(X_reg),
    "test_rows": len(yr_test),
    "mean_days": float(funded_only["days_to_fund"].mean()),
    "std_days": float(funded_only["days_to_fund"].std()),
    "mae_model_days": float(reg_mae),
    "mae_baseline_days": float(baseline_mae),
    "baseline_train_median_days": baseline_days,
    "r2_model": float(reg_r2),
}
""")

md("""
**Step 11.5:** Record the synthesis and business impact (section 9), and write
the file.
""")

code("""
results["priority_regions"] = {
    "regions": len(priority),
    "top5": [{"country": r.country, "region": r.region, "n_test_loans": int(r.n_test_loans),
              "MPI": float(r.MPI), "mean_model_score": float(r.mean_model_score),
              "priority_index": float(r.priority_index)} for r in priority.head(5).itertuples()],
}
results["shortfall"] = {
    "test_total_usd": float(total_shortfall),
    "flagged_loans": int(flagged.sum()),
    "flagged_share": float(flagged.mean()),
    "flagged_shortfall_usd": float(flagged_cover),
    "flagged_shortfall_share": float(flagged_cover / total_shortfall),
    "capture": capture.to_dict(orient="records"),
}

Path("outputs").mkdir(exist_ok=True)
with open("outputs/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Wrote outputs/results.json")
print(json.dumps(results["provenance"], indent=2))
""")

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
