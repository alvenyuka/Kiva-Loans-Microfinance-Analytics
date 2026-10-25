"""Generate Kiva_Loans_Microfinance_Analytics.ipynb.

This script is the source of the notebook: edit it, regenerate, and execute the
notebook. Never edit the .ipynb by hand.

    python build_notebook.py
    jupyter nbconvert --to notebook --execute Kiva_Loans_Microfinance_Analytics.ipynb \
        --output Kiva_Loans_Microfinance_Analytics.ipynb --ExecutePreprocessor.timeout=3600

The notebook is a guided build that a junior data scientist can follow cell by
cell. It has three parts (Prepare Data, Build Model, Communicate Results), and
each part is a list of numbered steps. Every step has one short instruction, one
"Concept" line that explains the idea behind it, one small code cell with a
comment above each statement, a printout of what changed, and a "Check your
work" cell whose assertions stop the run if the step went wrong. The expected
values in those assertions come from an executed run.

The tested logic stays in src/: the borrower parsing, the funding target and
the leakage guard (src/features.py) and the shortfall arithmetic
(src/impact.py) are imported, tried on a small example first, then used on the
real data. The notebook writes outputs/results.json, which tests/ checks the
README against.
"""
import nbformat as nbf

NOTEBOOK = "Kiva_Loans_Microfinance_Analytics.ipynb"

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    # Two markdown blocks in a row become one cell: the reading of a result and the next step's heading
    if cells and cells[-1]["cell_type"] == "markdown":
        cells[-1]["source"] = cells[-1]["source"] + "\n\n" + text.strip()
    else:
        cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


def check(text):
    # The assertion cell that closes a step
    code("# Check your work\n" + text.strip())


def step(number, title, instruction, concept_question, concept_answer):
    # A step heading: what to do and why, then the idea behind it in one or two sentences
    md(f"""
### Step {number}: {title}

{instruction.strip()}

**Concept: {concept_question}** {concept_answer.strip()}
""")


# =====================================================================
# OPENING
# =====================================================================
md("""
# Kiva Microloan Funding Risk: which new loans will fall short?

**The business question.** Kiva is a crowdfunding site for microloans. A field partner (a local lender such as a
microfinance institution) posts a small loan for a borrower, and members of the public fund it a few dollars at a
time. Some loans never reach their target, and the borrower may then miss a planting season or lose stock. Can the
loans that will fall short be spotted on the day they are posted, and does that risk fall hardest on the poorest
regions?

**Who acts on the answer.** A platform's partner-support team, or an impact funder's matching-fund desk, can help a
few loans each day (feature them, add matching money, ask the partner to change the terms) but not all of them. This
notebook builds a ranking they can work down from the riskiest end as far as their time allows. Missing a loan that
then falls short costs a borrower money; reviewing a loan that would have been funded anyway costs only staff time.

**The data.** 671,205 Kiva loans posted up to 26 Jul 2017, from Kaggle's "Data Science for Good: Kiva Crowdfunding"
dataset, plus Kiva's regional poverty file and its link from loan regions to poverty-index provinces.

**What you will build.** A cleaned loan table, a guard that keeps after-the-fact columns out of the model, a
constant-score baseline, four candidate models chosen on a validation period, one test on later loans, a check over
time, a check by borrower group, a second model for how long funding takes, and a review list priced in dollars.

**The finished result.**

- The chosen model scores a test PR-AUC of 0.374 for the at-risk class, against a base rate of 4.6%: about eight
  times what a random ranking scores (Step 2.12).
- Reviewing the riskiest 10% of test loans reaches 76% of the $5,797,550 they fell short; that list holds 62% of the
  loans that went unfunded at a precision of 28.7% (Step 3.11).
- The model ranks all-male loans less well than all-female loans (ROC-AUC 0.850 against 0.930), a gap to check before
  the score decides who gets help (Step 2.20).
- Poverty is only weakly related to funding: the least poor and the poorest fifths of loans are both funded a little
  less often than the middle (Step 1.55).

All findings are correlational, not causal: this is one snapshot of the platform. Every number above is printed by a
cell below and written to `outputs/results.json`.
""")

md("""
## How this notebook is organised

| Part | Sections | What happens |
|---|---|---|
| **1. Prepare Data** | 1A to 1F | **Import** (1A): read the three CSV files and clean them with small `wrangle` functions. **Explore** (1B to 1E): the loans, the outcome, the loan descriptions, and poverty by place. **Split** (1F): keep only what is known when a loan is posted, and split the loans by date |
| **2. Build Model** | 2A to 2D | **Baseline** (2A), **Iterate** (2B: choose a model on a validation period), **Evaluate** (2C: score it once on later loans, over time and by group, and test calibration), then a second model for how long funding takes (2D) |
| **3. Communicate Results** | 3A to 3C | What drives the risk (3A), where the funder could act and what it is worth (3B), and the results file (3C), then what you built, its limits and next steps |

Every step has the same shape:

1. **Step N.M: a title** naming the piece of work, one sentence on what to do, and a **Concept** line that explains
   the one new idea the step needs.
2. One small code cell, with a comment above each statement, that does one thing and prints what changed (the
   object's type, its shape or length, and a few rows or the value).
3. A **Check your work** cell of `assert` lines. The expected values come from the recorded run, so if your numbers
   differ, the notebook stops at the step where they first went wrong.
4. Where the output needs reading, a short paragraph says what it shows.
""")

md("""
## How to recreate this notebook

1. **Get the code.** `git clone https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics.git` and open the
   folder.
2. **Set up Python** (3.11 or newer). In a terminal in that folder:

   ```
   python -m venv .venv
   .venv\\Scripts\\activate          (Windows)   or   source .venv/bin/activate   (macOS, Linux)
   pip install -r requirements.txt
   ```

3. **Download the data.** Sign in to Kaggle, open
   [Data Science for Good: Kiva Crowdfunding](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding),
   download it, and put three of its files in a `data` folder next to this notebook:

   ```
   Kiva-Loans-Microfinance-Analytics/
   ├── Kiva_Loans_Microfinance_Analytics.ipynb   written by build_notebook.py
   ├── src/                       features.py and impact.py, imported in Step 1.5
   └── data/
       ├── kiva_loans.csv                    (about 196 MB, one row per loan)
       ├── kiva_mpi_region_locations.csv     (the poverty index by region)
       └── loan_themes_by_region.csv         (Kiva's link from loan regions to poverty-index provinces)
   ```

4. **Run it.** Open the notebook in Jupyter or VS Code and choose **Restart Kernel and Run All**. A full run takes
   about half an hour and needs about 5 GB of memory; the recorded run's time and peak memory are in
   `outputs/results.json` under `provenance`. From a terminal:

   ```
   python build_notebook.py
   jupyter nbconvert --to notebook --execute Kiva_Loans_Microfinance_Analytics.ipynb --output Kiva_Loans_Microfinance_Analytics.ipynb --ExecutePreprocessor.timeout=3600
   ```

5. **Run the tests** (no data needed): `python -m pytest`.

**Checkpoints.** If your run matches these, your setup is right.

| Where | You should see |
|---|---|
| Step 1.19 | 671,205 loans |
| Step 1.36 | 645,083 settled loans, 6.4% of them not fully funded |
| Step 1.49 | 7.6% of loans matched to a poverty score exactly |
| Step 1.61 | 412,853 / 103,213 / 129,017 loans in the fit, validation and test periods |
| Step 1.65 | a feature matrix of shape (645083, 306) |
| Step 2.7 | LightGBM + time trend chosen, validation PR-AUC 0.4902 |
| Step 2.12 | test PR-AUC (at-risk) 0.3742 |
| Step 3.11 | reviewing the riskiest 10% of loans reaches 75.9% of the funding shortfall |
""")

# =====================================================================
# PART 1: PREPARE DATA
# =====================================================================
md("""
---
# Part 1: Prepare Data

## 1A. Import

The three CSV files need the same cleaning every time the notebook runs, so the cleaning lives in **wrangle
functions**: each one takes a file path, reads the file, cleans it and returns a table. Before writing them you will
look at each raw file, and try the project's tested helper functions on a tiny example, so nothing inside the wrangle
functions is new when you reach them.
""")

step("1.1", "Import the standard library and the table and chart libraries",
     "Load the tools for paths, timing, tables, arrays and charts in one place.",
     "why import at the top?",
     "A reader sees every dependency at once, and a missing library fails here rather than halfway through a "
     "half-hour run.")

code("""
# sys: lets Python find the project's own modules in src/
import sys
# time: the clock that measures how long the whole notebook takes
import time
# warnings: hides two harmless start-up messages from SHAP in Step 1.3
import warnings
# Path: builds file paths that work on Windows, macOS and Linux
from pathlib import Path

# LightGBM: the gradient-boosting library behind two of the four candidate models
import lightgbm as lgb
# Matplotlib: the charts
import matplotlib.pyplot as plt
# NumPy: arrays and maths
import numpy as np
# pandas: tables
import pandas as pd
# Plotly Express: the province map in Step 1.57
import plotly.express as px
# seaborn: one consistent chart style
import seaborn as sns
# Image and display: show the saved map as a picture, which GitHub can render
from IPython.display import Image, display

# Report which table and array versions this run uses
print("pandas", pd.__version__, "| numpy", np.__version__, "| lightgbm", lgb.__version__)
""")

check("""
# The notebook relies on pandas 2 or newer (copy-on-write and the string methods used below)
assert int(pd.__version__.split(".")[0]) >= 2, f"pandas 2 or newer is needed, found {pd.__version__}"
# The plotting module loaded under its usual short name
assert plt.__name__ == "matplotlib.pyplot", plt.__name__
""")

step("1.2", "Import the scikit-learn tools",
     "Load the models, the text tool and the metrics the later parts use.",
     "what is scikit-learn?",
     "Python's standard machine-learning library: every model in it is fitted with `.fit()` and used with "
     "`.predict()` or `.predict_proba()`, so swapping one model for another is a one-line change.")

code("""
# The reliability curve used to judge calibration in Step 2.27
from sklearn.calibration import calibration_curve
# Two tree ensembles: a classifier for funding risk and a regressor for days to fund
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# TF-IDF: turns loan descriptions into word columns
from sklearn.feature_extraction.text import TfidfVectorizer
# Isotonic regression: the calibrator tried in Step 2.23
from sklearn.isotonic import IsotonicRegression
# Logistic regression: the simplest of the four candidates
from sklearn.linear_model import LogisticRegression
# The scores used to compare models and forecasts
from sklearn.metrics import (average_precision_score, brier_score_loss, classification_report,
                             mean_absolute_error, precision_recall_curve, r2_score, roc_auc_score)
# make_pipeline chains the scaler and the logistic regression into one model
from sklearn.pipeline import make_pipeline
# StandardScaler puts every feature on the same scale before the logistic regression
from sklearn.preprocessing import StandardScaler

# Confirm two of the imported names resolved to the expected objects
print(type(TfidfVectorizer).__name__, TfidfVectorizer.__name__, "|", average_precision_score.__name__)
""")

check("""
# The text tool is a class you create objects from
assert isinstance(TfidfVectorizer, type), "TfidfVectorizer should be a class"
# The headline metric is a plain function
assert callable(average_precision_score), "average_precision_score should be callable"
""")

step("1.3", "Import SHAP quietly",
     "Hide three start-up warnings that do not affect any value, then import SHAP.",
     "what is SHAP?",
     "A method that splits one prediction into a push from each feature; Part 3 uses it to say what drives the "
     "risk.")

code("""
# SHAP warns that the notebook has no progress-bar widgets
warnings.filterwarnings("ignore", message="IProgress not found")
# SHAP notes the output format of LightGBM's binary classifier
warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")
# SHAP mentions that NumPy's global random generator was seeded
warnings.filterwarnings("ignore", message="The NumPy global RNG was seeded")
# Imported after the filters, so its start-up warnings stay hidden
import shap

# The SHAP version this run uses
print("shap", shap.__version__)
""")

check("""
# SHAP's tree explainer, used in Step 3.2, is available
assert hasattr(shap, "TreeExplainer"), "this SHAP version has no TreeExplainer"
""")

step("1.4", "Fix the random seed and the display settings",
     "Set one seed for every model and one look for every table and chart.",
     "what is a random seed?",
     "Random forests and LightGBM draw random samples while they train. Fixing the seed makes every run draw the "
     "same samples, so the numbers come out the same each time.")

code("""
# The start time, used for the run time in the results file (Step 3.13)
RUN_STARTED = time.time()
# One seed for every model below
RANDOM_STATE = 42
# NumPy's own random draws repeat on every run
np.random.seed(RANDOM_STATE)
# Print up to 30 columns of a table before abbreviating
pd.set_option("display.max_columns", 30)
# Let printed tables run 200 characters wide
pd.set_option("display.width", 200)
# A light grid behind every chart
sns.set_theme(style="whitegrid", context="notebook")

# The seed in use and whether the run timer started
print("seed:", RANDOM_STATE, "| timer started:", RUN_STARTED > 0)
""")

check("""
# Every model below reads this seed
assert RANDOM_STATE == 42, f"expected seed 42, got {RANDOM_STATE}"
# The run timer holds a real clock reading
assert RUN_STARTED > 0, "the run timer did not start"
""")

step("1.5", "Import the project's tested functions",
     "Add `src/` to Python's search path and import the column lists, the parsing, the target, the leakage guard "
     "and the shortfall arithmetic.",
     "why keep logic in `src/` instead of the notebook?",
     "A function in a `.py` file can be checked by `tests/` on small hand-made inputs without the 196 MB "
     "dataset, and the notebook runs that same tested code.")

code("""
# Let Python find the modules in the src folder next to this notebook
sys.path.insert(0, str(Path.cwd() / "src"))

# From src/features.py: the column lists, the borrower parsing, the target and the leakage guard
from features import (
    CATEGORICAL_COLUMNS,     # columns one-hot encoded before modelling
    LEAKY_COLUMNS,           # columns that only exist after a loan is funded
    POSTING_TIME_FEATURES,   # everything known when a loan is posted
    LeakageError,            # the error the leakage guard raises
    add_borrower_features,   # parses borrower_genders into counts
    assert_no_leakage,       # raises if a leaky column reaches the model
    mark_fully_funded,       # the target: funded_amount >= loan_amount
)
# From src/impact.py: the business-impact arithmetic used in Part 3
from impact import funding_shortfall, shortfall_capture

# The four columns the model must never see
print("leaky columns:", LEAKY_COLUMNS)
# How many posting-time inputs there are, and which of them are text categories
print(type(POSTING_TIME_FEATURES).__name__, "of", len(POSTING_TIME_FEATURES), "posting-time features;",
      "categorical:", CATEGORICAL_COLUMNS)
""")

check("""
# The four post-funding columns are on the leaky list
assert sorted(LEAKY_COLUMNS) == ["disbursed_time", "funded_amount", "funded_time", "lender_count"], LEAKY_COLUMNS
# Fourteen columns are known when a loan is posted
assert len(POSTING_TIME_FEATURES) == 14, f"expected 14 posting-time features, got {len(POSTING_TIME_FEATURES)}"
# No posting-time feature is also a leaky column
assert not set(POSTING_TIME_FEATURES) & set(LEAKY_COLUMNS), "a leaky column is listed as a posting-time feature"
""")

step("1.6", "Look at the first rows of the raw loans file",
     "Read only the first five rows with a plain `pd.read_csv`, to see the columns before any cleaning.",
     "what does `nrows` do?",
     "It stops reading after that many rows, the quick way to learn a 196 MB file's columns.")

code("""
# The first five loans, exactly as they are stored
raw_peek = pd.read_csv("data/kiva_loans.csv", nrows=5)

# What the peek is and how big it is
print(type(raw_peek).__name__, raw_peek.shape)
# Show the five raw rows
raw_peek.head()
""")

check("""
# Five rows and the file's twenty columns
assert raw_peek.shape == (5, 20), f"expected 5 rows and 20 columns, got {raw_peek.shape}"
# The four columns the cleaning depends on are present
assert {"loan_amount", "funded_amount", "borrower_genders", "posted_time"} <= set(raw_peek.columns), raw_peek.columns
""")

md("""
Each row is one loan. `loan_amount` is what the borrower asked for and `funded_amount` what lenders put in;
`borrower_genders` lists one entry per borrower; `posted_time` is when the request went up. Four columns are filled in
only because a loan was funded: `funded_time`, `disbursed_time`, `lender_count` and `funded_amount`. Section 1F keeps all
four out of the model.
""")

step("1.7", "Read the poverty file",
     "Read the regional poverty file in full with `pd.read_csv`; it is small.",
     "what is the MPI?",
     "The Multidimensional Poverty Index, published by the Oxford Poverty and Human Development Initiative (OPHI) "
     "with the UNDP, scores a region from 0 to 1 on deprivations in health, education and living standards; "
     "higher means poorer.")

code("""
# One row per region, with its MPI and a map point
raw_mpi = pd.read_csv("data/kiva_mpi_region_locations.csv")
# How many rows actually carry a poverty score
mpi_scored = raw_mpi["MPI"].notna().sum()

# The table's type and size, and the scored rows
print(type(raw_mpi).__name__, raw_mpi.shape, "| rows with an MPI score:", mpi_scored)
# Show the columns the analysis uses
raw_mpi[["LocationName", "country", "region", "MPI", "lat", "lon"]].head()
""")

check("""
# The poverty file's size as downloaded
assert raw_mpi.shape == (2772, 9), f"expected (2772, 9), got {raw_mpi.shape}"
# Only 984 of its rows carry a score
assert mpi_scored == 984, f"expected 984 MPI scores, got {mpi_scored}"
""")

md("""
Only 984 of the 2,772 rows carry an MPI score; the rest are empty rows in the upstream file. `LocationName` names a
province together with its country ("Badakhshan, Afghanistan"), which is the key Kiva's region link uses.
""")

step("1.8", "Read Kiva's region link file",
     "Read `loan_themes_by_region.csv`, where Kiva links the region written on a loan to a poverty-index province "
     "(column `mpi_region`).",
     "why is a link table needed?",
     "A loan's `region` is usually a town (\"Lahore\"), while the poverty index is published by province "
     "(\"Punjab\"), so matching the two names directly fails for most loans.")

code("""
# Kiva's link table, read in full (about 5 MB)
raw_themes = pd.read_csv("data/loan_themes_by_region.csv")
# Rows that name a poverty-index province
themes_linked = raw_themes["mpi_region"].notna().sum()

# The table's type and size, and the rows with a link
print(type(raw_themes).__name__, raw_themes.shape, "| rows with a linked province:", themes_linked)
# Show the three columns the link uses
raw_themes[["country", "region", "mpi_region"]].head()
""")

check("""
# The link file's size as downloaded
assert raw_themes.shape == (15736, 21), f"expected (15736, 21), got {raw_themes.shape}"
# Almost every row names a province
assert themes_linked == 15722, f"expected 15,722 linked rows, got {themes_linked}"
""")

step("1.9", "Set the column types and the two constants",
     "Choose a type for each loan column, and name the censoring window and the time-trend column.",
     "what is a categorical dtype?",
     "A text column with few distinct values (sector, country) stored as `category` keeps each value once and a "
     "small code per row, which saves memory on 671,205 rows.")

code("""
# The type to read each loan column as: repeated text as category, numbers as compact integers and floats
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
# The four timestamp columns, parsed as dates
TIMESTAMPS = ["posted_time", "disbursed_time", "funded_time", "date"]
# A loan's outcome counts as known 60 days after it was posted (Step 1.35 shows why)
CENSOR_DAYS = 60
# The name of the time-trend feature: months since the first loan in the data
TREND_COLUMN = "months_since_start"

# How many columns get a set type, and the two constants
print(type(LOAN_DTYPES).__name__, "of", len(LOAN_DTYPES), "| CENSOR_DAYS:", CENSOR_DAYS, "| TREND_COLUMN:", TREND_COLUMN)
""")

check("""
# Eleven columns get an explicit type
assert len(LOAN_DTYPES) == 11, f"expected 11 typed columns, got {len(LOAN_DTYPES)}"
# The outcome is judged 60 days after posting
assert CENSOR_DAYS == 60, f"expected 60, got {CENSOR_DAYS}"
# Four columns are dates
assert len(TIMESTAMPS) == 4, f"expected 4 timestamp columns, got {len(TIMESTAMPS)}"
""")

step("1.10", "Count borrowers on a small example",
     "Try `add_borrower_features` from `src/features.py` on three made-up loans, so you can see exactly what it "
     "adds before it runs on 671,205.",
     "what is a group loan?",
     "Many Kiva loans go to a group of borrowers who guarantee each other, so `borrower_genders` lists one entry "
     "per borrower (\"female, female, male\") and has to be counted, not read as one label.")

code("""
# Three made-up loans: one borrower, a group of three, and a loan with no genders recorded
example = pd.DataFrame({
    "borrower_genders": ["female", "female, female, male", None],
    "loan_amount": [300.0, 500.0, 1000.0],
    "funded_amount": [300.0, 500.0, 725.0],
})
# Add n_male, n_female, n_borrowers and pct_female
example = add_borrower_features(example)

# The example's type and size after the four new columns
print(type(example).__name__, example.shape)
# Show the three loans with their counts
example
""")

check("""
# One, three and zero borrowers were counted
assert list(example["n_borrowers"]) == [1, 3, 0], f"got {list(example['n_borrowers'])}"
# The group of three is two thirds female
assert abs(example.loc[1, "pct_female"] - 2 / 3) < 1e-9, example.loc[1, "pct_female"]
# A loan with no recorded genders gets NaN, not 0
assert pd.isna(example.loc[2, "pct_female"]), "no recorded genders must give NaN, not 0"
""")

md("""
The group loan counts as two women and one man, a female share of 0.667. The loan with no genders gets
`pct_female = NaN`, not 0: NaN means "we could not tell", while 0 would claim the loan had only male borrowers, and
the model reads this column.
""")

step("1.11", "Mark the target on the example",
     "Apply `mark_fully_funded` to the same three loans.",
     "what is the target?",
     "The column the model learns to predict: here 1 when a loan raised at least what it asked for "
     "(`funded_amount >= loan_amount`) and 0 when it fell short.")

code("""
# 1 if the loan raised at least what it asked for, else 0
example["fully_funded"] = mark_fully_funded(example)

# Show the amounts next to the new target
example[["loan_amount", "funded_amount", "fully_funded"]]
""")

check("""
# The third loan raised $725 of $1,000, so only it is marked 0
assert list(example["fully_funded"]) == [1, 1, 0], f"got {list(example['fully_funded'])}"
""")

md("""
The third loan raised $725 of $1,000, so it is not fully funded. The rule uses `>=` rather than `==` because a few
real loans closed slightly above their target (Step 1.24 counts them), and those are funded.
""")

step("1.12", "Write a function that finds misplaced map points",
     "Write `find_misplaced`, which flags a region whose map point lies far from the rest of its country, and try "
     "it on four made-up points.",
     "what is a robust standard deviation?",
     "The median distance from the median, times 1.4826: on normal data it matches the usual standard deviation, "
     "but a few wild points cannot inflate it.")

code("""
# Flag a region whose point is far from its country's other points
def find_misplaced(mpi):
    # The map point of every region
    coords = mpi[["lat", "lon"]]
    # Each country's median point, repeated on every row of that country
    country_median = coords.groupby(mpi["country"]).transform("median")
    # How far each region sits from its country's median point
    offset = (coords - country_median).abs()
    # The robust standard deviation of that distance within each country
    spread = offset.groupby(mpi["country"]).transform("median") * 1.4826
    # The limit: five robust standard deviations, and never less than 10 degrees
    limit = np.maximum(10, 5 * spread)
    # True where the point is beyond the limit in latitude or in longitude
    return (offset["lat"] > limit["lat"]) | (offset["lon"] > limit["lon"])


# Four made-up regions of one country: three at the same spot and one 50 degrees away
toy_points = pd.DataFrame({"country": ["A", "A", "A", "A"], "lat": [0.0, 0.0, 0.0, 50.0], "lon": [0.0, 0.0, 0.0, 50.0]})
# Which of the four the function flags
toy_flags = find_misplaced(toy_points)
# The flags, one per made-up region
print(type(toy_flags).__name__, list(toy_flags))
""")

check("""
# Only the far point is flagged
assert list(toy_flags) == [False, False, False, True], f"got {list(toy_flags)}"
""")

md("""
The three points at one spot have no spread, so the limit falls back to its 10-degree floor, and the point 50 degrees
away is flagged. The floor stops a country whose regions sit close together from flagging every small offset.
""")

step("1.13", "Find the misplaced points in the poverty file",
     "Keep one row per country and region, then run `find_misplaced` on it.",
     "what does `drop_duplicates(subset=...)` do?",
     "It keeps the first row for each combination of the named columns and drops the repeats, so each region "
     "appears once.")

code("""
# One row per country and region, with its MPI and map point
mpi_regions = raw_mpi[["country", "region", "MPI", "lat", "lon"]].drop_duplicates(subset=["country", "region"])
# True for the regions whose map point is far from the rest of their country
misplaced = find_misplaced(mpi_regions)
# How many points were flagged
n_misplaced = int(misplaced.sum())
# How many regions have a map point at all
n_located = int(mpi_regions["lat"].notna().sum())

# The flag series and the three counts
print(type(misplaced).__name__, f"regions: {len(mpi_regions):,}, with a map point: {n_located}, flagged: {n_misplaced}")
""")

check("""
# One row per region in the poverty file
assert len(mpi_regions) == 1009, f"expected 1,009 regions, got {len(mpi_regions)}"
# Regions with a map point
assert n_located == 892, f"expected 892 located regions, got {n_located}"
# Points flagged as misplaced
assert n_misplaced == 94, f"expected 94 misplaced points, got {n_misplaced}"
""")

md("""
94 of the 892 located regions are flagged; Step 1.48 lists some of them. The wrangle function below blanks these
points on the map but keeps their MPI values, which the analysis uses.
""")

step("1.14", "Write `wrangle_mpi` for the poverty file",
     "Fold the poverty cleaning into one function that returns the exact-join lookup, one cleaned map point per "
     "province, and each province's MPI.",
     "why return three tables from one function?",
     "All three come from the same file and the same cleaning, so the map, the exact join and the link can never "
     "disagree about which points were dropped.")

code("""
# Read and clean the poverty file; return the lookup, the province points and the province scores
def wrangle_mpi(filepath):
    # Read the poverty file
    raw = pd.read_csv(filepath)

    # 1. Exact-join lookup: one row per country and region
    lookup = raw[["country", "region", "LocationName", "MPI", "lat", "lon"]].drop_duplicates(subset=["country", "region"])

    # 2. Blank out the map points that are mis-keyed upstream; keep their MPI values
    lookup.loc[find_misplaced(lookup), ["lat", "lon"]] = np.nan

    # 3. One cleaned point per poverty-index province ("Punjab, Pakistan"), for the province map
    province_coords = (lookup.dropna(subset=["LocationName", "lat"]).drop_duplicates("LocationName")
                       .set_index("LocationName")[["lat", "lon"]])

    # 4. Each province's MPI, for Kiva's region link in wrangle_link
    mpi_by_location = (raw.dropna(subset=["MPI", "LocationName"])
                       .drop_duplicates("LocationName")
                       .set_index("LocationName")["MPI"])

    # The lookup no longer needs the province name
    return lookup.drop(columns="LocationName"), province_coords, mpi_by_location


# The function's inputs, as Python sees them
print(type(wrangle_mpi).__name__, wrangle_mpi.__code__.co_varnames[:wrangle_mpi.__code__.co_argcount])
""")

check("""
# wrangle_mpi takes one input, the file path; Step 1.15 runs it
assert wrangle_mpi.__code__.co_argcount == 1, "wrangle_mpi should take only the file path"
""")

step("1.15", "Run `wrangle_mpi`",
     "Build the three poverty tables and look at each one's size.",
     "what is tuple unpacking?",
     "A function that returns three things returns them as one tuple; `a, b, c = f()` gives each its own name.")

code("""
# The exact-join lookup, the province map points and the province MPI scores
mpi_lookup, province_coords, mpi_by_location = wrangle_mpi("data/kiva_mpi_region_locations.csv")

# The size of each of the three results
print("mpi_lookup", mpi_lookup.shape, "| province_coords", province_coords.shape, "| mpi_by_location", mpi_by_location.shape)
# Show the start of the lookup
mpi_lookup.head()
""")

check("""
# One lookup row per region, with country, region, MPI and map point
assert mpi_lookup.shape == (1009, 5), f"expected (1009, 5), got {mpi_lookup.shape}"
# One cleaned point per province that kept one
assert province_coords.shape == (798, 2), f"expected (798, 2), got {province_coords.shape}"
# One poverty score per scored province
assert len(mpi_by_location) == 984, f"expected 984 province scores, got {len(mpi_by_location)}"
""")

step("1.16", "Write `wrangle_link` for Kiva's region link",
     "Write a function that gives each loan region the poverty-index province Kiva links it to most often, and "
     "that province's MPI.",
     "what is the mode?",
     "The most frequent value in a list; `Series.mode()` lists ties in sorted order, so `.iloc[0]` makes a tie's "
     "choice repeatable.")

code("""
# Read Kiva's link file; return one linked province and its MPI per loan region
def wrangle_link(filepath, mpi_by_location):
    # Read only the three columns the link needs
    themes = pd.read_csv(filepath, usecols=["country", "region", "mpi_region"])
    # Drop rows that name no province
    themes = themes.dropna(subset=["mpi_region"])

    # 1. The most common poverty-index province for each loan region
    rows = []
    for (country, region), group in themes.groupby(["country", "region"]):
        most_common = group["mpi_region"].mode().iloc[0]
        rows.append({"country": country, "region": region, "mpi_region": most_common})
    link = pd.DataFrame(rows)

    # 2. That province's MPI ("Punjab, Pakistan" -> its MPI)
    link["MPI_linked"] = link["mpi_region"].map(mpi_by_location)

    # Keep the four columns the loans table will receive
    return link[["country", "region", "mpi_region", "MPI_linked"]]


# The function's two inputs
print(type(wrangle_link).__name__, wrangle_link.__code__.co_varnames[:wrangle_link.__code__.co_argcount])
""")

check("""
# wrangle_link takes the file path and the province scores; Step 1.17 runs it
assert wrangle_link.__code__.co_argcount == 2, "wrangle_link should take a path and the province scores"
""")

step("1.17", "Run `wrangle_link`",
     "Build the link table and count the loan regions that reach a poverty score through it.",
     "what does `Series.map` do with another Series?",
     "It looks each value up in the other Series' index and returns the match, or NaN when there is none: here a "
     "province name becomes that province's MPI.")

code("""
# One row per loan region: its linked province and that province's MPI
link = wrangle_link("data/loan_themes_by_region.csv", mpi_by_location)
# Loan regions whose linked province has a score
regions_with_linked_mpi = link["MPI_linked"].notna().sum()

# The table's type and size, and the regions with a score
print(type(link).__name__, link.shape, "| loan regions with a linked MPI:", regions_with_linked_mpi)
# Show the start of the link table
link.head()
""")

check("""
# One row per linked loan region
assert link.shape == (9554, 4), f"expected (9554, 4), got {link.shape}"
# Loan regions that reach a poverty score
assert regions_with_linked_mpi == 5671, f"expected 5,671, got {regions_with_linked_mpi}"
# Each loan region appears once, so the join in wrangle cannot duplicate loans
assert not link.duplicated(subset=["country", "region"]).any(), "a loan region appears twice"
""")

step("1.18", "Write `wrangle` for the loans file",
     "Fold everything so far into one function with seven numbered blocks: read, borrowers, target, censoring, "
     "text, poverty and calendar.",
     "what is a join, and why `validate=\"many_to_one\"`?",
     "`merge` adds the lookup's columns to every loan with the same country and region; `validate=\"many_to_one\"` "
     "raises an error if a region appears twice in the lookup, which would otherwise duplicate loans silently.")

code("""
# Read the loans and add every column the analysis needs
def wrangle(filepath, mpi_lookup, link):
    # 1. Read the loans: repeated text as category, the four timestamps as dates
    df = pd.read_csv(filepath, dtype=LOAN_DTYPES, parse_dates=TIMESTAMPS)
    df["funded_amount"] = df["funded_amount"].astype("float32")
    df["loan_amount"] = df["loan_amount"].astype("float32")
    n_loans = len(df)

    # 2. Borrowers: counts of male and female borrowers, and the female share (Step 1.10)
    df = add_borrower_features(df)

    # 3. Target: 1 if funded_amount >= loan_amount (Step 1.11)
    df["fully_funded"] = mark_fully_funded(df)

    # 4. Censoring: days from posting to the end of the data; the outcome is known after CENSOR_DAYS
    df["age_days"] = (df["posted_time"].max() - df["posted_time"]).dt.days
    df["settled"] = df["age_days"] > CENSOR_DAYS

    # 5. Text: empty string for a missing description, and its length
    df["use"] = df["use"].fillna("")
    df["use_len"] = df["use"].str.len().astype("int32")

    # 6. Poverty: the exact-join MPI, then the MPI through Kiva's region link
    df = df.merge(mpi_lookup, on=["country", "region"], how="left", validate="many_to_one")
    df = df.merge(link, on=["country", "region"], how="left", validate="many_to_one")

    # 7. Calendar: month and weekday of posting, and months since the first loan (the time trend)
    df["post_month"] = df["posted_time"].dt.month.astype("int8")
    df["post_dow"] = df["posted_time"].dt.dayofweek.astype("int8")
    df[TREND_COLUMN] = ((df["posted_time"] - df["posted_time"].min()).dt.days / 30.4375).astype("float32")

    # The joins must add columns, never rows
    assert len(df) == n_loans, "a join added rows"
    return df


# The function's three inputs
print(type(wrangle).__name__, wrangle.__code__.co_varnames[:wrangle.__code__.co_argcount])
""")

check("""
# wrangle takes the loans path, the lookup and the link; Step 1.19 runs it
assert wrangle.__code__.co_argcount == 3, "wrangle should take a path, the lookup and the link"
""")

step("1.19", "Build the loan table",
     "Run `wrangle` on the full loans file; this is the slowest cell in Part 1 (about a minute).",
     "what is `memory_usage(deep=True)`?",
     "The memory each column takes, counting the text inside text columns; summed, it says whether the table fits "
     "comfortably in memory.")

code("""
# Read and clean every loan
df = wrangle("data/kiva_loans.csv", mpi_lookup, link)
# The table's memory footprint in megabytes
df_megabytes = df.memory_usage(deep=True).sum() / 1e6

# The loan table's type, size and memory use
print(type(df).__name__, df.shape, f"{df_megabytes:.1f} MB")
# Show the first five loans after cleaning
df.head()
""")

check("""
# Every loan in the file was read, once
assert len(df) == 671_205, f"expected 671,205 loans, got {len(df):,}"
# Twenty raw columns plus sixteen added by wrangle
assert df.shape[1] == 36, f"expected 36 columns, got {df.shape[1]}"
""")

md("""
Each row is one loan. The first twenty columns come straight from the CSV; the rest were added by `wrangle`: the
borrower counts, the target and censoring flags, the description length, the two poverty scores with their map
point, and the calendar features.
""")

step("1.20", "Name the censoring variables",
     "Give the settled flag, the loan age and the last posting date their own names, since many later steps use "
     "them.",
     "what is a snapshot date?",
     "The data was exported on one day; the last posting date in the file marks that day, and every loan's age is "
     "measured back from it.")

code("""
# True when the outcome is known: posted more than CENSOR_DAYS before the snapshot
settled = df["settled"]
# Days between posting and the snapshot
age_days = df["age_days"]
# The last posting date in the data
snapshot_end = df["posted_time"].max()

# The settled flag's type and length, the snapshot date and the settled count
print(type(settled).__name__, settled.shape, "| snapshot ends:", snapshot_end.date(), f"| settled: {settled.sum():,}")
""")

check("""
# The data stops on 26 July 2017
assert str(snapshot_end.date()) == "2017-07-26", f"expected 2017-07-26, got {snapshot_end.date()}"
# Loans whose outcome is known
assert settled.sum() == 645_083, f"expected 645,083 settled loans, got {settled.sum():,}"
""")

# ---------------------------------------------------------------------
# 1B. Explore the loans
# ---------------------------------------------------------------------
md("""
## 1B. Explore the loans

Before asking which loans go unfunded, get to know the loans themselves: what is missing, what looks impossible, how
large the loans are, and where they go.
""")

step("1.21", "Count the descriptions `wrangle` filled in",
     "Count the loans whose description (`use`) `wrangle` replaced with empty text.",
     "why fill missing text with an empty string?",
     "The text tools in section 1D expect a string in every row and fail on NaN; an empty string says \"no "
     "description\" without stopping them.")

code(r"""
# Loans whose description is empty text
empty_descriptions = (df["use"] == "").sum()

# The empty-description count: type and value
print(type(empty_descriptions).__name__, f"descriptions filled with empty text by wrangle: {empty_descriptions:,}")
""")

check(r"""
# 4,232 loans had no description in the file
assert empty_descriptions == 4_232, f"expected 4,232, got {empty_descriptions:,}"
""")

step("1.22", "Find the missing values in the original columns",
     "Count missing values in the twenty columns the CSV itself holds, largest first.",
     "why does missing data matter?",
     "A model cannot use a NaN directly, and a column can be missing for a reason (here, because the loan was "
     "never funded), which is information in itself.")

code(r"""
# The CSV's own columns, taken from the five-row peek in Step 1.6
raw_columns = raw_peek.columns
# Missing values per raw column, largest first
missing_counts = df[raw_columns].isna().sum().sort_values(ascending=False)

# The missing-value counts: type and length
print(type(missing_counts).__name__, missing_counts.shape)
# Show the ten columns with the most gaps
missing_counts.head(10)
""")

check(r"""
# region is blank for 56,800 loans, which limits the poverty join
assert missing_counts["region"] == 56_800, f"expected 56,800 missing regions, got {missing_counts['region']:,}"
# funded_time is blank for loans never fully funded
assert missing_counts["funded_time"] == 48_331, f"expected 48,331, got {missing_counts['funded_time']:,}"
# borrower_genders is blank for 4,221 loans
assert missing_counts["borrower_genders"] == 4_221, f"expected 4,221, got {missing_counts['borrower_genders']:,}"
""")

md("""
`funded_time` is missing for loans that were never fully funded, which is expected: it is part of the outcome.
`region` is missing for 56,800 loans, which limits the poverty join in section 1E. `use` shows no gaps because
`wrangle` replaced the missing descriptions with empty text.
""")

step("1.23", "Look for duplicate loans and impossible amounts",
     "Count repeated loan ids and loans asking for zero dollars or less.",
     "what is a sanity check?",
     "A quick test for values that cannot be true; finding them early stops one bad row from distorting every "
     "later result.")

code(r"""
# Loan ids that appear more than once
duplicate_ids = df["id"].duplicated().sum()
# Loans asking for zero or less
non_positive_amounts = (df["loan_amount"] <= 0).sum()

# The two counts
print("duplicate loan ids:", duplicate_ids, "| loan_amount <= 0:", non_positive_amounts)
""")

check(r"""
# Every loan id is unique
assert duplicate_ids == 0, f"found {duplicate_ids} duplicate ids"
# Every loan asks for a positive amount
assert non_positive_amounts == 0, f"found {non_positive_amounts} non-positive amounts"
""")

step("1.24", "Count the overfunded loans",
     "Count loans that raised more than they asked for.",
     "why does this decide `>=` over `==`?",
     "If some loans close above target, a rule of \"raised exactly the amount\" would call them unfunded, which "
     "they are not.")

code(r"""
# Loans that raised more than they asked for
overfunded = (df["funded_amount"] > df["loan_amount"]).sum()

# The overfunded count: type and value
print(type(overfunded).__name__, "funded_amount > loan_amount:", overfunded)
""")

check(r"""
# Only two loans closed above their target
assert overfunded == 2, f"expected 2 overfunded loans, got {overfunded}"
""")

md("""
Two loans closed slightly above their target. They are funded loans, which is why the target uses `>=` rather than
`==`.
""")

step("1.25", "Summarise the loan amounts",
     "Print the count, mean, spread and percentiles of `loan_amount`.",
     "what does `describe()` show?",
     "The count, mean, standard deviation, minimum, 25th, 50th (median) and 75th percentiles and maximum: a "
     "one-table picture of a numeric column.")

code(r"""
# Summary statistics of the amount each loan asked for
amount_summary = df["loan_amount"].describe()

# The summary's type and length
print(type(amount_summary).__name__, amount_summary.shape)
# Show the eight statistics
amount_summary
""")

check(r"""
# The median loan asks for $500
assert amount_summary["50%"] == 500, f"expected a median of $500, got {amount_summary['50%']}"
# The largest asks for $100,000
assert amount_summary["max"] == 100_000, f"expected a maximum of $100,000, got {amount_summary['max']}"
# The smallest asks for $25
assert amount_summary["min"] == 25, f"expected a minimum of $25, got {amount_summary['min']}"
""")

step("1.26", "Plot the distribution of loan amounts",
     "Draw the raw amounts capped at the 99th percentile, and the amounts on a log scale, side by side.",
     "what is a skewed distribution?",
     "Most values are small and a few are very large, so the histogram has a long right tail and the mean sits "
     "above the median; a log scale pulls the tail in.")

code(r"""
# The 99th percentile: 99% of loans ask for this much or less
amount_cap = df["loan_amount"].quantile(0.99)
# Two empty charts side by side for the amounts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left: amounts, with everything above the cap drawn at the cap
axes[0].hist(df["loan_amount"].clip(upper=amount_cap), bins=50)
# Name the left chart and its axes
axes[0].set(title="Loan amount distribution (99th pct clipped)", xlabel="Loan amount (USD)", ylabel="Number of loans")
# Right: log(1 + amount), which pulls in the long tail
axes[1].hist(np.log1p(df["loan_amount"]), bins=50)
# Name the right chart and its axes
axes[1].set(title="log1p(loan_amount)", xlabel="log(1 + loan amount in USD)", ylabel="Number of loans")
# Stop the two charts' labels from overlapping
plt.tight_layout()
# Draw the histograms
plt.show()

# The cap used on the left chart
print(f"99th percentile of loan_amount: ${amount_cap:,.0f}")
""")

check(r"""
# 99% of loans ask for $5,600 or less
assert amount_cap == 5_600, f"expected a 99th percentile of $5,600, got {amount_cap}"
# The left histogram has its 50 bars
assert len(axes[0].patches) == 50, f"expected 50 bars, got {len(axes[0].patches)}"
""")

md("""
The amounts are right-skewed: the median loan is $500, the mean $842 is pulled up by a long tail, and the largest is
$100,000. On the log scale the shape is close to symmetric.
""")

step("1.27", "Count loans by sector",
     "Count the loans in each of the 15 sectors.",
     "what does `value_counts()` return?",
     "A Series with one row per distinct value and how often it appears, largest first.")

code(r"""
# Loans per sector, largest first
top_sectors = df["sector"].value_counts().head(15)

# The sector counts: type and length
print(type(top_sectors).__name__, top_sectors.shape)
# Show the fifteen sectors
top_sectors
""")

check(r"""
# Agriculture is the largest sector
assert top_sectors.index[0] == "Agriculture", f"expected Agriculture first, got {top_sectors.index[0]}"
# with 180,302 loans
assert top_sectors["Agriculture"] == 180_302, f"expected 180,302, got {top_sectors['Agriculture']:,}"
# and there are 15 sectors in all
assert len(top_sectors) == 15, f"expected 15 sectors, got {len(top_sectors)}"
""")

step("1.28", "Plot loans by sector",
     "Show the sector counts as a horizontal bar chart, largest at the top.",
     "why a horizontal bar chart?",
     "Category names such as \"Personal Use\" read easily down the side, and sorted bars show the ranking at a "
     "glance.")

code(r"""
# An empty chart for the sectors
fig, ax = plt.subplots(figsize=(9, 6))
# One bar per sector; sorting smallest first puts the largest at the top
top_sectors.sort_values().plot(kind="barh", ax=ax)
# Name the sector chart and its axes
ax.set(title="Loan count by sector", xlabel="Number of loans", ylabel="Sector")
# Keep the sector names inside the figure
plt.tight_layout()
# Draw the sector bars
plt.show()
""")

check(r"""
# One bar per sector
assert len(ax.patches) == 15, f"expected 15 bars, got {len(ax.patches)}"
""")

md("""
Agriculture (180,302 loans), Food (136,657) and Retail (124,494) together hold about two thirds of all loans; the
smallest sectors, Entertainment and Wholesale, have fewer than 1,000 each.
""")

step("1.29", "Count loans by country",
     "Count the loans in the 15 largest countries.",
     "why look at concentration?",
     "If a few countries hold most of the loans, any average across countries mostly describes those few.")

code(r"""
# Loans per country, the 15 largest
top_countries = df["country"].value_counts().head(15)

# The country counts: type and length
print(type(top_countries).__name__, top_countries.shape)
# Show the fifteen countries
top_countries
""")

check(r"""
# The Philippines has the most loans
assert top_countries.index[0] == "Philippines", f"expected the Philippines first, got {top_countries.index[0]}"
# 160,441 of them
assert top_countries["Philippines"] == 160_441, f"expected 160,441, got {top_countries['Philippines']:,}"
# Kenya is next with 75,825
assert top_countries["Kenya"] == 75_825, f"expected 75,825, got {top_countries['Kenya']:,}"
""")

step("1.30", "Plot loans by country",
     "Show the country counts as a horizontal bar chart.",
     "why sort before plotting?",
     "pandas draws horizontal bars from the bottom up, so sorting smallest first puts the largest country at the "
     "top.")

code(r"""
# An empty chart for the countries
fig, ax = plt.subplots(figsize=(9, 6))
# One bar per country, the largest at the top
top_countries.sort_values().plot(kind="barh", ax=ax)
# Name the country chart and its axes
ax.set(title="Loan count by country (top 15)", xlabel="Number of loans", ylabel="Country")
# Keep the country names inside the figure
plt.tight_layout()
# Draw the country bars
plt.show()
""")

check(r"""
# One bar per country shown
assert len(ax.patches) == 15, f"expected 15 country bars, got {len(ax.patches)}"
""")

md("""
Lending is concentrated. The Philippines alone has 160,441 loans, about one in four and more than twice Kenya's
75,825; the top five countries hold about half of all loans. Any country-level result is dominated by these few.
""")

step("1.31", "Look at the borrower counts",
     "Summarise the counts `wrangle` parsed from `borrower_genders`, the parsing you tried on three loans in Step "
     "1.10.",
     "what is a median?",
     "The middle value once the values are sorted, so a few very large groups cannot pull it the way they pull the "
     "mean.")

code(r"""
# Loans where no borrower could be parsed (genders missing)
no_parsed_borrower = (df["n_borrowers"] == 0).sum()
# The female share of the middle loan
median_female_share = df["pct_female"].median()

# The two figures
print(f"loans with 0 parsed borrowers: {no_parsed_borrower:,} | median female share per loan: {median_female_share:.2%}")
# Summarise the four borrower columns
df[["n_male", "n_female", "n_borrowers", "pct_female"]].describe()
""")

check(r"""
# pct_female is a share between 0 and 1
assert df["pct_female"].dropna().between(0, 1).all(), "pct_female must lie between 0 and 1"
# It is missing exactly where nobody was parsed
assert (df["pct_female"].isna() == (df["n_borrowers"] == 0)).all(), "NaN must mean no parsed borrower"
# Those are the 4,221 loans with no genders recorded
assert no_parsed_borrower == 4_221, f"expected 4,221, got {no_parsed_borrower:,}"
""")

md("""
The 4,221 loans with no parsed borrower are exactly the ones whose `borrower_genders` is missing (Step 1.22). Most
loans have a single female borrower, so the median female share is 100%.
""")

step("1.32", "Count the repayment schedules",
     "Count how many loans are repaid monthly, irregularly, in one bullet payment, or weekly.",
     "what is a bullet loan?",
     "A loan repaid in one sum at the end of its term rather than in instalments; farmers often repay after the "
     "harvest.")

code(r"""
# Loans per repayment schedule
repayment_counts = df["repayment_interval"].value_counts()

# The schedule counts: type and length
print(type(repayment_counts).__name__, repayment_counts.shape)
# Show the four schedules
repayment_counts
""")

check(r"""
# Monthly is the most common schedule
assert repayment_counts["monthly"] == 342_717, f"expected 342,717, got {repayment_counts['monthly']:,}"
# Weekly is rare
assert repayment_counts["weekly"] == 602, f"expected 602, got {repayment_counts['weekly']}"
# Every loan has exactly one schedule
assert repayment_counts.sum() == len(df), "every loan has one schedule"
""")

step("1.33", "Count the loans posted each month",
     "Group the loans by calendar month of posting.",
     "what does `resample(\"MS\")` do?",
     "On a table indexed by date it groups rows into calendar months (\"MS\" = month start), like a `groupby` on "
     "the month.")

code(r"""
# Loans posted per calendar month
monthly = df.set_index("posted_time").resample("MS").size()

# The result's type and length, and the first and last months
print(type(monthly).__name__, monthly.shape, "| first:", monthly.index[0].date(), monthly.iloc[0],
      "| last:", monthly.index[-1].date(), monthly.iloc[-1])
""")

check(r"""
# The data covers 43 calendar months
assert len(monthly) == 43, f"expected 43 months, got {len(monthly)}"
# January 2014 has 11,735 loans
assert monthly.iloc[0] == 11_735, f"expected 11,735 loans in the first month, got {monthly.iloc[0]:,}"
# The part-month of July 2017 has only 3,706
assert monthly.iloc[-1] == 3_706, f"expected 3,706 loans in the last month, got {monthly.iloc[-1]:,}"
""")

step("1.34", "Plot the schedules and the monthly volume",
     "Draw the two counts side by side.",
     "why plot a time series before modelling?",
     "A trend or a break over time warns that a random train/test split would mix past and future; section 1F "
     "splits by date for that reason.")

code(r"""
# Two empty charts side by side for schedules and months
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Left: one bar per repayment schedule
repayment_counts.plot(kind="bar", ax=axes[0])
# Name the schedule chart and its axes
axes[0].set(title="Repayment interval", xlabel="Repayment interval", ylabel="Number of loans")
# Right: a line of loans posted per month
monthly.plot(ax=axes[1])
# Name the monthly chart and its axes
axes[1].set(title="Loans posted per month", xlabel="Month posted", ylabel="Loans posted per month")
# Keep the two charts from overlapping
plt.tight_layout()
# Draw the bar chart and the line
plt.show()
""")

check(r"""
# One bar per schedule
assert len(axes[0].patches) == 4, f"expected 4 bars, got {len(axes[0].patches)}"
# One point per month on the line
assert len(axes[1].lines[0].get_xdata()) == 43, "expected 43 monthly points"
""")

md("""
Most loans are repaid monthly, then on an irregular schedule; bullet loans are a smaller group and weekly loans are
rare. Volume grew from about 12,000 loans a month in early 2014 to about 18,000 by 2017, with a dip every January. The
sharp drop at the end is not real: the data stops partway through July 2017, so the last month holds only 3,706
loans.
""")

# ---------------------------------------------------------------------
# 1C. Explore the outcome
# ---------------------------------------------------------------------
md("""
## 1C. Explore the outcome: which loans go unfunded?

Both questions are about whether a loan gets funded, so the outcome is examined before any model is built. A model's
later findings should first show up here as plain differences in funding rates; if the model leans on a feature that
shows no difference here, that is worth questioning.
""")

step("1.35", "Measure the censoring",
     "Tabulate the funded rate by how many days before the end of the data each loan was posted.",
     "what is right-censoring?",
     "The data is a snapshot taken on one day, so a loan posted a week earlier may still be raising money: it is "
     "\"not funded yet\", not \"not funded\", and its true outcome is cut off (censored).")

code(r"""
# The band of each loan's age at the snapshot
age_band = pd.cut(age_days, [-1, 7, 14, 21, 30, 45, 60, 90, 10**6],
                  labels=["0-7", "8-14", "15-21", "22-30", "31-45", "46-60", "61-90", ">90"])
# The funded rate and the number of loans in each age band
censoring = df.groupby(age_band, observed=True)["fully_funded"].agg(funded_rate="mean", loans="size")

# The table's type and size, and the snapshot date it is measured from
print(type(censoring).__name__, censoring.shape, f"| snapshot ends {snapshot_end.date()}")
# The funded rate by days between posting and snapshot
print(censoring.to_string(float_format="{:.3f}".format))
""")

check(r"""
# Loans in their first week: 16.4% funded so far
assert abs(censoring.loc["0-7", "funded_rate"] - 0.1636) < 1e-4, censoring.loc["0-7", "funded_rate"]
# Loans older than 90 days: 93.5% funded
assert abs(censoring.loc[">90", "funded_rate"] - 0.9354) < 1e-4, censoring.loc[">90", "funded_rate"]
# Every loan falls in one band
assert censoring["loans"].sum() == len(df), "every loan falls in one band"
""")

md("""
Loans posted in the last three weeks show funded rates between 15% and 35%, against more than 90% for loans older
than 60 days. Counting them as "not funded" would fill the at-risk class with loans that were still raising money, so
from here on only **settled** loans, posted more than 60 days before the end of the data, are used for the outcome.
""")

step("1.36", "Count the settled loans",
     "Count the loans excluded as too recent, and the share of settled loans that went unfunded.",
     "what is a base rate?",
     "The share of loans that have the outcome you are looking for; a model that guesses at random finds unfunded "
     "loans at exactly this rate, so every model is judged against it.")

code(r"""
# Loans too recent to have a known outcome
recent_excluded = (~settled).sum()
# The share of settled loans that were not fully funded
settled_unfunded_share = 1 - df.loc[settled, "fully_funded"].mean()

# The excluded count and its share of all loans
print(f"Excluded {recent_excluded:,} loans ({(~settled).mean():.1%}) posted within {CENSOR_DAYS} days of the snapshot.")
# The settled count and how many of them fell short
print(f"Settled loans: {settled.sum():,}, of which {settled_unfunded_share:.1%} not fully funded")
""")

check(r"""
# 26,122 loans are too recent to judge
assert recent_excluded == 26_122, f"expected 26,122 excluded loans, got {recent_excluded:,}"
# 6.4% of settled loans went unfunded
assert abs(settled_unfunded_share - 0.0637) < 1e-4, f"expected 6.4% unfunded, got {settled_unfunded_share:.4f}"
""")

step("1.37", "Build the outcome table",
     "Keep the settled loans and the columns this section compares, and add a `not_funded` column so the at-risk "
     "outcome reads as 1.",
     "why flip the target?",
     "The question is about loans that go unfunded; with `not_funded = 1` for them, the mean of the column is the "
     "unfunded share.")

code(r"""
# Settled loans, with the columns compared below
outcome = df.loc[settled, ["fully_funded", "term_in_months", "loan_amount", "sector",
                           "repayment_interval", "n_male", "n_female"]].copy()
# 1 for a loan that went unfunded
outcome["not_funded"] = 1 - outcome["fully_funded"]
# The unfunded share across all settled loans: the reference line in later charts
overall_unfunded = outcome["not_funded"].mean()

# The table's type and size, and the overall unfunded share
print(type(outcome).__name__, outcome.shape, f"| not fully funded, all settled loans: {overall_unfunded:.1%}")
# Show the first settled loans
outcome.head()
""")

check(r"""
# The outcome table has one row per settled loan
assert len(outcome) == 645_083, f"expected 645,083 settled loans, got {len(outcome):,}"
# The same base rate as Step 1.36
assert abs(overall_unfunded - 0.06365) < 1e-5, f"expected 0.06365, got {overall_unfunded:.5f}"
""")

step("1.38", "Cut loan term and amount into bands",
     "Cut each of the two columns into six bands of roughly equal size.",
     "what is a quantile band?",
     "`pd.qcut` cuts a column into bands holding roughly equal numbers of loans, so every bar rests on a similar "
     "sample; `duplicates=\"drop\"` merges bands whose edges coincide.")

code(r"""
# Six bands of loan term
outcome["term_band"] = pd.qcut(outcome["term_in_months"], 6, duplicates="drop")
# Six bands of loan amount
outcome["amount_band"] = pd.qcut(outcome["loan_amount"], 6, duplicates="drop")

# How many loans fall in each term band
outcome["term_band"].value_counts(sort=False)
""")

check(r"""
# Six term bands
assert outcome["term_band"].nunique() == 6, "expected six term bands"
# Six amount bands
assert outcome["amount_band"].nunique() == 6, "expected six amount bands"
# Every loan lands in a term band
assert outcome["term_band"].notna().all(), "every loan should fall in a term band"
""")

step("1.39", "Tabulate the unfunded share by band",
     "Average `not_funded` within each band of term and of amount.",
     "what does `.agg(name=\"function\")` do?",
     "It computes several summaries per group and names each result column, here the unfunded share and the "
     "number of loans.")

code(r"""
# Unfunded share and loan count per term band
by_term = outcome.groupby("term_band", observed=True)["not_funded"].agg(not_funded_share="mean", loans="size")
# Unfunded share and loan count per amount band
by_amount = outcome.groupby("amount_band", observed=True)["not_funded"].agg(not_funded_share="mean", loans="size")

# The term table
print(by_term.to_string(float_format="{:.3f}".format))
# The amount table
print(by_amount.to_string(float_format="{:.3f}".format))
""")

check(r"""
# Loans of 8 months or less: 0.3% unfunded
assert abs(by_term["not_funded_share"].iloc[0] - 0.0034) < 1e-4, by_term["not_funded_share"].iloc[0]
# Loans longer than 18 months: 15.6% unfunded
assert abs(by_term["not_funded_share"].iloc[-1] - 0.1558) < 1e-4, by_term["not_funded_share"].iloc[-1]
# Loans above $1,250: 14.6% unfunded
assert abs(by_amount["not_funded_share"].iloc[-1] - 0.1456) < 1e-4, by_amount["not_funded_share"].iloc[-1]
""")

step("1.40", "Plot the unfunded share by term and by amount",
     "Draw both tables as bar charts with the overall share as a dashed line.",
     "why draw a reference line?",
     "A bar means something only next to the average; the dashed line shows which bands are riskier than a "
     "typical loan.")

code(r"""
# Draw one band table as bars, with the all-loan share as a dashed line
def plot_unfunded_share(ax, table, xlabel):
    # One bar per band, as a percentage
    (table["not_funded_share"] * 100).plot(kind="bar", ax=ax, color="#2b6cb0")
    # The share across all settled loans
    ax.axhline(overall_unfunded * 100, color="#a0aec0", ls="--", lw=1, label="All settled loans")
    # Name the horizontal axis after the banded column, and the vertical axis after the share
    ax.set(xlabel=xlabel, ylabel="Not fully funded (%)")
    # Tilt the band labels so they do not collide
    ax.tick_params(axis="x", rotation=30)
    # A legend without a box
    ax.legend(frameon=False)


# Two charts side by side, sharing one vertical scale
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
# Left: the term bands
plot_unfunded_share(axes[0], by_term, "Loan term (months)")
# Right: the amount bands
plot_unfunded_share(axes[1], by_amount, "Loan amount (USD)")
# Title the term chart
axes[0].set_title("Unfunded share by loan term")
# Title the amount chart
axes[1].set_title("Unfunded share by loan amount")
# Keep the tilted labels inside the figure
plt.tight_layout()
# Draw both band charts
plt.show()
""")

check(r"""
# Six bars in each chart
assert len(axes[0].patches) == 6 and len(axes[1].patches) == 6, "expected six bars per chart"
""")

md("""
**What the outcome shows.** The two strongest drivers the model finds later (section 3A) are already visible here.
Loans of 8 months or less almost never go unfunded (0.3%), while loans longer than 18 months go unfunded 15.6% of the
time, about two and a half times the 6.4% average. Amount runs the same way: 0.1% for loans of $225 or less, 14.6%
above $1,250. Each chart looks at one variable at a time, so it cannot say how much of the term effect is really an
amount effect; the model weighs them together, and SHAP in section 3A separates them.
""")

step("1.41", "Label each loan's borrower group",
     "Group borrowers as all female, all male, mixed (group loans with both), or unknown (genders missing).",
     "why build groups from counts?",
     "A loan has one row but may have many borrowers, so four labels give one category per loan that can be "
     "compared like sector or schedule.")

code(r"""
# all female / all male / mixed / unknown, from the parsed gender counts
def borrower_group(frame):
    # Start every loan as "mixed"
    group = pd.Series("mixed", index=frame.index)
    # No parsed borrower: unknown
    group[frame["n_male"] + frame["n_female"] == 0] = "unknown"
    # Only women
    group[(frame["n_male"] == 0) & (frame["n_female"] > 0)] = "all female"
    # Only men
    group[(frame["n_female"] == 0) & (frame["n_male"] > 0)] = "all male"
    return group


# Label every settled loan
outcome["borrower_group"] = borrower_group(outcome)
# Settled loans in each borrower group
group_counts = outcome["borrower_group"].value_counts()
# Show the four group sizes
group_counts
""")

check(r"""
# All-female loans are the large majority
assert group_counts["all female"] == 468_410, f"expected 468,410, got {group_counts['all female']:,}"
# All-male loans
assert group_counts["all male"] == 133_361, f"expected 133,361, got {group_counts['all male']:,}"
# The four groups cover every settled loan
assert group_counts.sum() == settled.sum(), "every settled loan has one group"
""")

step("1.42", "Compare the repayment schedules",
     "Write `unfunded_by`, which tabulates the unfunded share for any column, and apply it to the repayment "
     "schedule.",
     "what is a relative rate?",
     "`vs_overall` divides a group's unfunded share by the overall share, so 2.0 means twice as likely to go "
     "unfunded; groups under 1,000 settled loans are left out as too noisy.")

code(r"""
# The unfunded share, loan count and relative rate for each value of one column
def unfunded_by(col, min_loans=1000):
    # Unfunded share and loan count per value of col
    table = outcome.groupby(col, observed=True)["not_funded"].agg(not_funded_share="mean", loans="size")
    # Drop groups too small to trust
    table = table[table["loans"] >= min_loans]
    # Each group's share relative to all settled loans
    table["vs_overall"] = table["not_funded_share"] / overall_unfunded
    return table.sort_values("not_funded_share", ascending=False)


# The schedule table
by_schedule = unfunded_by("repayment_interval")
# Show the schedule table with three decimals
print(by_schedule.to_string(float_format="{:.3f}".format))
""")

check(r"""
# Bullet loans: 13.7% unfunded
assert abs(by_schedule.loc["bullet", "not_funded_share"] - 0.1366) < 1e-4, by_schedule.loc["bullet"]
# Irregular schedules: 2.2% unfunded
assert abs(by_schedule.loc["irregular", "not_funded_share"] - 0.0215) < 1e-4, by_schedule.loc["irregular"]
""")

step("1.43", "Compare the borrower groups",
     "Apply `unfunded_by` to the borrower groups from Step 1.41.",
     "why compare groups at all?",
     "If one group goes unfunded far more often, a model will learn that, and the funder needs to know whether its "
     "review list ends up favouring or neglecting that group.")

code(r"""
# The borrower-group table
by_group = unfunded_by("borrower_group")
# Show the borrower-group table with three decimals
print(by_group.to_string(float_format="{:.3f}".format))
""")

check(r"""
# All-male loans: 14.5% unfunded
assert abs(by_group.loc["all male", "not_funded_share"] - 0.1450) < 1e-4, by_group.loc["all male"]
# All-female loans: 3.9% unfunded
assert abs(by_group.loc["all female", "not_funded_share"] - 0.0391) < 1e-4, by_group.loc["all female"]
# With no size floor, the groups add back up to every settled loan
assert unfunded_by("borrower_group", min_loans=0)["loans"].sum() == settled.sum(), "groups must cover every loan"
""")

step("1.44", "Compare the sectors",
     "Apply `unfunded_by` to the sectors.",
     "why reuse one function three times?",
     "It guarantees the three tables are built the same way, so they can be compared directly.")

code(r"""
# The sector table
by_sector = unfunded_by("sector")
# Show the sector table with three decimals
print(by_sector.to_string(float_format="{:.3f}".format))
""")

check(r"""
# Housing is the riskiest sector
assert by_sector.index[0] == "Housing", f"expected Housing to be riskiest, got {by_sector.index[0]}"
# at 11.3% unfunded
assert abs(by_sector.loc["Housing", "not_funded_share"] - 0.1128) < 1e-4, by_sector.loc["Housing"]
""")

md("""
**Schedules, borrowers and sectors.** Bullet loans go unfunded 13.7% of the time, more than twice the average, against
2.2% for irregular schedules. Loans to all-male borrowers go unfunded 14.5% of the time, 2.3 times the average,
against 3.9% for all-female borrowers, who hold 468,410 of the 645,083 settled loans. Housing (11.3%) and
Transportation (10.1%) are the riskiest sectors; Arts and Manufacturing almost always fund.

These are raw rates, one variable at a time. All-male loans may also differ in amount, term or country, so the gap
alone does not show that lenders prefer women. The 4,205 loans with no recorded genders go unfunded 15.9% of the
time, so a missing value carries information here; the model can pick it up through `n_borrowers` being 0.
""")

# ---------------------------------------------------------------------
# 1D. Explore the text
# ---------------------------------------------------------------------
md("""
## 1D. Explore the text: what are these loans for?

`use` is a free-text field ("to buy seasonal, fresh fruits to sell"). This section is exploratory only: the
vocabulary fitted here, on every loan, is not used by the model. The model's text features are refitted on the
training period alone in Step 1.63, so it never learns words from the loans it is tested on.
""")

step("1.45", "Find the words that characterise the descriptions",
     "Fit a 30-term vocabulary of single words and two-word phrases, English stop words removed, each in at least "
     "50 descriptions.",
     "what is TF-IDF?",
     "Term frequency times inverse document frequency: a word scores high in a description when it appears there "
     "but is rare across all descriptions, so filler words score low.")

code(r"""
# A 30-term vocabulary over all loans; exploratory only, not a model input
tfidf_eda = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2), min_df=50)
# Learn the vocabulary and score every description: one row per loan, one column per term
eda_matrix = tfidf_eda.fit_transform(df["use"])
# The 30 terms, in alphabetical order
eda_terms = tfidf_eda.get_feature_names_out()

# The matrix's type and size, and the first 15 terms
print(type(eda_matrix).__name__, eda_matrix.shape, "| first 15 terms:", list(eda_terms[:15]))
""")

check(r"""
# One row per loan and one column per term
assert eda_matrix.shape == (671_205, 30), f"expected (671205, 30), got {eda_matrix.shape}"
# "buy" made the vocabulary
assert "buy" in eda_terms, "expected 'buy' in the vocabulary"
""")

step("1.46", "Count the loans that use each term",
     "Count how many descriptions contain each vocabulary term.",
     "what is a sparse matrix?",
     "A matrix that stores only its non-zero cells; most descriptions contain few of the 30 terms, so storing the "
     "zeros would waste memory.")

code(r"""
# Number of loans containing each term: count the non-zero cells in each column
loans_per_term = pd.Series(np.asarray((eda_matrix > 0).sum(axis=0)).ravel(), index=eda_terms, name="loans_using_term")
# The exploratory matrix is no longer needed
del eda_matrix

# The term counts: type and length
print(type(loans_per_term).__name__, loans_per_term.shape)
# Show the ten most used terms
loans_per_term.sort_values(ascending=False).head(10)
""")

check(r"""
# "buy" is in 388,683 descriptions
assert loans_per_term["buy"] == 388_683, f"expected 388,683, got {loans_per_term['buy']:,}"
# "purchase" is in 158,100
assert loans_per_term["purchase"] == 158_100, f"expected 158,100, got {loans_per_term['purchase']:,}"
""")

md("""
"buy" appears in 388,683 descriptions, more than half of all loans, followed by "purchase" and "sell": most loans pay
for stock or inputs that will be resold or used to earn income.
""")

step("1.47", "Find each large sector's own vocabulary",
     "Fit a small eight-term vocabulary within each of the five largest sectors.",
     "why fit per sector?",
     "Across all loans generic words such as \"buy\" dominate; within one sector the words of that line of "
     "business come forward.")

code(r"""
# sector name -> its eight distinctive terms
sector_terms = {}
# Fit one small vocabulary per large sector and print its terms
for sector in df["sector"].value_counts().head(5).index:
    sector_mask = df["sector"] == sector
    sector_tfidf = TfidfVectorizer(max_features=8, stop_words="english", min_df=10)
    sector_tfidf.fit(df.loc[sector_mask, "use"])
    sector_terms[sector] = list(sector_tfidf.get_feature_names_out())
    print(f"{sector}: {sector_terms[sector]}")
""")

check(r"""
# A vocabulary for each of the five sectors
assert len(sector_terms) == 5, f"expected five sectors, got {len(sector_terms)}"
# Agriculture's words include fertilizer
assert "fertilizer" in sector_terms["Agriculture"], sector_terms["Agriculture"]
# Personal Use's words include water
assert "water" in sector_terms["Personal Use"], sector_terms["Personal Use"]
""")

md("""
Each sector's vocabulary is coherent (fertilizer and seeds in Agriculture, canned goods in Retail, water filters and
solar lights in Personal Use), which suggests the text carries information a model can use.
""")

# ---------------------------------------------------------------------
# 1E. Explore poverty and place
# ---------------------------------------------------------------------
md("""
## 1E. Explore poverty and place: the first answer to the poverty question

`wrangle` attached two poverty scores to each loan: `MPI` from an exact match of country and region, and `MPI_linked`
through Kiva's region link. This section checks how many loans each one reaches, then asks of each whether funding
risk falls hardest on the poorest regions.

Some region names in the poverty file are mis-encoded upstream: the file is valid UTF-8, but one placeholder letter
stands in for each accented letter, so it stores "Maranhðo" and "Rondðnia" where the Brazilian states are Maranhão and
Rondônia. The names print as stored rather than being repaired by hand.
""")

step("1.48", "Look at the map points that were dropped",
     "List a few of the regions `find_misplaced` flagged in Step 1.13.",
     "why drop a point but keep its MPI?",
     "The map point only draws the map; the MPI is what the analysis measures, and it is still valid when the point "
     "sits in the wrong continent.")

code(r"""
# A few of the flagged regions with their wrong map points
flagged_examples = mpi_regions.loc[misplaced, ["country", "region", "lat", "lon"]].head(8)

# The examples' type and size
print(type(flagged_examples).__name__, flagged_examples.shape, f"| {n_misplaced} of {n_located} located regions flagged")
# Show the eight examples
print(flagged_examples.to_string(index=False))
""")

check(r"""
# Herat, placed in South-East Asia, is among the flagged regions
assert "Herat" in set(mpi_regions.loc[misplaced, "region"]), "Herat's point should be flagged"
""")

md("""
Herat in Afghanistan placed in South-East Asia, or Oruro in Bolivia placed in the Himalayas, are clearly wrong, so
dropping these points loses nothing real.
""")

step("1.49", "Measure the exact join's coverage",
     "Count the loans the exact country-and-region join matched to a poverty score.",
     "what is coverage?",
     "The share of rows a join actually reaches; a result built on a join with low coverage describes only the "
     "matched rows.")

code(r"""
# Share of loans with an exact-join MPI
coverage = df["MPI"].notna().mean()
# Settled loans with an exact-join MPI
settled_with_mpi = df.loc[settled, "MPI"].notna().sum()

# The coverage, the matched count and the settled matched count
print(f"MPI join coverage: {coverage:.1%} of loans | matched: {df['MPI'].notna().sum():,} / {len(df):,} "
      f"| settled with an exact MPI: {settled_with_mpi:,}")
""")

check(r"""
# 50,955 loans matched exactly
assert df["MPI"].notna().sum() == 50_955, f"expected 50,955 matched loans, got {df['MPI'].notna().sum():,}"
# 48,778 of them are settled
assert settled_with_mpi == 48_778, f"expected 48,778, got {settled_with_mpi:,}"
""")

md("""
Only 7.6% of loans match. The free-text `region` field does not line up with the poverty file's region names well
enough for an exact join, so the results built on it (the first look below and the `MPI` model feature) describe a
small, non-random subset. Step 1.52 widens the poverty score to most loans, and the map and the priority table in
section 3B use that wider score.
""")

step("1.50", "Summarise each region with a known MPI",
     "For regions with an exact-join MPI and at least 20 settled loans, count the loans and total amount, and "
     "compute the share fully funded.",
     "why a minimum of 20 loans?",
     "A funded share from three loans can be 0% or 100% by chance; twenty is a small but sensible floor.")

code(r"""
# One row per region with an exact-join MPI
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
# Keep regions with at least 20 settled loans
region_summary = region_summary[region_summary["n_loans"] >= 20]

# The region table's type and size
print(type(region_summary).__name__, region_summary.shape)
# Show the ten poorest regions
region_summary.sort_values("MPI", ascending=False).head(10)
""")

check(r"""
# 75 regions pass the floor
assert len(region_summary) == 75, f"expected 75 regions, got {len(region_summary)}"
# They hold 48,590 settled loans
assert region_summary["n_loans"].sum() == 48_590, f"expected 48,590 loans, got {region_summary['n_loans'].sum():,}"
""")

md("""
The poorest regions in the matched set include Port Loko in Sierra Leone (MPI 0.522) and Ermera in Timor-Leste
(0.497); most of them are funded more than 90% of the time. Port Loko has no map point because its coordinates were
among those dropped in Step 1.13.
""")

step("1.51", "Measure how poverty and funding move together across regions",
     "Correlate each region's MPI with its share of loans fully funded, in two ways.",
     "what is the difference between Pearson and Spearman correlation?",
     "Pearson measures how close the points lie to a straight line; Spearman does the same on the ranks, so a few "
     "extreme regions cannot pull it around.")

code(r"""
# Straight-line correlation between poverty and funded share
mpi_corr = region_summary["MPI"].corr(region_summary["pct_fully_funded"])
# Rank correlation between the same two columns
mpi_corr_rank = region_summary["MPI"].corr(region_summary["pct_fully_funded"], method="spearman")
# The share of settled loans these regions hold
region_loan_share = region_summary["n_loans"].sum() / settled.sum()

# How much of the data the regions hold
print(f"Regions: {len(region_summary)}, settled loans: {region_summary['n_loans'].sum():,} ({region_loan_share:.1%})")
# The two correlations
print(f"MPI vs share fully funded: Pearson {mpi_corr:.3f}, Spearman {mpi_corr_rank:.3f}")
""")

check(r"""
# Pearson correlation across the 75 regions
assert abs(mpi_corr - 0.303) < 5e-4, f"expected Pearson 0.303, got {mpi_corr:.4f}"
# Spearman correlation across the same regions
assert abs(mpi_corr_rank - 0.375) < 5e-4, f"expected Spearman 0.375, got {mpi_corr_rank:.4f}"
""")

md("""
**Does funding risk fall hardest on the poorest regions?** Not in the data that can answer it so far. Across these 75
regions both correlations are positive: in this matched subset, poorer regions were funded slightly more often, not
less. The subset holds only 7.5% of settled loans, and the correlation is ecological, measured across regions rather
than borrowers (see the Limitations), so it does not show that poorer borrowers are favoured. Because the answer rests
on so few loans, the next steps test it on Kiva's region link.
""")

step("1.52", "Measure the linked score's coverage",
     "Count the loans that have a poverty score through Kiva's link.",
     "why is the link approximate?",
     "Kiva assigned each loan region to the nearest poverty-index province, so a region near a border can carry its "
     "neighbour's score (Battambang, in Cambodia, is linked to Banteay Mean Chey).")

code(r"""
# Share of loans with a poverty score through Kiva's link
linked_coverage = df["MPI_linked"].notna().mean()

# The two coverages side by side
print(f"loans with a poverty score: exact join {coverage:.1%}, through Kiva's link {linked_coverage:.1%}")
""")

check(r"""
# 70.9% of loans reach a score through the link
assert abs(linked_coverage - 0.7093) < 1e-4, f"expected 70.9%, got {linked_coverage:.4f}"
""")

md("""
Coverage rises from about one loan in thirteen to about seven in ten. The model keeps the exact-join score; this wider
link is used to re-test the poverty question, to draw the map, to build the priority table in section 3B, and in Step
2.9 as a challenger input.
""")

step("1.53", "Summarise each province on the linked score",
     "Group settled loans by their linked province, keeping provinces with at least 20 settled loans.",
     "what is a province here?",
     "A poverty-index region such as \"Punjab, Pakistan\": the unit the MPI is published for, usually larger than the "
     "town written on the loan.")

code(r"""
# One row per linked province
province_summary = (
    df[settled].dropna(subset=["MPI_linked"])
    .groupby(["country", "mpi_region"], observed=True)
    .agg(n_loans=("id", "count"), pct_fully_funded=("fully_funded", "mean"), MPI=("MPI_linked", "first"))
    .reset_index()
)
# Keep provinces with at least 20 settled loans
province_summary = province_summary[province_summary["n_loans"] >= 20]

# The province table's type and size
print(type(province_summary).__name__, province_summary.shape)
# Show the first provinces
province_summary.head()
""")

check(r"""
# 240 provinces pass the floor
assert len(province_summary) == 240, f"expected 240 provinces, got {len(province_summary)}"
# in 47 countries
assert province_summary["country"].nunique() == 47, f"expected 47 countries, got {province_summary['country'].nunique()}"
""")

step("1.54", "Re-test the poverty question by province",
     "Correlate each province's MPI with its funded share, as in Step 1.51.",
     "why test the same question twice?",
     "If two ways of measuring poverty give different answers, the first answer was resting on how the data was "
     "matched, not on the platform.")

code(r"""
# Pearson correlation across provinces
linked_corr = province_summary["MPI"].corr(province_summary["pct_fully_funded"])
# Spearman correlation across provinces
linked_corr_rank = province_summary["MPI"].corr(province_summary["pct_fully_funded"], method="spearman")
# The share of settled loans these provinces hold
linked_share = province_summary["n_loans"].sum() / settled.sum()

# How much of the data the provinces hold
print(f"provinces: {len(province_summary)} in {province_summary['country'].nunique()} countries, {linked_share:.1%} of settled loans")
# The two correlations on the wider score
print(f"poverty vs share fully funded: Pearson {linked_corr:.3f}, Spearman {linked_corr_rank:.3f}")
""")

check(r"""
# Pearson correlation close to zero
assert abs(linked_corr - 0.081) < 5e-4, f"expected Pearson 0.081, got {linked_corr:.4f}"
# Spearman correlation close to zero
assert abs(linked_corr_rank - 0.121) < 5e-4, f"expected Spearman 0.121, got {linked_corr_rank:.4f}"
# The provinces hold 70.6% of settled loans
assert abs(linked_share - 0.706) < 5e-4, f"expected 70.6% of settled loans, got {linked_share:.4f}"
""")

md("""
Across 240 provinces in 47 countries, holding 70.6% of settled loans, both correlations are close to zero (0.081 and
0.121): on most of the data, poverty says little about whether a loan gets funded.
""")

step("1.55", "Compare funding across poverty fifths",
     "Split settled loans into five equal groups by their province's poverty score, from least to most poor, and "
     "compare funding rates.",
     "what is a quintile?",
     "One of five groups holding a fifth of the rows each, cut at the 20th, 40th, 60th and 80th percentiles; unlike "
     "a correlation, it shows the shape of a relationship.")

code(r"""
# The poverty fifth of each settled loan's province
poverty_fifth = pd.qcut(df.loc[settled, "MPI_linked"], 5)
# The funded share and loan count in each fifth
funded_by_fifth = df[settled].groupby(poverty_fifth, observed=True)["fully_funded"].agg(
    share_fully_funded="mean", loans="size")
# The five funded shares, least poor first
fifth_shares = list(funded_by_fifth["share_fully_funded"])

# The fifths table's type and size
print(type(funded_by_fifth).__name__, funded_by_fifth.shape)
# Show the five fifths
funded_by_fifth
""")

check(r"""
# The least poor fifth: 92.1% funded
assert abs(fifth_shares[0] - 0.9206) < 1e-4, f"expected 92.1%, got {fifth_shares[0]:.4f}"
# The poorest fifth: 93.0% funded
assert abs(fifth_shares[4] - 0.9300) < 1e-4, f"expected 93.0%, got {fifth_shares[4]:.4f}"
# Each middle fifth is funded more often than both ends
assert min(fifth_shares[1:4]) > max(fifth_shares[0], fifth_shares[4]), "the middle fifths should fund most often"
""")

md("""
**The wider answer to the poverty question.** The shape is not a line: loans from the least poor fifth and the poorest
fifth are both funded less often (about 92% and 93%) than loans from the middle three fifths (about 96% to 97%). So
funding risk does not fall hardest on the poorest regions alone; it is somewhat higher at both ends. This is still
measured by province, not by borrower, so the ecological caveat applies, but it rests on most of the data rather than
a small, skewed slice.
""")

step("1.56", "Attach a map point to each province",
     "Join the cleaned province map points from Step 1.15, and drop provinces without one.",
     "what does `join(..., on=...)` do?",
     "It matches a column of the left table to the index of the right table, here the province name to the index "
     "of `province_coords`.")

code(r"""
# Each province's map point; provinces whose point was dropped in Step 1.13 are left off
map_df = province_summary.join(province_coords, on="mpi_region").dropna(subset=["lat", "lon"])

# The table's type and size, and how many provinces lost their point
print(type(map_df).__name__, map_df.shape, f"| on the map: {len(map_df)} of {len(province_summary)} provinces")
""")

check(r"""
# 230 of the 240 provinces have a reliable point
assert len(map_df) == 230, f"expected 230 provinces on the map, got {len(map_df)}"
""")

step("1.57", "Map the funded share by province",
     "Draw each province sized by its settled loans and coloured by the share fully funded, and save it as a PNG.",
     "why save a static image?",
     "GitHub does not display interactive Plotly charts, so the map is also written as a picture and shown from the "
     "file.")

code(r"""
# One dot per province: size = settled loans, colour = share fully funded
fig = px.scatter_geo(
    map_df,
    lat="lat", lon="lon",
    size="n_loans",
    color="pct_fully_funded",
    hover_name="mpi_region",
    hover_data={"country": True, "MPI": ":.3f", "n_loans": True},
    color_continuous_scale="RdYlGn",
    labels={"pct_fully_funded": "Share fully funded", "n_loans": "Settled loans"},
    title="Share of loans fully funded by province (size = settled loans)",
)
# A world projection, cropped to the latitudes with loans
fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#cbd5e0", lataxis_range=[-40, 55])
# The map's size and margins
fig.update_layout(height=520, width=1000, margin=dict(l=10, r=10, t=60, b=10))

# Save and show a static PNG; without kaleido, fall back to the interactive figure
try:
    fig.write_image("figs/geo_funding_vs_poverty.png", scale=2)
    map_saved = True
    display(Image("figs/geo_funding_vs_poverty.png", width=900))
except Exception as export_error:
    map_saved = False
    print(f"Static export skipped (kaleido not available?): {export_error}")
    fig.show()
""")

check(r"""
# One dot per mapped province
assert len(fig.data[0].lat) == 230, f"expected 230 dots, got {len(fig.data[0].lat)}"
# The PNG the README shows was written by this run
assert map_saved and Path("figs/geo_funding_vs_poverty.png").stat().st_mtime > RUN_STARTED, "the map PNG was not written"
""")

md("""
Most provinces are funded more than 90% of the time. The map is dominated in size by the Philippines, whose large
provinces are funded almost every time, then by Kenya and Pakistan. The provinces funded less often are mostly small
ones, which fits the weak link between poverty and funding found in Step 1.54: there is no broad region of the map
where funding fails.
""")

# ---------------------------------------------------------------------
# 1F. Split
# ---------------------------------------------------------------------
md("""
## 1F. Split: posting-time features and a split by date

The last stage of preparing the data decides what the model may see, and which loans it learns from and is tested on.
""")

step("1.58", "Build the modelling table",
     "Keep settled loans and posting-time columns only, plus the time trend, the target and the description text.",
     "what is data leakage?",
     "Training on information the model would not have when it is used: `funded_time`, `disbursed_time`, "
     "`lender_count` and `funded_amount` exist only because a loan was funded, so a model given them would be "
     "reading the answer.")

code(r"""
# Settled loans; posting-time columns, the time trend, the target and the description
model_df = df.loc[settled, POSTING_TIME_FEATURES + [TREND_COLUMN, "fully_funded", "use"]]
# Drop loans missing an amount or a term (there should be none)
model_df = model_df.dropna(subset=["loan_amount", "term_in_months"])

# The table's type and size, and how many settled loans were dropped
print(type(model_df).__name__, model_df.shape, f"| dropped {settled.sum() - len(model_df):,} settled loans with missing core fields")
# Show the first rows of the modelling table
model_df.head()
""")

check(r"""
# Every settled loan has an amount and a term
assert len(model_df) == 645_083, f"expected 645,083 rows, got {len(model_df):,}"
# 14 posting-time columns, the trend, the target and the text
assert model_df.shape[1] == 17, f"expected 17 columns, got {model_df.shape[1]}"
""")

step("1.59", "Run the leakage guard",
     "Pass the input columns to `assert_no_leakage` from `src/features.py`.",
     "why run a guard instead of just being careful?",
     "A leaked column does not break a model, it improves the score, so nothing downstream would flag it; a guard "
     "that raises an error is a check that cannot be skipped by accident.")

code(r"""
# The input columns: everything except the target and the raw text
input_columns = model_df.drop(columns=["fully_funded", "use"]).columns
# Raises LeakageError naming any post-outcome column
assert_no_leakage(input_columns)

# The columns the guard keeps out, and how many inputs passed
print(f"Excluded as leakage: {LEAKY_COLUMNS} | guard passed on {len(input_columns)} input columns")
""")

check(r"""
# None of the four post-funding columns is an input
assert not set(LEAKY_COLUMNS) & set(input_columns), "a post-funding column reached the inputs"
# Fourteen posting-time inputs plus the trend
assert len(input_columns) == 15, f"expected 15 inputs, got {len(input_columns)}"
""")

step("1.60", "See the guard catch a leaked column",
     "Add `funded_amount` to the inputs on purpose and read the error a real mistake would raise.",
     "what does `try` / `except` do?",
     "It runs the code under `try`, and if that raises the named error, runs the `except` block instead of stopping "
     "the notebook, so the message can be read.")

code(r"""
# The real inputs plus one post-outcome column, added on purpose
leaky_inputs = list(input_columns) + ["funded_amount"]
# Run the guard and keep its message instead of stopping
try:
    assert_no_leakage(leaky_inputs)
    guard_message = "no error raised"
except LeakageError as leak_error:
    guard_message = str(leak_error)

# The error the guard raised
print(guard_message)
""")

check(r"""
# The guard named the leaked column
assert "funded_amount" in guard_message, "the guard should have named funded_amount"
""")

md("""
This is the error a real leak raises: the guard names the exact column, and also catches derived names such as
`funded_amount_bucket` that encoding or feature engineering can create. The fix is to remove the named column from the
inputs, never to loosen the guard. Step 1.66 runs it again after encoding for that reason.
""")

step("1.61", "Split the loans by posting date",
     "Put the earliest 80% of settled loans in the training period, split it again 80/20 into a fit slice and a "
     "validation slice, and keep the latest 20% as the test period.",
     "what is a time-based split?",
     "A random split would let the model learn from loans posted after the ones it is tested on, which a real "
     "platform never can; splitting by date tests it the way it would be used, trained on the past and scored on "
     "loans it has not seen.")

code(r"""
# The posting date of every modelling row
posted = df.loc[model_df.index, "posted_time"]
# The training period holds the earliest 80% of settled loans
TRAIN_SHARE = 0.8
# End of the training period
split_date = posted.quantile(TRAIN_SHARE)
# True for loans in the training period
is_train = posted <= split_date
# End of the fit slice: 80% of the way through the training period
val_cut = posted[is_train].quantile(TRAIN_SHARE)
# True for loans in the fit slice
is_fit = is_train & (posted <= val_cut)
# True for loans in the validation slice
is_val = is_train & (posted > val_cut)

# The three periods and their sizes
print(f"Fit slice:        posted up to {val_cut.date()} ({is_fit.sum():,} loans)")
# The validation slice's end and size
print(f"Validation slice: posted after it, up to {split_date.date()} ({is_val.sum():,} loans)")
# The test period's start and size
print(f"Test period:      posted after {split_date.date()} ({(~is_train).sum():,} loans)")
""")

check(r"""
# Every loan is in exactly one slice
assert (is_fit.astype(int) + is_val.astype(int) + (~is_train).astype(int) == 1).all(), "a loan is in two slices"
# The slices follow each other in time
assert posted[is_fit].max() <= posted[is_val].min() and posted[is_val].max() <= posted[~is_train].min(), "slices overlap"
# The three sizes of the recorded run
assert (is_fit.sum(), is_val.sum(), (~is_train).sum()) == (412_853, 103_213, 129_017), "unexpected slice sizes"
""")

md("""
The fit slice runs to 23 Apr 2016, the validation slice to 27 Oct 2016, and the test period covers the 129,017 loans
posted after that. The test period is touched once, in Step 2.10, after the model has been chosen.
""")

step("1.62", "Write the text-flag helper",
     "Write `tfidf_flags`, which marks which vocabulary terms each loan's description contains, and try it on two "
     "made-up descriptions.",
     "why 0/1 flags instead of TF-IDF weights?",
     "A tree model only needs to know whether a word is present, and small integers keep the matrix compact.")

code(r"""
# One 0/1 column per vocabulary term: does this description contain the term?
def tfidf_flags(vec, texts):
    # True where a description contains a term, as small integers
    flags = (vec.transform(texts) > 0).toarray().astype("int8")
    # Name each column after its term, with spaces replaced
    names = []
    for term in vec.get_feature_names_out():
        names.append("use_tfidf_" + term.replace(" ", "_"))
    return pd.DataFrame(flags, columns=names, index=texts.index)


# Two made-up descriptions
toy_texts = pd.Series(["buy seeds", "buy goats"])
# A tiny vocabulary learned from them
toy_vec = TfidfVectorizer().fit(toy_texts)
# Their flags
toy_text_flags = tfidf_flags(toy_vec, toy_texts)
# Show the flag table
toy_text_flags
""")

check(r"""
# One column per word, named after it
assert list(toy_text_flags.columns) == ["use_tfidf_buy", "use_tfidf_goats", "use_tfidf_seeds"], toy_text_flags.columns
# Both descriptions contain "buy"; only the second contains "goats"
assert toy_text_flags.values.tolist() == [[1, 0, 1], [1, 1, 0]], toy_text_flags.values.tolist()
""")

step("1.63", "Write the function that builds the feature matrix",
     "Write `build_matrix`, which turns `model_df` into numbers, learning the text vocabulary and the fill-in "
     "medians from `fit_rows` only.",
     "what is one-hot encoding?",
     "Models need numbers, so `pd.get_dummies` turns a text column such as `sector` into one 0/1 column per value; "
     "`drop_first=True` leaves one value out, since it is implied when all the others are 0.")

code(r"""
# Text flags, one-hot categories and filled gaps, with everything learned from fit_rows only
def build_matrix(fit_rows):
    # 1. Text: learn a 30-term vocabulary from fit_rows, mark which terms each loan's description contains
    vec = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2), min_df=50)
    vec.fit(model_df.loc[fit_rows, "use"])
    terms = tfidf_flags(vec, model_df["use"])

    # 2. Categories: one 0/1 column per sector, activity, country and repayment interval
    features = model_df.drop(columns=["use", "fully_funded"])
    features = pd.concat([features, terms], axis=1)
    encoded = pd.get_dummies(features, columns=CATEGORICAL_COLUMNS, drop_first=True)

    # 3. Missing values: flag a missing MPI (so "no match" differs from "median poverty"), then fill MPI and
    #    pct_female with fit_rows medians
    encoded["MPI_missing"] = encoded["MPI"].isna().astype("int8")
    medians = {}
    medians["MPI"] = encoded.loc[fit_rows, "MPI"].median()
    medians["pct_female"] = encoded.loc[fit_rows, "pct_female"].median()
    encoded = encoded.fillna(medians)
    return encoded, vec, medians


# The function's one input
print(type(build_matrix).__name__, build_matrix.__code__.co_varnames[:build_matrix.__code__.co_argcount])
""")

check(r"""
# build_matrix takes only the rows to learn from; Steps 1.64 and 1.65 run it
assert build_matrix.__code__.co_argcount == 1, "build_matrix should take only fit_rows"
""")

step("1.64", "Build the matrix for choosing the model",
     "Build the matrix with everything learned from the fit slice only; section 2B uses it to choose between "
     "models.",
     "why two matrices?",
     "Model choice must not see the validation slice's words or medians, but the final model may learn from the "
     "whole training period, so each stage gets a matrix learned from exactly the loans it trains on.")

code(r"""
# Vocabulary and medians learned from the fit slice only
X_sel, _, _ = build_matrix(is_fit)
# The text-flag columns the fit-slice vocabulary produced
sel_text_columns = []
# Collect every column whose name marks it as a text flag
for sel_column in X_sel.columns:
    if sel_column.startswith("use_tfidf_"):
        sel_text_columns.append(sel_column)

# The matrix's type and size, and its text-flag count
print(type(X_sel).__name__, X_sel.shape, "| text-flag columns:", len(sel_text_columns))
""")

check(r"""
# The selection matrix has one row per settled loan
assert len(X_sel) == 645_083, f"expected 645,083 rows, got {len(X_sel):,}"
# Thirty text flags
assert len(sel_text_columns) == 30, f"expected 30 text flags, got {len(sel_text_columns)}"
# MPI and pct_female have no gaps left in the selection matrix
assert X_sel[["MPI", "pct_female"]].notna().all().all(), "MPI and pct_female should have no gaps after the fill"
""")

step("1.65", "Build the matrix for the final model",
     "Build the matrix again with everything learned from the whole training period, and keep the fitted vocabulary "
     "and medians for section 2D.",
     "what is a feature matrix?",
     "The table a model reads: one row per loan and one numeric column per feature.")

code(r"""
# Vocabulary and medians learned from the whole training period
X, tfidf_train, train_medians = build_matrix(is_train)
# Every column name, used by the results file and the leakage guard
ALL_COLUMNS = list(X.columns)

# The matrix's type and size, and the medians used to fill gaps
print(type(X).__name__, X.shape, "| training medians:", train_medians)
""")

check(r"""
# 645,083 loans by 306 columns
assert X.shape == (645_083, 306), f"expected (645083, 306), got {X.shape}"
# The missing-poverty flag is one of them
assert "MPI_missing" in ALL_COLUMNS, "the missing-poverty flag is missing"
""")

step("1.66", "Run the leakage guard again after encoding",
     "Run `assert_no_leakage` on the final column names, since encoding creates new ones.",
     "why check twice?",
     "Each transformation can introduce a new name, such as `funded_amount_bucket`; checking at the last moment "
     "before the model sees the data covers every step before it.")

code(r"""
# Raises LeakageError if any encoded column is, or derives from, a leaky column
assert_no_leakage(ALL_COLUMNS)

# How many encoded columns passed
print(f"Leakage guard passed on all {len(ALL_COLUMNS)} encoded columns.")
""")

check(r"""
# No encoded column starts with a leaky name
for encoded_column in ALL_COLUMNS:
    for leaky_prefix in LEAKY_COLUMNS:
        assert not encoded_column.startswith(leaky_prefix), f"{encoded_column} derives from {leaky_prefix}"
""")

step("1.67", "Choose the columns each model sees",
     "Write `model_columns`, which returns a matrix's columns with or without the time trend.",
     "what is a time-trend feature?",
     "The number of months since the first loan in the data; offering it to only one candidate stops the posting "
     "month from quietly standing in for a platform-wide trend.")

code(r"""
# A matrix's columns, with or without the time trend
def model_columns(frame, with_trend):
    # The two matrices can hold different text terms, so columns are chosen per matrix
    columns = list(frame.columns)
    if not with_trend:
        columns.remove(TREND_COLUMN)
    return columns


# The final matrix's column counts without and with the trend
print(f"X shape: {X.shape} ({len(model_columns(X, False))} base features, plus {TREND_COLUMN})")
""")

check(r"""
# 305 columns without the trend
assert len(model_columns(X, False)) == 305, "expected 305 columns without the trend"
# 306 with it
assert len(model_columns(X, True)) == 306, "expected 306 columns with the trend"
""")

step("1.68", "Set the target and compare the slices",
     "Name the target `y` and print the unfunded share in each slice.",
     "what is class imbalance?",
     "When one outcome is much rarer than the other; here about one loan in twenty goes unfunded, which shapes how "
     "the models in section 2B are trained and judged.")

code(r"""
# The target: 1 = fully funded, 0 = not
y = model_df["fully_funded"]

# The target's type and length
print(type(y).__name__, y.shape)
# The unfunded share in each of the three periods
print(f"Not fully funded, fit: {1 - y[is_fit].mean():.1%} | validation: {1 - y[is_val].mean():.1%} | test: {1 - y[~is_train].mean():.1%}")
""")

check(r"""
# 6.1% unfunded in the fit slice
assert abs((1 - y[is_fit].mean()) - 0.0610) < 1e-4, 1 - y[is_fit].mean()
# 9.6% in the validation slice
assert abs((1 - y[is_val].mean()) - 0.0962) < 1e-4, 1 - y[is_val].mean()
# 4.6% in the test period
assert abs((1 - y[~is_train].mean()) - 0.0462) < 1e-4, 1 - y[~is_train].mean()
""")

md("""
Between about 5% and 10% of loans go unfunded in each slice, so the at-risk class is the minority. The validation slice
(9.6%) and the test period (4.6%) differ: the platform got better at funding loans over time, which matters again in
the calibration steps of section 2C.
""")

# =====================================================================
# PART 2: BUILD MODEL
# =====================================================================
md("""
---
# Part 2: Build Model (which loans are at risk?)

Three stages: a **Baseline** (2A) sets the score any model must beat, **Iterate** (2B) chooses a model on the
validation slice, and **Evaluate** (2C) scores it once on the test period, then checks it over time, by borrower group
and for calibration. Section 2D repeats the cycle for a second question, how long funding takes.

## 2A. Baseline

The question is *which loans are at risk of not getting fully funded*, so what matters is performance on the minority
"not funded" class (label 0).
""")

step("2.1", "Write the headline metric",
     "Write `at_risk_pr_auc`, which scores how well a ranking puts the unfunded loans first, and test it on four "
     "made-up loans whose answer is known.",
     "what are precision, recall and PR-AUC?",
     "Precision is the share of flagged loans that really went unfunded and recall the share of unfunded loans that "
     "were flagged; PR-AUC (average precision) sums up precision at every recall level in one number from 0 to 1, "
     "and a random ranking scores the unfunded share.")

code(r"""
# PR-AUC for the not-funded class: flip the labels and the scores so "not funded" is the class being found
def at_risk_pr_auc(y_true, funded_score):
    return average_precision_score(1 - y_true, 1 - funded_score)


# Four made-up loans: two funded (1) and two not (0)
toy_labels = pd.Series([1, 0, 1, 0])
# Funded scores that put the two unfunded loans lowest: a perfect ranking
toy_scores = np.array([0.9, 0.2, 0.8, 0.1])
# The metric on the perfect ranking
toy_pr_auc = at_risk_pr_auc(toy_labels, toy_scores)

# The result's type and value
print(type(toy_pr_auc).__name__, "PR-AUC of a perfect ranking:", toy_pr_auc)
""")

check(r"""
# A perfect ranking scores exactly 1
assert toy_pr_auc == 1.0, f"expected 1.0, got {toy_pr_auc}"
""")

step("2.2", "See why the metric must be flipped",
     "Score a ranking that knows nothing (the same score for every validation loan) with the unflipped "
     "`average_precision_score`, which looks for label 1, the funded loans.",
     "what is the majority-class trap?",
     "A metric aimed at the common class scores about that class's share even for a useless model, so with 90% of "
     "loans funded it looks excellent whatever the model does.")

code(r"""
# The same score for every validation loan: a ranking with no information
constant_scores = np.full(is_val.sum(), 0.5)
# The unflipped metric, which looks for the funded loans
funded_class_ap_of_nothing = average_precision_score(y[is_val], constant_scores)

# What a useless ranking scores on the funded class
print(f"funded-class PR-AUC of a constant score: {funded_class_ap_of_nothing:.4f}")
""")

check(r"""
# The useless ranking scores the funded share, 0.9038
assert abs(funded_class_ap_of_nothing - y[is_val].mean()) < 1e-9, funded_class_ap_of_nothing
""")

md("""
A ranking that knows nothing scores 0.90 on the funded class, because 90% of validation loans were funded. That is
the trap: `average_precision_score` scores label 1 by default, and here label 1 is "funded". From here on two numbers
are reported:

- **PR-AUC (at-risk, minority class), the headline:** `average_precision_score(1 - y_test, 1 - y_proba)`, which flips
  both the labels and the scores so that "not funded" is the class being found;
- **PR-AUC (funded, majority class):** `average_precision_score(y_test, y_proba)`, shown only for comparison.
""")

step("2.3", "Set the baseline",
     "Score the same constant ranking with the flipped metric on the validation slice.",
     "what is a baseline model?",
     "A deliberately naive model that makes the same prediction for every loan; it sets the score any real model has "
     "to beat to be worth anything.")

code(r"""
# The constant ranking's at-risk PR-AUC on the validation slice
baseline_val = at_risk_pr_auc(y[is_val], constant_scores)

# The baseline, which equals the unfunded share
print(f"Baseline validation PR-AUC (at-risk): {baseline_val:.4f}  (= the share not funded)")
""")

check(r"""
# A constant score scores exactly the unfunded share
assert abs(baseline_val - (1 - y[is_val].mean())) < 1e-9, "the baseline should equal the unfunded share"
# 0.0962 on this validation slice
assert abs(baseline_val - 0.0962) < 1e-4, f"expected 0.0962, got {baseline_val:.4f}"
""")

md("""
The baseline scores 0.0962, exactly the share of validation loans that went unfunded. Every model below has to beat
that number to be worth anything.
""")

# ---------------------------------------------------------------------
# 2B. Iterate
# ---------------------------------------------------------------------
md("""
## 2B. Iterate

Iterating means trying several models and keeping the best, judged on the validation slice only. The test period
stays untouched until section 2C.
""")

step("2.4", "Write the three model builders",
     "Write one small function per model type, each returning a fresh, unfitted model with balanced class weights.",
     "what are these models, and what does `class_weight=\"balanced\"` do?",
     "Logistic regression adds up weighted features, a random forest averages many decision trees, and LightGBM adds "
     "small trees one at a time, each fixing the last one's errors; balanced weights make the unfunded loans count "
     "as much in total as the funded ones, so the outputs rank loans but are not probabilities.")

code(r"""
# Scale every feature, then fit a logistic regression with balanced class weights
def make_logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE))


# 200 trees, each at most 12 levels deep, using every processor core
def make_forest():
    return RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                  n_jobs=-1, random_state=RANDOM_STATE)


# 300 small trees added one at a time, each at most 8 levels deep
def make_lgbm():
    return lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                              class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1)


# The kind of object each builder returns
print(type(make_logistic()).__name__, type(make_forest()).__name__, type(make_lgbm()).__name__)
""")

check(r"""
# The logistic builder returns a two-stage pipeline
assert type(make_logistic()).__name__ == "Pipeline", type(make_logistic()).__name__
# The forest builder returns a random forest
assert isinstance(make_forest(), RandomForestClassifier), type(make_forest())
# The LightGBM builder returns a LightGBM classifier
assert isinstance(make_lgbm(), lgb.LGBMClassifier), type(make_lgbm())
""")

step("2.5", "List the four candidates",
     "Name each candidate with the builder it uses and whether it may see the time trend; the fourth is the same "
     "LightGBM given the trend.",
     "why store builders rather than models?",
     "Calling a builder gives a fresh, unfitted model every time, so no stage can reuse a fit from another.")

code(r"""
# name -> function that builds a fresh, unfitted model
CANDIDATES = {
    "Logistic Regression": make_logistic,
    "Random Forest": make_forest,
    "LightGBM": make_lgbm,
    "LightGBM + time trend": make_lgbm,
}
# name -> whether the model is given the time-trend column
USES_TREND = {
    "Logistic Regression": False,
    "Random Forest": False,
    "LightGBM": False,
    "LightGBM + time trend": True,
}

# The four candidate names
print(type(CANDIDATES).__name__, "of", len(CANDIDATES), list(CANDIDATES))
""")

check(r"""
# Four candidates
assert len(CANDIDATES) == 4, f"expected 4 candidates, got {len(CANDIDATES)}"
# Only the last one sees the trend
assert list(USES_TREND.values()) == [False, False, False, True], "only the last candidate uses the trend"
""")

step("2.6", "Score every candidate on the validation slice",
     "Fit each candidate on the fit slice and score the validation slice; this cell takes several minutes.",
     "what is a validation slice?",
     "Loans the model did not train on, used to compare models; choosing on it keeps the test period unseen, so the "
     "test score stays a fair estimate.")

code(r"""
# name -> validation PR-AUC (at-risk)
validation_pr_auc = {}
# name -> funded scores on the validation slice, kept for the calibration steps only
selection_val_scores = {}
# name -> funded scores on the test period, kept for the calibration steps only
selection_test_scores = {}
# Fit each candidate on the fit slice, score the validation slice and the test period
for name in CANDIDATES:
    cols = model_columns(X_sel, USES_TREND[name])
    model = CANDIDATES[name]()
    model.fit(X_sel.loc[is_fit, cols], y[is_fit])
    val_scores = model.predict_proba(X_sel.loc[is_val, cols])[:, 1]
    test_scores = model.predict_proba(X_sel.loc[~is_train, cols])[:, 1]
    validation_pr_auc[name] = at_risk_pr_auc(y[is_val], val_scores)
    selection_val_scores[name] = val_scores
    selection_test_scores[name] = test_scores
    print(f"{name:22s} validation PR-AUC (at-risk): {validation_pr_auc[name]:.4f}")
# Free the selection matrix and the last fitted model
del X_sel, model
""")

check(r"""
# Every candidate beats the constant-score baseline
assert min(validation_pr_auc.values()) > baseline_val, "a candidate failed to beat the baseline"
# The time-trend LightGBM scores 0.4902
assert abs(validation_pr_auc["LightGBM + time trend"] - 0.4902) < 1e-4, validation_pr_auc["LightGBM + time trend"]
# The logistic regression scores 0.3293
assert abs(validation_pr_auc["Logistic Regression"] - 0.3293) < 1e-4, validation_pr_auc["Logistic Regression"]
""")

step("2.7", "Choose the model",
     "Keep the candidate with the highest validation PR-AUC.",
     "why decide before seeing the test period?",
     "A choice made after looking at the test scores turns the test period into a second validation slice, and the "
     "reported score becomes optimistic.")

code(r"""
# No candidate chosen yet
best_model_name = None
# A starting score every real PR-AUC beats
best_score = -1
# Walk the candidates in order; a later one wins only with a strictly higher score
for name in validation_pr_auc:
    if validation_pr_auc[name] > best_score:
        best_model_name = name
        best_score = validation_pr_auc[name]

# The winner and its validation score
print(f"Chosen on the validation slice: {best_model_name} ({best_score:.4f})")
""")

check(r"""
# LightGBM with the time trend wins
assert best_model_name == "LightGBM + time trend", f"expected LightGBM + time trend, got {best_model_name}"
""")

md("""
All four beat the baseline of 0.0962 by a wide margin. Both LightGBM candidates are clearly ahead of the random forest
and logistic regression, and the time-trend version wins by 0.004, a narrow margin that Step 2.11 returns to.
""")

step("2.8", "Write the challenger: the wider poverty score",
     "Write `with_linked_mpi`, which swaps the linked province score (about 71% of loans) in for the exact-join "
     "score (7.6%).",
     "what is a decision rule fixed in advance?",
     "Deciding, before running a test, what result would make you adopt a change: here, a validation PR-AUC gain of "
     "at least 0.005; without it, any small gain can look like a reason to switch.")

code(r"""
# The chosen model's builder, reused by the challenger and the walk-forward test
make_model = CANDIDATES[best_model_name]
# Whether the chosen model uses the trend
with_trend = USES_TREND[best_model_name]


# Replace the exact-join MPI (and its missing flag) with the linked province score
def with_linked_mpi(frame, fit_rows):
    out = frame.copy()
    linked = df.loc[model_df.index, "MPI_linked"]
    out["MPI_missing"] = linked.isna().astype("int8")
    out["MPI"] = linked.fillna(linked[fit_rows].median())
    return out


# The chosen builder and whether it sees the trend
print(make_model.__name__, "| with trend:", with_trend)
""")

check(r"""
# The challenger refits the chosen LightGBM
assert make_model is make_lgbm and with_trend is True, (make_model, with_trend)
""")

step("2.9", "Run the challenger",
     "Refit the chosen model on the fit slice with the wider score and compare on the validation slice.",
     "what is a challenger model?",
     "A variant tested against the current choice under the same conditions, which replaces it only if it wins by "
     "the margin set in advance.")

code(r"""
# The selection matrix with the linked score in place of the exact one
X_linked = with_linked_mpi(build_matrix(is_fit)[0], is_fit)
# The challenger's columns, trend included
cols = model_columns(X_linked, with_trend)
# A fresh copy of the chosen model
model = make_model()
# Fit it on the fit slice
model.fit(X_linked.loc[is_fit, cols], y[is_fit])
# Its funded scores on the validation slice
linked_val_scores = model.predict_proba(X_linked.loc[is_val, cols])[:, 1]
# Its validation PR-AUC
linked_val_pr_auc = at_risk_pr_auc(y[is_val], linked_val_scores)
# The gain over the exact-join score
linked_gain = linked_val_pr_auc - validation_pr_auc[best_model_name]
# The decision under the rule
adopt_linked = linked_gain >= 0.005
# Free the challenger's matrix and model
del X_linked, model

# Both scores and the change
print(f"validation PR-AUC (at-risk): exact-join MPI {validation_pr_auc[best_model_name]:.4f}, "
      f"linked MPI {linked_val_pr_auc:.4f} (change {linked_gain:+.4f})")
# The decision in words
print("decision:", "adopt the linked score" if adopt_linked else "keep the exact-join score (gain below 0.005)")
""")

check(r"""
# The linked score reaches 0.4880
assert abs(linked_val_pr_auc - 0.4880) < 1e-4, f"expected 0.4880, got {linked_val_pr_auc:.4f}"
# so the rule keeps the exact-join score
assert not adopt_linked, "the linked score should not be adopted"
""")

md("""
The wider poverty score does not help the model: validation PR-AUC moves from 0.4902 to 0.4880, below the 0.005 gain
the rule required, so the model keeps the exact-join score. That fits the answer to the poverty question: across most
of the data, poverty is only weakly related to whether a loan gets funded, and the model's ranking rests mainly on the
loan's own terms (section 3A).
""")

# ---------------------------------------------------------------------
# 2C. Evaluate
# ---------------------------------------------------------------------
md("""
## 2C. Evaluate

One rule governs this section. While iterating you can change and refit the model as often as you like; once you score
it on the test period, the model is fixed. If you went back and changed it after seeing the test score, the test period
would stop being unseen data and its score would overstate how the model does on new loans.
""")

step("2.10", "Refit every candidate on the training period and score the test period once",
     "Refit each candidate on the whole training period and score the test period; the choice is already fixed, so "
     "the other three are for reference only (several minutes).",
     "why refit before testing?",
     "The final model may learn from the whole training period, including the validation slice, so it is refitted "
     "on all of it before scoring the later loans once.")

code(r"""
# Training and test rows of the final matrix
X_train, X_test = X[is_train], X[~is_train]
# Training and test outcomes
y_train, y_test = y[is_train], y[~is_train]
# name -> test PR-AUC for the at-risk class
test_pr_auc = {}
# name -> test PR-AUC for the funded class
test_pr_auc_majority = {}
# Refit each candidate and score the test period; keep the chosen model and its scores
for name in CANDIDATES:
    cols = model_columns(X, USES_TREND[name])
    model = CANDIDATES[name]()
    model.fit(X_train[cols], y_train)
    proba = model.predict_proba(X_test[cols])[:, 1]
    test_pr_auc[name] = at_risk_pr_auc(y_test, proba)
    test_pr_auc_majority[name] = average_precision_score(y_test, proba)
    if name == best_model_name:
        best_model = model
        y_proba = proba
# The last fitted copy is no longer needed
del model

# The chosen model's test scores: type and length
print(type(y_proba).__name__, y_proba.shape, "| chosen:", type(best_model).__name__)
""")

check(r"""
# One score per test loan
assert len(y_proba) == 129_017, f"expected 129,017 test scores, got {len(y_proba):,}"
# The chosen model scores 0.3742 on the test period
assert abs(test_pr_auc["LightGBM + time trend"] - 0.3742) < 1e-4, test_pr_auc["LightGBM + time trend"]
# Plain LightGBM scores 0.3859
assert abs(test_pr_auc["LightGBM"] - 0.3859) < 1e-4, test_pr_auc["LightGBM"]
""")

step("2.11", "Compare the candidates in one table",
     "Put each candidate's validation and test scores side by side.",
     "why show the candidates that were not chosen?",
     "It shows how much the order of models moves between periods, which is part of how far the headline can be "
     "trusted.")

code(r"""
# One row per candidate, sorted by the validation score that chose the model
model_table = pd.DataFrame({
    "validation PR-AUC (at-risk)": validation_pr_auc,
    "test PR-AUC (at-risk)": test_pr_auc,
    "test PR-AUC (funded)": test_pr_auc_majority,
}).sort_values("validation PR-AUC (at-risk)", ascending=False)

# Show the table with four decimals
print(model_table.to_string(float_format="{:.4f}".format))
""")

check(r"""
# The chosen model tops the validation column
assert model_table.index[0] == best_model_name, "the chosen model should top the validation column"
# but plain LightGBM is higher on the test period
assert test_pr_auc["LightGBM"] > test_pr_auc[best_model_name], "the test order should differ from the validation order"
""")

md("""
The time-trend LightGBM was chosen on validation. On the test period the plain LightGBM scores slightly higher, but
switching to it now would be choosing on the test set, which would make the test score optimistic. The chosen model is
kept, and both numbers are reported.
""")

step("2.12", "Score the chosen model against both floors",
     "Print the headline PR-AUC, the funded-class PR-AUC and ROC-AUC, each next to what random ranking would score.",
     "what is ROC-AUC?",
     "The chance that the model ranks a random unfunded loan as riskier than a random funded one: 0.5 is random, 1 "
     "is perfect, and unlike PR-AUC it does not depend on how rare the unfunded loans are.")

code(r"""
# PR-AUC for the funded class, for comparison only
pr_auc_majority = average_precision_score(y_test, y_proba)
# PR-AUC for the at-risk class: the headline
pr_auc_minority = at_risk_pr_auc(y_test, y_proba)
# ROC-AUC of the chosen model
roc_auc = roc_auc_score(y_test, y_proba)

# The chosen model's name
print(f"Chosen model (on validation PR-AUC): {best_model_name}")
# The headline and its floor
print(f"PR-AUC (at-risk, minority class) [headline]: {pr_auc_minority:.4f}  | floor (not-funded share): {1 - y_test.mean():.3f}")
# The funded-class figure and its floor
print(f"PR-AUC (funded, majority class):             {pr_auc_majority:.4f}  | floor (funded share): {y_test.mean():.3f}")
# Print ROC-AUC, which ignores how rare unfunded loans are
print(f"ROC-AUC: {roc_auc:.4f}")
""")

check(r"""
# The headline: 0.3742
assert abs(pr_auc_minority - 0.3742) < 1e-4, f"expected 0.3742, got {pr_auc_minority:.4f}"
# The funded-class figure: 0.9958
assert abs(pr_auc_majority - 0.9958) < 1e-4, f"expected 0.9958, got {pr_auc_majority:.4f}"
# ROC-AUC: 0.9196
assert abs(roc_auc - 0.9196) < 1e-4, f"expected 0.9196, got {roc_auc:.4f}"
""")

md("""
The headline PR-AUC is about eight times its floor of 0.046. The funded-class number looks far better but barely clears
its own floor of 0.954, which is why it is not the headline.
""")

step("2.13", "Measure precision and recall at the default cut-off",
     "Compute the whole precision-recall curve, then read off the point where a loan is flagged when its funded score "
     "is below 0.5.",
     "what is a cut-off (threshold)?",
     "The score at which a loan is flagged; moving it trades catching more unfunded loans (recall) against how many "
     "flagged loans really go unfunded (precision).")

code(r"""
# The at-risk outcome, as in at_risk_pr_auc
at_risk_true = 1 - y_test
# The at-risk score
at_risk_score = 1 - y_proba
# Precision and recall at every possible cut-off
precision, recall, _ = precision_recall_curve(at_risk_true, at_risk_score)
# Loans flagged at the default cut-off of 0.5
flag_default = y_proba < 0.5
# Test loans that really went unfunded
actually_unfunded = y_test == 0
# Flagged loans that really went unfunded
caught = flag_default & actually_unfunded
# Share of unfunded loans that were flagged
default_recall = caught.sum() / actually_unfunded.sum()
# Share of flagged loans that were unfunded
default_precision = caught.sum() / flag_default.sum()

# The curve's length and the default point, recall with its precision
print(f"points on the curve: {len(precision):,} | cut-off 0.5: recall {default_recall:.1%} at precision {default_precision:.1%}")
""")

check(r"""
# Recall at the default cut-off: 0.8600
assert abs(default_recall - 0.8600) < 1e-4, f"expected recall 0.8600, got {default_recall:.4f}"
# Precision at the same cut-off: 0.1941
assert abs(default_precision - 0.1941) < 1e-4, f"expected precision 0.1941, got {default_precision:.4f}"
""")

step("2.14", "Find the best precision at three recall levels",
     "For catching at least 25%, 50% and 75% of unfunded loans, find the best precision on the curve.",
     "why report precision at a fixed recall?",
     "A team decides how many of the unfunded loans it wants to reach; this says how many of the loans it reviews "
     "will really need help at that level.")

code(r"""
# recall level -> the best precision at points that reach at least that recall
precision_at_recall = {}
# For each recall level, keep the best precision among points that reach it
for target_recall in (0.25, 0.5, 0.75):
    reaches_target = recall >= target_recall
    precision_at_recall[target_recall] = float(precision[reaches_target].max())

# The three pairs, recall with its precision
for target_recall in precision_at_recall:
    print(f"catching at least {target_recall:.0%} of unfunded loans: best precision {precision_at_recall[target_recall]:.1%}")
""")

check(r"""
# 42.9% precision at 25% recall
assert abs(precision_at_recall[0.25] - 0.4288) < 1e-4, precision_at_recall[0.25]
# 32.6% at 50%
assert abs(precision_at_recall[0.5] - 0.3255) < 1e-4, precision_at_recall[0.5]
# 24.4% at 75%
assert abs(precision_at_recall[0.75] - 0.2441) < 1e-4, precision_at_recall[0.75]
""")

step("2.15", "Plot the precision-recall curve",
     "Draw the curve, the random-ranking floor and the default cut-off, and save it for the README.",
     "what is the area under this curve?",
     "PR-AUC, the curve summed up as one number; the dashed floor is what a random ranking would score.")

code(r"""
# An empty chart for the curve
fig, ax = plt.subplots(figsize=(7, 5))
# The model's precision-recall curve
ax.plot(recall, precision, color="#2b6cb0", lw=2, label=f"{best_model_name} (PR-AUC {pr_auc_minority:.3f})")
# Random ranking: precision equals the unfunded share at every recall
ax.axhline(1 - y_test.mean(), color="#a0aec0", ls="--", lw=1, label=f"Random ranking ({1 - y_test.mean():.3f})")
# The default cut-off of 0.5 as one orange point
ax.scatter([default_recall], [default_precision], color="#c05621", zorder=3,
           label=f"Cut-off 0.5: recall {default_recall:.0%}, precision {default_precision:.0%}")
# Both axes run from 0 to 1, and each says what it measures
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall: share of unfunded loans flagged",
       ylabel="Precision: share of flagged loans that went unfunded")
# The chart's title, on the left
ax.set_title("Precision-recall curve, at-risk class, test period", loc="left", fontsize=11)
# A boxless legend for the curve, the floor and the cut-off
ax.legend(frameon=False)
# Drop the top and right borders
ax.spines[["top", "right"]].set_visible(False)
# Fit the labels inside the figure
plt.tight_layout()
# Save the curve for the README
plt.savefig("figs/pr_curve.png", dpi=150, bbox_inches="tight")
# Draw the curve
plt.show()
""")

check(r"""
# This run wrote the curve the README shows
assert Path("figs/pr_curve.png").stat().st_mtime > RUN_STARTED, "figs/pr_curve.png was not written by this run"
""")

md("""
The curve shows the trade-off that the single PR-AUC number sums up. A quarter of the unfunded loans can be caught at
42.9% precision, more than nine times the 4.6% base rate; catching half drops precision to 32.6%, and three quarters to
24.4%. The default cut-off of 0.5 sits far to the right: it catches 86% of unfunded loans at a precision of 19%. Which
point to use depends on how many loans the team can act on, and section 3B turns that choice into dollars.
""")

step("2.16", "Print the classification report at the default cut-off",
     "Print precision, recall and F1 for both classes at a cut-off of 0.5.",
     "what is the F1 score?",
     "The harmonic mean of precision and recall, high only when both are high; here it shows how lopsided the "
     "default cut-off is for the unfunded class.")

code(r"""
# 1 = predicted funded, 0 = flagged at risk
predicted_label = (y_proba >= 0.5).astype(int)
# The report as a dictionary, kept for the results file
report = classification_report(y_test, predicted_label, output_dict=True)
# The same report as a printed table
print(classification_report(y_test, predicted_label))
""")

check(r"""
# The report's recall for class 0 agrees with Step 2.13
assert abs(report["0"]["recall"] - default_recall) < 1e-9, "report recall should match Step 2.13"
# and so does its precision
assert abs(report["0"]["precision"] - default_precision) < 1e-9, "report precision should match Step 2.13"
""")

md("""
For class 0 (not funded), the model flags 86% of unfunded test loans at a precision of 0.19: about one flagged loan in
five really went unfunded. A high recall with a low precision fits a review queue: most at-risk loans are caught, at
the cost of reviewing many that would have been funded anyway. Recall alone would hide that cost, so it is always
quoted with its precision.
""")

step("2.17", "Check the score over time with walk-forward validation",
     "Retrain the chosen specification before each of four later windows and score each window (several minutes).",
     "what is walk-forward validation?",
     "Train on everything posted before a cut-off, score the next block of loans, move the cut-off on and repeat; "
     "it shows whether the model keeps working as time moves on, not just in one lucky or unlucky window.")

code(r"""
# Window edges at the 60th, 70th, 80th, 90th and 100th percentiles of posting date
cut_dates = list(posted.quantile([0.6, 0.7, 0.8, 0.9, 1.0]))
# One summary row per window
wf_rows = []
# For each window: learn vocabulary and medians from the earlier loans, fit, score the window
for i in range(len(cut_dates) - 1):
    start = cut_dates[i]
    end = cut_dates[i + 1]
    train_rows = posted <= start
    test_rows = (posted > start) & (posted <= end)
    X_wf, _, _ = build_matrix(train_rows)
    cols = model_columns(X_wf, with_trend)
    model = make_model()
    model.fit(X_wf.loc[train_rows, cols], y[train_rows])
    scores = model.predict_proba(X_wf.loc[test_rows, cols])[:, 1]
    floor = 1 - y[test_rows].mean()
    wf_rows.append({
        "test_window": f"{start.date()} to {end.date()}",
        "train_loans": int(train_rows.sum()),
        "test_loans": int(test_rows.sum()),
        "not_funded_share": floor,
        "pr_auc_at_risk": at_risk_pr_auc(y[test_rows], scores),
    })
# Free the last window's matrix and model
del X_wf, model

# One row per window
walk_forward = pd.DataFrame(wf_rows)
# How many times better than random each window's PR-AUC is
walk_forward["lift_over_random"] = walk_forward["pr_auc_at_risk"] / walk_forward["not_funded_share"]
# Show the four windows
walk_forward.round(3)
""")

check(r"""
# Four windows
assert len(walk_forward) == 4, f"expected 4 windows, got {len(walk_forward)}"
# The weakest window scores 0.394
assert abs(walk_forward["pr_auc_at_risk"].min() - 0.3940) < 1e-4, walk_forward["pr_auc_at_risk"].min()
# The strongest scores 0.503
assert abs(walk_forward["pr_auc_at_risk"].max() - 0.5026) < 1e-4, walk_forward["pr_auc_at_risk"].max()
""")

md("""
Across the four windows the at-risk PR-AUC runs from about 0.39 to 0.50, and in every window it is between about 5 and
14 times what random ranking would score. The raw number moves partly because the unfunded share moves (from about 8%
down to about 3%), which is why the lift over random sits next to it. The headline 0.374 is at the low end: it comes
from one model trained up to October 2016 and scored on the last two windows together, whereas the walk-forward refits
before each window, so the headline is the cautious figure.

A single headline can hide groups where the model does much worse, which matters if the score decides which borrowers
get help. The next steps score the model within each segment of the test period.
""")

step("2.18", "Prepare the test loans for segment scoring",
     "Collect each test loan's segment labels, its outcome and its score in one table.",
     "what is a segment?",
     "A subgroup of loans sharing one value, such as all bullet loans or all loans in Kenya.")

code(r"""
# Segment labels of every test loan
seg = df.loc[X_test.index, ["repayment_interval", "sector", "country", "n_male", "n_female"]].copy()
# The borrower group, as in Step 1.41
seg["borrower_group"] = borrower_group(seg)
# The outcome: 1 = funded
seg["funded"] = y_test.values
# The model's funded score
seg["score"] = y_proba

# The segment table's type and size
print(type(seg).__name__, seg.shape)
# Show the first test loans with their labels
seg.head()
""")

check(r"""
# One row per test loan
assert len(seg) == 129_017, f"expected 129,017 test loans, got {len(seg):,}"
# The four borrower groups appear
assert set(seg["borrower_group"]) == {"all female", "all male", "mixed", "unknown"}, set(seg["borrower_group"])
""")

step("2.19", "Write the segment scorer",
     "Write `segment_table`, which scores the model within each value of a column and skips segments with fewer "
     "than 50 unfunded test loans.",
     "why compare segments by ROC-AUC rather than PR-AUC?",
     "PR-AUC's floor is the segment's unfunded share, so its lift over random can be at most 1 divided by that share; "
     "ROC-AUC does not depend on the share, so segments are ranked by it, with PR-AUC and lift alongside.")

code(r"""
# Score the model within each value of one column; weakest segment first
def segment_table(col, min_unfunded=50):
    rows = []
    for value, group in seg.groupby(col, observed=True):
        n_loans = len(group)
        n_unfunded = int((group["funded"] == 0).sum())
        # Skip segments too small to trust, or with no funded loan to compare against
        if n_unfunded < min_unfunded or n_unfunded == n_loans:
            continue
        unfunded_share = n_unfunded / n_loans
        pr_auc = at_risk_pr_auc(group["funded"], group["score"])
        roc = roc_auc_score(group["funded"], group["score"])
        rows.append({
            col: value,
            "test_loans": n_loans,
            "unfunded_share": unfunded_share,
            "roc_auc": roc,
            "pr_auc_at_risk": pr_auc,
            "lift_over_random": pr_auc / unfunded_share,
            "max_lift": 1 / unfunded_share,
        })
    return pd.DataFrame(rows).sort_values("roc_auc").reset_index(drop=True)


# The scorer tried on the repayment schedules
schedule_segments = segment_table("repayment_interval")
# Show the schedule segments
schedule_segments.round(3)
""")

check(r"""
# Bullet loans are ranked least well of the three schedules
assert schedule_segments.iloc[0]["repayment_interval"] == "bullet", schedule_segments.iloc[0]
# at ROC-AUC 0.870
assert abs(schedule_segments.iloc[0]["roc_auc"] - 0.8699) < 1e-4, schedule_segments.iloc[0]["roc_auc"]
""")

step("2.20", "Score the model within each segment",
     "Apply `segment_table` to borrower group, repayment schedule, sector and country.",
     "what is a fairness gap?",
     "A difference in how well a model works for different groups of people; it does not prove unfair treatment, but "
     "it must be checked before a score decides who receives help.")

code(r"""
# column -> its segment table
segments = {}
# Score the model within each segment of the four columns
for col in ("borrower_group", "repayment_interval", "sector", "country"):
    segments[col] = segment_table(col)

# Print each segment table, weakest segment first
for col in segments:
    print(segments[col].to_string(index=False, float_format="{:.3f}".format))
    print()
""")

check(r"""
# The borrower-group table, looked up by group
by_borrower = segments["borrower_group"].set_index("borrower_group")
# All-male loans: ROC-AUC 0.850
assert abs(by_borrower.loc["all male", "roc_auc"] - 0.8499) < 1e-4, by_borrower.loc["all male", "roc_auc"]
# All-female loans: ROC-AUC 0.930
assert abs(by_borrower.loc["all female", "roc_auc"] - 0.9305) < 1e-4, by_borrower.loc["all female", "roc_auc"]
# Pakistan is the weakest country
assert segments["country"].iloc[0]["country"] == "Pakistan", segments["country"].iloc[0]["country"]
""")

md("""
**Where the model is weaker.** By ROC-AUC, the model ranks worst exactly where loans are most at risk. Within all-male
loans it reaches 0.850, against 0.930 within all-female loans; within bullet loans 0.870, against 0.954 for irregular
schedules; and among sectors, Transportation (0.862) and Agriculture (0.870) are lowest. By country the weakest are
Pakistan (0.725) and Lebanon (0.754): few of their loans go unfunded (3.6% and 2.7%), and the model finds them poorly,
with PR-AUC of 0.084 and 0.075.

The lift column shows why ROC-AUC is the fair comparison. Within all-female loans the lift over random is 12.8 and
within all-male loans only 3.6, which reads as a huge gap; but the largest lift possible is 41.6 for one and 8.3 for
the other, because their unfunded shares differ fivefold.
""")

step("2.21", "See how the review list treats each borrower group",
     "Take the riskiest 10% of test loans, as a review team would, and measure for each borrower group the share of "
     "its unfunded loans the list reaches and the share of its listed loans that went unfunded.",
     "why check the list, not only the score?",
     "The list is what the team acts on; a group whose unfunded loans the list misses is a group the funder's help "
     "misses.")

code(r"""
# Test loans from riskiest to safest, ordered the same way as src/impact.py
review_order = np.argsort(-(1 - y_proba), kind="stable")
# How many loans a 10% list holds
review_size = int(round(0.10 * len(y_proba)))
# True for the loans on the list
on_review_list = np.zeros(len(y_proba), dtype=bool)
# Mark the riskiest tenth
on_review_list[review_order[:review_size]] = True
# The list and the outcome next to each loan's borrower group
review_by_group = pd.DataFrame({"borrower_group": seg["borrower_group"].values, "listed": on_review_list,
                                "unfunded": seg["funded"].values == 0})
# The list's loans that went unfunded
review_by_group["listed_and_unfunded"] = review_by_group["listed"] & review_by_group["unfunded"]
# Per group: test loans, unfunded loans, listed loans and listed unfunded loans
group_review = review_by_group.groupby("borrower_group").agg(
    test_loans=("listed", "size"), unfunded=("unfunded", "sum"), listed=("listed", "sum"),
    listed_and_unfunded=("listed_and_unfunded", "sum"))
# Share of each group's unfunded loans that the list reaches
group_review["recall"] = group_review["listed_and_unfunded"] / group_review["unfunded"]
# Share of each group's listed loans that went unfunded
group_review["precision"] = group_review["listed_and_unfunded"] / group_review["listed"]

# The group review table's type and size
print(type(group_review).__name__, group_review.shape)
# Show the four groups, recall next to precision
group_review.round(3)
""")

check(r"""
# The list holds 12,902 loans in all
assert group_review["listed"].sum() == 12_902, group_review["listed"].sum()
# and reaches 3,708 unfunded loans, the same as src/impact.py in Step 3.11
assert group_review["listed_and_unfunded"].sum() == 3_708, group_review["listed_and_unfunded"].sum()
""")

md("""
This is the fairness check a funder would ask for. The list reaches a different share of each group's unfunded loans,
and is right a different share of the time, because both the unfunded share and the model's ranking quality differ
between groups. Before the list decides who gets help, a team should decide whether those gaps are acceptable, and
could, for example, review each group's riskiest loans separately.
""")

md("""
The scores so far rank loans but are not probabilities, because balanced class weights push them towards 0.5. A funder
budgeting matching money needs probabilities, so the next six steps test whether the scores can be turned into them. A
calibrator has to learn from loans the model did not train on, and the final model in Step 2.10 trained on the
validation slice, so its scores cannot be calibrated there. Instead, the selection-stage version of the chosen model
(fitted on the fit slice in Step 2.6) is used: its validation scores teach the calibrator, which is then applied to the
same model's test-period scores.
""")

step("2.22", "Collect the scores to calibrate",
     "Take the selection-stage model's scores and turn them, and the outcomes, into the chance of NOT being funded.",
     "what is calibration?",
     "A model is calibrated when its scores can be read as probabilities: of the loans it gives a 20% chance of going "
     "unfunded, about 20% do.")

code(r"""
# The selection-stage model's funded scores on the validation slice
sel_val_score = selection_val_scores[best_model_name]
# and on the test period
sel_test_score = selection_test_scores[best_model_name]
# Validation outcomes as 1 = not funded
val_risk_true = 1 - y[is_val].to_numpy()
# Test outcomes as 1 = not funded
test_risk_true = 1 - y_test.to_numpy()
# Validation scores as the chance of not being funded
raw_val_risk = 1 - sel_val_score
# Test scores as the chance of not being funded
raw_test_risk = 1 - sel_test_score

# The two score arrays' type and lengths
print(type(raw_val_risk).__name__, raw_val_risk.shape, raw_test_risk.shape)
""")

check(r"""
# One validation score per validation loan, one test score per test loan
assert len(raw_val_risk) == 103_213 and len(raw_test_risk) == 129_017, (len(raw_val_risk), len(raw_test_risk))
""")

step("2.23", "Fit the calibrator on the validation slice",
     "Learn the mapping from raw score to observed unfunded rate on the validation slice, then apply it to the test "
     "period.",
     "what is isotonic regression?",
     "A calibrator that learns a staircase from score to probability with one rule, that a higher score never gets "
     "a lower probability, so it keeps the model's order of loans.")

code(r"""
# A staircase calibrator that clips scores outside the range it learned
calibrator = IsotonicRegression(out_of_bounds="clip")
# Learn the staircase on the validation slice
calibrator.fit(raw_val_risk, val_risk_true)
# Apply it to the test period
cal_test_risk = calibrator.predict(raw_test_risk)

# The calibrated array's type and length, and its first five values
print(type(cal_test_risk).__name__, cal_test_risk.shape, "| first five:", np.round(cal_test_risk[:5], 4))
""")

check(r"""
# The calibrated values are probabilities
assert ((cal_test_risk >= 0) & (cal_test_risk <= 1)).all(), "calibrated values must lie between 0 and 1"
# and calibration kept the model's order of loans
assert (np.diff(cal_test_risk[np.argsort(raw_test_risk, kind="stable")]) >= 0).all(), "calibration changed the order"
""")

step("2.24", "Compare Brier scores",
     "Score a constant forecast (the validation slice's unfunded share), the raw scores and the calibrated scores on "
     "the test period.",
     "what is the Brier score?",
     "The mean squared gap between the predicted probability and what happened (0 or 1); lower is better, and a "
     "constant forecast sets the floor to beat.")

code(r"""
# The constant forecast: the validation slice's unfunded share
constant_risk = val_risk_true.mean()
# That share for every test loan
constant_forecast = np.full(len(test_risk_true), constant_risk)
# Brier score of the constant forecast
brier_constant = brier_score_loss(test_risk_true, constant_forecast)
# of the raw scores
brier_raw = brier_score_loss(test_risk_true, raw_test_risk)
# and of the calibrated scores
brier_calibrated = brier_score_loss(test_risk_true, cal_test_risk)

# The three Brier scores, lower is better
print(f"Brier score: constant {constant_risk:.3f} forecast {brier_constant:.4f} | raw {brier_raw:.4f} | calibrated {brier_calibrated:.4f}")
""")

check(r"""
# Constant forecast: 0.0465
assert abs(brier_constant - 0.0465) < 1e-4, f"expected 0.0465, got {brier_constant:.4f}"
# Raw scores: 0.0345
assert abs(brier_raw - 0.0345) < 1e-4, f"expected 0.0345, got {brier_raw:.4f}"
# Calibrated: 0.0371, worse than raw
assert abs(brier_calibrated - 0.0371) < 1e-4, f"expected 0.0371, got {brier_calibrated:.4f}"
""")

step("2.25", "Compare the average prediction with what happened",
     "Average each forecast, and check whether calibration cost any ranking quality.",
     "why can calibration lower PR-AUC?",
     "Isotonic regression merges nearby scores into one step, so loans that were ranked apart become tied and a "
     "little of the ranking is lost.")

code(r"""
# Average raw chance of going unfunded
mean_predicted_raw = float(raw_test_risk.mean())
# Average calibrated chance
mean_predicted_calibrated = float(cal_test_risk.mean())
# What actually happened
actual_unfunded_share = float(test_risk_true.mean())
# Ranking quality before calibration
pr_auc_raw = average_precision_score(test_risk_true, raw_test_risk)
# and after
pr_auc_calibrated = average_precision_score(test_risk_true, cal_test_risk)

# The averages against the actual share
print(f"mean predicted chance of going unfunded: raw {mean_predicted_raw:.1%}, calibrated {mean_predicted_calibrated:.1%}; actual {actual_unfunded_share:.1%}")
# The two PR-AUCs
print(f"at-risk PR-AUC: raw {pr_auc_raw:.4f}, calibrated {pr_auc_calibrated:.4f}")
""")

check(r"""
# Calibrated scores average 7.8% against 4.6% actual
assert abs(mean_predicted_calibrated - 0.0780) < 1e-4, f"expected 7.8%, got {mean_predicted_calibrated:.4f}"
# Raw PR-AUC 0.3989
assert abs(pr_auc_raw - 0.3989) < 1e-4, pr_auc_raw
# Calibrated PR-AUC 0.3845
assert abs(pr_auc_calibrated - 0.3845) < 1e-4, pr_auc_calibrated
""")

step("2.26", "Compare with the final model's scale",
     "Compare the final model's average score and its ranking with the selection-stage model's.",
     "what is a rank correlation used for here?",
     "If two models order the loans almost the same way, they make the same review list, even when their scores sit "
     "on different scales.")

code(r"""
# The final model's average chance of going unfunded
mean_predicted_final_model = float((1 - y_proba).mean())
# How similarly the two fits rank the test loans
rank_corr_selection_vs_final = pd.Series(sel_test_score).corr(pd.Series(y_proba), method="spearman")

# The final model's average and the rank correlation
print(f"final model (trained up to {split_date.date()}): mean score {mean_predicted_final_model:.1%}; "
      f"rank correlation with the selection-stage scores {rank_corr_selection_vs_final:.3f}")
""")

check(r"""
# The final model averages 22.4%
assert abs(mean_predicted_final_model - 0.2237) < 1e-4, f"expected 22.4%, got {mean_predicted_final_model:.4f}"
# yet ranks the test loans almost like the selection-stage model
assert abs(rank_corr_selection_vs_final - 0.9588) < 1e-4, f"expected 0.959, got {rank_corr_selection_vs_final:.4f}"
""")

step("2.27", "Draw the calibration curve",
     "Plot the predicted chance against the share that actually went unfunded, in ten equal-size bins, for the raw and "
     "calibrated scores.",
     "how do you read a calibration curve?",
     "Points on the dotted diagonal are perfectly calibrated; points below it mean the model over-predicts.")

code(r"""
# Observed share and mean prediction in ten equal-size bins, raw scores
raw_observed, raw_predicted = calibration_curve(test_risk_true, raw_test_risk, n_bins=10, strategy="quantile")
# The same for the calibrated scores
cal_observed, cal_predicted = calibration_curve(test_risk_true, cal_test_risk, n_bins=10, strategy="quantile")
# An empty chart for the calibration curves
fig, ax = plt.subplots(figsize=(6.5, 5))
# The raw scores in grey
ax.plot(raw_predicted, raw_observed, marker="o", color="#a0aec0", lw=2, label="Raw score")
# The calibrated scores in blue
ax.plot(cal_predicted, cal_observed, marker="o", color="#2b6cb0", lw=2, label="Calibrated on validation")
# The diagonal: perfect calibration
ax.plot([0, 1], [0, 1], color="black", ls=":", lw=1, label="Perfectly calibrated")
# Say what each axis measures
ax.set(xlabel="Predicted chance of going unfunded", ylabel="Share that actually went unfunded")
# The calibration chart's title, on the left
ax.set_title("Calibration on the test period (10 equal-size bins)", loc="left", fontsize=11)
# A boxless legend for the three lines
ax.legend(frameon=False)
# Drop the calibration chart's top and right borders
ax.spines[["top", "right"]].set_visible(False)
# Fit the calibration chart's labels inside the figure
plt.tight_layout()
# Save the calibration chart for the README
plt.savefig("figs/calibration.png", dpi=150, bbox_inches="tight")
# Draw the calibration chart
plt.show()
""")

check(r"""
# Ten bins for each line
assert len(raw_predicted) == 10 and len(cal_predicted) == 10, (len(raw_predicted), len(cal_predicted))
# This run wrote the chart the README shows
assert Path("figs/calibration.png").stat().st_mtime > RUN_STARTED, "figs/calibration.png was not written by this run"
""")

md("""
**What calibration shows here, and why it did not help.** The selection-stage model's raw scores averaged a 5.0%
chance of going unfunded against an actual 4.6%, and already sat close to the diagonal; their Brier score (0.0345)
beats the constant forecast (0.0465). That was luck rather than design. The final model in Step 2.10, the same
specification trained to a later date, averages 22.4%, yet ranks the test loans almost the same way (rank correlation
0.959). With a time trend held at its last level, where the training data ends moves the whole scale of the scores
while barely changing their order.

Calibration learned on the validation slice, where 9.6% of loans went unfunded, then over-predicts a test period where
4.6% did: the calibrated scores average 7.8% and the Brier score worsens to 0.0371. Isotonic calibration also merges
nearby scores into steps, which costs a little ranking (PR-AUC 0.3989 to 0.3845).

The lesson: a calibration holds only while the next period's unfunded rate looks like the period it was learned on,
and here that rate halved. Probabilities used to budget matching funds would need recalibrating on the most recent
settled loans, and checking again as each period settles. The ranking, which the review list in section 3B relies on,
does not depend on this.
""")

# ---------------------------------------------------------------------
# 2D. Days to fund
# ---------------------------------------------------------------------
md("""
## 2D. A second model: how long does it take to get funded?

Among loans that *did* get fully funded, predict the number of days from posting to full funding from the same
posting-time features. This is a regression (a number), not a classification (a yes or no), so it gets its own small
Prepare, Baseline, Iterate and Evaluate cycle. The population is already restricted to funded loans and the target is
a time gap, so the funding outcome itself cannot leak in.
""")

step("2.28", "Compute days to fund for funded loans",
     "Keep settled, funded loans and measure the days from posting to full funding.",
     "what is a regression target?",
     "A number to predict rather than a yes or no; here a duration in days, which can include fractions of a day.")

code(r"""
# Settled loans that were fully funded
funded_only = df[(df["fully_funded"] == 1) & settled].copy()
# Days from posting to full funding
funded_only["days_to_fund"] = (funded_only["funded_time"] - funded_only["posted_time"]).dt.total_seconds() / 86400
# Drop any loan recorded as funded before it was posted
funded_only = funded_only[funded_only["days_to_fund"] >= 0]

# The funded-loan table's type and size
print(type(funded_only).__name__, funded_only.shape)
# Summarise the days to fund
print(funded_only["days_to_fund"].describe())
""")

check(r"""
# 604,019 settled, funded loans
assert len(funded_only) == 604_019, f"expected 604,019 funded loans, got {len(funded_only):,}"
# They took 14.7 days on average
assert abs(funded_only["days_to_fund"].mean() - 14.71) < 0.01, funded_only["days_to_fund"].mean()
""")

md("""
Funded loans took 14.7 days on average, but the median is 9.5: most loans fund within two weeks, and a long tail of
slow ones (the slowest took 420 days) pulls the mean up.
""")

step("2.29", "Encode the regression features",
     "Reuse the vocabulary learned on the training period (Step 1.65), one-hot encode the categories, and add the "
     "missing-poverty flag.",
     "why reuse the classification vocabulary?",
     "It was learned from the training period only, so applying it here adds no information from later loans.")

code(r"""
# Text flags from the training-period vocabulary
reg_terms = tfidf_flags(tfidf_train, funded_only["use"])
# Posting-time features, text flags and the target in one table
reg_df = pd.concat([funded_only[POSTING_TIME_FEATURES], reg_terms, funded_only[["days_to_fund"]]], axis=1)
# One 0/1 column per category value
reg_encoded = pd.get_dummies(reg_df, columns=CATEGORICAL_COLUMNS, drop_first=True)
# Flag a missing MPI before filling it
reg_encoded["MPI_missing"] = reg_encoded["MPI"].isna().astype("int8")
# Take the target out of the feature table
y_reg = reg_encoded.pop("days_to_fund")
# The same date split as Step 1.61
reg_train = funded_only["posted_time"] <= split_date

# The table's type and size, and the training and test row counts
print(type(reg_encoded).__name__, reg_encoded.shape, f"| training rows: {reg_train.sum():,}, test rows: {(~reg_train).sum():,}")
""")

check(r"""
# One row per funded loan
assert len(reg_encoded) == 604_019, f"expected 604,019 rows, got {len(reg_encoded):,}"
# The target is not among the features
assert "days_to_fund" not in reg_encoded.columns, "the target must not be a feature"
""")

step("2.30", "Fill the missing values with training medians",
     "Fill `MPI` and `pct_female` with their medians over the training rows.",
     "why medians from the training rows only?",
     "A median computed over the test rows would let information from later loans into the features, a small but "
     "real leak.")

code(r"""
# The MPI median over the training rows
mpi_median = reg_encoded.loc[reg_train, "MPI"].median()
# The female-share median over the training rows
pct_female_median = reg_encoded.loc[reg_train, "pct_female"].median()
# Fill the two columns that have gaps
X_reg = reg_encoded.fillna({"MPI": mpi_median, "pct_female": pct_female_median})

# The filled table's size and the two medians
print(f"Regression rows: {len(X_reg):,} | MPI median: {mpi_median} | pct_female median: {pct_female_median}")
""")

check(r"""
# MPI and pct_female have no gaps left in the regression table
assert X_reg[["MPI", "pct_female"]].notna().all().all(), "MPI and pct_female should have no gaps"
""")

step("2.31", "Set the baseline for days to fund",
     "Give every test loan the median days-to-fund of the training period, and measure the error.",
     "what are MAE and R-squared?",
     "The mean absolute error is the average number of days a prediction is off by; R-squared is the share of the "
     "variation the model explains, 0 meaning no better than the average and 1 perfect.")

code(r"""
# Training and test rows of the regression table
Xr_train, Xr_test = X_reg[reg_train], X_reg[~reg_train]
# Training and test targets
yr_train, yr_test = y_reg[reg_train], y_reg[~reg_train]
# The baseline guess: the training median
baseline_days = float(yr_train.median())
# Its error on the test rows
baseline_mae = mean_absolute_error(yr_test, np.full(len(yr_test), baseline_days))

# The test size and the baseline's error
print(f"Test loans: {len(yr_test):,} | MAE, naive baseline (train median {baseline_days:.1f} days): {baseline_mae:.2f} days")
""")

check(r"""
# 123,059 funded test loans
assert len(yr_test) == 123_059, f"expected 123,059 test loans, got {len(yr_test):,}"
# The median guess is off by 8.08 days on average
assert abs(baseline_mae - 8.08) < 0.005, f"expected 8.08 days, got {baseline_mae:.3f}"
""")

md("""
If we always predicted 10.2 days, the training median, our predictions would be off by 8.08 days on average. A model has
to beat that to be useful.
""")

step("2.32", "Fit a random forest for days to fund",
     "Fit a random forest on the training rows and compare it with the baseline on the test rows.",
     "why a random forest here?",
     "It handles hundreds of 0/1 columns and curved effects without scaling, a solid first model for a table like "
     "this.")

code(r"""
# 150 trees, each at most 10 levels deep
reg_model = RandomForestRegressor(n_estimators=150, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE)
# Fit it on the training rows
reg_model.fit(Xr_train, yr_train)
# Predicted days for each test loan
yr_pred = reg_model.predict(Xr_test)
# Average error on the test rows
reg_mae = mean_absolute_error(yr_test, yr_pred)
# Share of the variation explained
reg_r2 = r2_score(yr_test, yr_pred)

# The forest's error against the baseline's, and its R-squared
print(f"MAE: random forest {reg_mae:.2f} days, naive baseline {baseline_mae:.2f} days | R-squared {reg_r2:.3f}")
""")

check(r"""
# The forest beats the median guess
assert reg_mae < baseline_mae, "the forest should beat the median guess"
# by about a day: 7.04
assert abs(reg_mae - 7.04) < 0.005, f"expected 7.04 days, got {reg_mae:.3f}"
# R-squared 0.248
assert abs(reg_r2 - 0.248) < 5e-4, f"expected R-squared 0.248, got {reg_r2:.4f}"
""")

md("""
The forest is about a day closer than the baseline on average, and explains about a quarter of the variation
(R-squared): most of what decides how fast a loan funds is not in the information available when it is posted.
""")

# =====================================================================
# PART 3: COMMUNICATE RESULTS
# =====================================================================
md("""
---
# Part 3: Communicate Results

This part speaks to a manager rather than a data scientist: what drives the risk, where the funder could act and what
it is worth in dollars, and the results file that backs every number.

## 3A. What drives the risk?
""")

step("3.1", "Draw a sample of test loans to explain",
     "Take 1,000 test loans at random, with the chosen model's columns.",
     "why a sample?",
     "SHAP values take a while to compute for every loan; 1,000 loans give a stable ranking of features in a fraction "
     "of the time.")

code(r"""
# The 306 columns the chosen model was fitted on
best_cols = model_columns(X, USES_TREND[best_model_name])
# 1,000 test loans, drawn with the fixed seed
sample = X_test[best_cols].sample(n=1000, random_state=RANDOM_STATE)

# The sample's type and size
print(type(sample).__name__, sample.shape)
""")

check(r"""
# 1,000 loans by the chosen model's 306 columns
assert sample.shape == (1000, 306), f"expected (1000, 306), got {sample.shape}"
""")

step("3.2", "Compute SHAP values",
     "Split each sampled loan's score into a contribution from every feature.",
     "what is SHAP?",
     "For one loan, SHAP splits the model's score into a push from each feature against the average loan, in "
     "log-odds (the model's internal scale): positive pushes towards funded, negative towards not funded.")

code(r"""
# SHAP's tree explainer only applies to a tree model such as LightGBM
assert "LightGBM" in best_model_name
# One log-odds contribution per sampled loan and feature
shap_values = shap.TreeExplainer(best_model).shap_values(sample)

# The SHAP table's type and size
print(type(shap_values).__name__, shap_values.shape)
""")

check(r"""
# One value per loan and feature
assert shap_values.shape == (1000, 306), f"expected (1000, 306), got {shap_values.shape}"
""")

step("3.3", "Rank the features by their average impact",
     "Average the size of each feature's SHAP values over the 1,000 loans.",
     "why the absolute value?",
     "A feature can push some loans up and others down, and averaging signed values would let those cancel; the "
     "average size measures how much the feature moves scores at all.")

code(r"""
# Mean |SHAP| per feature, largest first
mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=best_cols).sort_values(ascending=False)

# The ranking's heading
print(f"Top 12 features by mean |SHAP| ({best_model_name}, 1,000 test loans, log-odds of being funded):")
# The twelve largest
print(mean_abs_shap.head(12).to_string(float_format="{:.4f}".format))
""")

check(r"""
# Loan term and loan amount lead
assert list(mean_abs_shap.index[:2]) == ["term_in_months", "loan_amount"], list(mean_abs_shap.index[:2])
# Loan term moves the score by 1.5522 on average
assert abs(mean_abs_shap["term_in_months"] - 1.5522) < 1e-4, mean_abs_shap["term_in_months"]
""")

md("""
Loan term (1.55) and loan amount (1.34) move the score about twice as much as the time trend (0.67), and far more than
anything else: posting month, the female share, then individual sectors and countries.
""")

step("3.4", "Chart the top drivers",
     "Draw the ranking as a horizontal bar chart and save it for the README.",
     "what does a ranking not show?",
     "How much a feature matters, not in which direction; the next two steps show the direction.")

code(r"""
# Horizontal bars of mean |SHAP| for the top 12 features
shap.summary_plot(shap_values, sample, plot_type="bar", max_display=12, show=False)
# The bar chart's title
plt.title(f"Top features by mean |SHAP| ({best_model_name})")
# What the bar length measures
plt.xlabel("Mean |SHAP value| (impact on log-odds of being funded)")
# Fit the feature names inside the figure
plt.tight_layout()
# Save the bar chart for the README
plt.savefig("figs/shap_summary.png", dpi=150, bbox_inches="tight")
# Draw the bar chart
plt.show()
""")

check(r"""
# This run wrote the driver chart the README shows
assert Path("figs/shap_summary.png").stat().st_mtime > RUN_STARTED, "figs/shap_summary.png was not written by this run"
""")

step("3.5", "Show the direction of each driver",
     "Draw a beeswarm plot, where each dot is one test loan placed by how far a feature moved its score and coloured "
     "by the feature's value.",
     "how do you read a beeswarm?",
     "If a feature's red (high-value) dots sit on the left, high values of that feature push loans towards not "
     "funded.")

code(r"""
# One row of dots per feature, coloured by the feature's value
shap.summary_plot(shap_values, sample.astype("float32"), max_display=12, show=False)
# The beeswarm's title
plt.title("How each feature moves the score (red = high value)")
# What the dots' position measures
plt.xlabel("SHAP value (impact on log-odds of being funded)")
# Fit the beeswarm's labels inside the figure
plt.tight_layout()
# Save the beeswarm
plt.savefig("figs/shap_beeswarm.png", dpi=150, bbox_inches="tight")
# Draw the beeswarm
plt.show()
""")

check(r"""
# This run wrote the beeswarm
assert Path("figs/shap_beeswarm.png").stat().st_mtime > RUN_STARTED, "figs/shap_beeswarm.png was not written by this run"
""")

step("3.6", "Measure the direction of term and amount in bands",
     "Average the SHAP value of loan term and of loan amount within five bands of each.",
     "why band the values?",
     "Five averages say in numbers what the beeswarm shows in colour, and can be checked against the raw rates in "
     "Step 1.39.")

code(r"""
# feature -> mean SHAP value per band
shap_direction = {}
# For term and amount: band the sampled loans, average each band's SHAP value, print it
for col in ("term_in_months", "loan_amount"):
    band = pd.qcut(sample[col], 5, duplicates="drop")
    col_shap = pd.Series(shap_values[:, best_cols.index(col)], index=sample.index)
    shap_direction[col] = col_shap.groupby(band, observed=True).mean()
    print(f"Mean SHAP value of {col} by band (negative = pushes towards not funded):")
    print(shap_direction[col].to_string(float_format="{:+.3f}".format))
    print()
""")

check(r"""
# The shortest terms push towards funded by +2.230
assert abs(shap_direction["term_in_months"].iloc[0] - 2.230) < 5e-4, shap_direction["term_in_months"].iloc[0]
# The largest amounts push away by -1.723
assert abs(shap_direction["loan_amount"].iloc[-1] - (-1.723)) < 5e-4, shap_direction["loan_amount"].iloc[-1]
""")

md("""
**Direction.** Short loans push strongly towards funded: terms of 8 months or less add +2.23 to the log-odds on
average, while every band above 11 months pulls the score down by 1.3 to 1.5. Small loans do the same: $225 or less
adds +2.39, while loans above $975 subtract 1.72. Both agree with the raw rates in Step 1.39. In the beeswarm, a higher
female share and the Education, Arts and Cambodia columns push towards funded, while Retail, Kenya and Food push the
other way. The time trend pushes every test loan towards not funded: test loans lie past the end of the training
period, and a tree model holds the trend at the last level it saw rather than extending it.

**The model in words, for a manager:** it mostly asks two questions of a new loan, how long is it and how large is it.
Long, large loans are the ones most likely to fall short, and both are terms a field partner can adjust before
posting.
""")

# ---------------------------------------------------------------------
# 3B. Where could the funder act?
# ---------------------------------------------------------------------
md("""
## 3B. Where could the funder act?

**Combining poverty and risk.** Average the model's risk score by province over the test period and set it next to each
province's poverty score from Kiva's region link, which covers most loans. The scores are not probabilities (section
2C), so they are not multiplied by MPI; instead each province is ranked twice, on poverty and on risk, and the
**relative priority index** is the product of the two percentile ranks.
""")

step("3.7", "Average the risk score by province",
     "For provinces with a linked poverty score and at least 30 test loans, average the test-period scores and keep "
     "the share of loans that actually went unfunded.",
     "why at least 30 loans?",
     "A province average over a handful of loans moves a lot with each loan; 30 keeps the ranking from being driven "
     "by a few cases.")

code(r"""
# Each test loan's country, linked province and poverty score
test_region_info = df.loc[X_test.index, ["country", "mpi_region", "MPI_linked"]].copy()
# Its funded score: higher = more likely to be funded, a ranking score
test_region_info["model_score"] = y_proba
# Its outcome: 1 = went unfunded
test_region_info["not_funded"] = 1 - y_test.values
# One row per province
priority = (
    test_region_info.dropna(subset=["MPI_linked"])
    .groupby(["country", "mpi_region"], observed=True)
    .agg(
        n_test_loans=("MPI_linked", "count"),
        MPI=("MPI_linked", "first"),
        mean_model_score=("model_score", "mean"),
        actual_not_funded_share=("not_funded", "mean"),
    )
    .reset_index()
)
# Keep provinces with at least 30 test loans
priority = priority[priority["n_test_loans"] >= 30].copy()

# The table's type and size, and the countries it spans
print(type(priority).__name__, priority.shape, f"| {len(priority)} provinces in {priority['country'].nunique()} countries")
""")

check(r"""
# 154 provinces pass the floor
assert len(priority) == 154, f"expected 154 provinces, got {len(priority)}"
# in 42 countries
assert priority["country"].nunique() == 42, f"expected 42 countries, got {priority['country'].nunique()}"
""")

step("3.8", "Rank the provinces and compute the priority index",
     "Rank provinces on poverty and on risk, multiply the two ranks, and check the model's province ranking against "
     "the share that actually went unfunded.",
     "what is a percentile rank?",
     "A value's position in a list as a share, 1.0 the highest; the product of two ranks is close to 1 only for a "
     "province that is top on both, and its scale means nothing else.")

code(r"""
# 1 = poorest province in the table
priority["poverty_rank"] = priority["MPI"].rank(pct=True)
# 1 = province least likely to be funded
priority["risk_rank"] = priority["mean_model_score"].rank(pct=True, ascending=False)
# Top on both ranks -> close to 1
priority["priority_index"] = priority["poverty_rank"] * priority["risk_rank"]
# Highest priority first; ties broken by poverty
priority = priority.sort_values(["priority_index", "MPI"], ascending=False)
# Does the model's province ranking match what happened?
risk_agreement = (1 - priority["mean_model_score"]).corr(priority["actual_not_funded_share"], method="spearman")

# The agreement between predicted and actual risk
print(f"Rank correlation, predicted risk vs actual unfunded share across provinces: {risk_agreement:.3f}")
# Show the fifteen highest-priority provinces
priority.head(15)
""")

check(r"""
# Moyamba, Sierra Leone tops the list
assert priority.iloc[0]["mpi_region"] == "Moyamba, Sierra Leone", priority.iloc[0]["mpi_region"]
# Predicted and actual province risk agree at 0.674
assert abs(risk_agreement - 0.674) < 5e-4, f"expected 0.674, got {risk_agreement:.4f}"
""")

md("""
**The province priority list.** 154 provinces in 42 countries have at least 30 test loans and a linked poverty score.
Across them, the model's province ranking agrees well with what happened: the rank correlation between predicted risk
and the share of loans that actually went unfunded is 0.674. Moyamba (Sierra Leone) and Mopti (Mali) top the list: both
are among the poorest provinces (MPI 0.569 and 0.594), and about one loan in five actually went unfunded there (21.4%
and 22.6%). Two Timor-Leste provinces, Bobonaro and Ermera, come next on poverty and predicted risk, yet none of their
53 and 47 test loans went unfunded. A province's average score is a reason to look, not a verdict, which is why the
list works as a review queue rather than an allocation rule.

**Business impact.** If the team reviewed only the loans the model ranks riskiest in the test period, what share of
the money those loans fell short would the reviews reach? `src/impact.py` does the arithmetic and `tests/` covers it.
""")

step("3.9", "Measure the funding shortfall of the test period",
     "Compute each test loan's shortfall with `funding_shortfall` from `src/impact.py`, and the total.",
     "what is a shortfall?",
     "The part of a loan's requested amount that lenders never funded, `loan_amount - funded_amount` and never below "
     "zero: money the borrower did not receive through Kiva, not a loss to the platform.")

code(r"""
# The amounts asked for and raised, for each test loan
test_loans = df.loc[X_test.index, ["loan_amount", "funded_amount"]]
# Each loan's shortfall, never below 0
shortfall = funding_shortfall(test_loans["loan_amount"], test_loans["funded_amount"])
# The test period's total shortfall
total_shortfall = shortfall.sum()

# The array's type and length, and the total
print(type(shortfall).__name__, shortfall.shape, f"| {len(test_loans):,} test loans, total funding shortfall ${total_shortfall:,.0f}")
""")

check(r"""
# Test loans fell $5,797,550 short
assert total_shortfall == 5_797_550, f"expected $5,797,550, got ${total_shortfall:,.0f}"
# No loan has a negative shortfall
assert (shortfall >= 0).all(), "a shortfall is never negative"
""")

step("3.10", "Measure what the flagged loans hold",
     "Sum the shortfall of the loans flagged at the 0.5 cut-off (the same cut-off as Step 2.13).",
     "what share of a shortfall does a list hold?",
     "The share of all unfunded dollars that sits in the loans you chose to review.")

code(r"""
# Loans flagged at risk at the default cut-off
flagged = y_proba < 0.5
# Their shortfall
flagged_cover = shortfall[flagged].sum()

# The flagged count and share, and the shortfall they hold
print(f"flagged at the default cut-off: {flagged.sum():,} loans ({flagged.mean():.1%}), "
      f"holding ${flagged_cover:,.0f} ({flagged_cover / total_shortfall:.1%}) of the shortfall")
""")

check(r"""
# 26,388 loans are flagged
assert flagged.sum() == 26_388, f"expected 26,388 flagged loans, got {flagged.sum():,}"
# holding $5,405,175 of the shortfall
assert flagged_cover == 5_405_175, f"expected $5,405,175, got ${flagged_cover:,.0f}"
""")

step("3.11", "Measure the shortfall reached by reviewing the riskiest loans first",
     "Use `shortfall_capture` from `src/impact.py` for the riskiest 5%, 10%, 20% and 30% of test loans, and work out "
     "each list's precision next to its recall.",
     "what is a capture table?",
     "For each review budget, the share of the problem the reviewed loans hold; a random 10% of loans holds about "
     "10%, so the gap is what the ranking adds.")

code(r"""
# Higher = more likely to go unfunded
risk = 1 - y_proba
# One row per review share
capture = shortfall_capture(risk, test_loans["loan_amount"], test_loans["funded_amount"])
# Each list's precision: its unfunded loans over the loans it holds (kept apart from capture, which is saved as is)
capture_precision = capture["unfunded_loans_covered"] / capture["loans_reviewed"]

# The capture table's type and size
print(type(capture).__name__, capture.shape)
# The capture table with the list precision beside it
print(capture.assign(list_precision=capture_precision).to_string(index=False, formatters={
    "review_share": "{:.0%}".format, "shortfall_covered_usd": "${:,.0f}".format,
    "shortfall_covered_pct": "{:.1%}".format, "unfunded_loans_covered_pct": "{:.1%}".format,
    "list_precision": "{:.1%}".format}))
""")

check(r"""
# The row for the riskiest 10%
top_tenth = capture[capture["review_share"] == 0.1].iloc[0]
# It holds 12,902 loans
assert top_tenth["loans_reviewed"] == 12_902, f"expected 12,902 loans, got {top_tenth['loans_reviewed']:,}"
# reaching $4,402,525, 75.9% of the shortfall
assert top_tenth["shortfall_covered_usd"] == 4_402_525, top_tenth["shortfall_covered_usd"]
# The 10% list reaches 62.2% of unfunded loans at 28.7% precision
assert abs(top_tenth["unfunded_loans_covered_pct"] - 0.6225) < 1e-4 and abs(capture_precision.iloc[1] - 0.2874) < 1e-4, capture_precision
""")

step("3.12", "Plot the shortfall reached against the share reviewed",
     "Draw the capture curve for every review share from 1% to 100%, next to what a random review would reach, and "
     "save it for the README.",
     "what does the diagonal mean?",
     "A random review reaches the shortfall in proportion to the loans reviewed, so the gap between curve and "
     "diagonal is what the ranking adds.")

code(r"""
# Review shares from 0% to 100% in steps of 1%
shares = np.linspace(0, 1, 101)
# The capture table at every share above 0
curve = shortfall_capture(risk, test_loans["loan_amount"], test_loans["funded_amount"], review_shares=shares[1:])
# Horizontal positions, starting at the origin
x_values = [0] + list(curve["review_share"] * 100)
# Vertical positions, starting at the origin
y_values = [0] + list(curve["shortfall_covered_pct"] * 100)
# An empty chart for the capture curve
fig, ax = plt.subplots(figsize=(8, 4.5))
# The model's capture curve
ax.plot(x_values, y_values, color="#2b6cb0", lw=2, label=f"{best_model_name} ranking")
# The random-review diagonal
ax.plot([0, 100], [0, 100], color="#a0aec0", ls="--", lw=1, label="Random review")
# Label the four review shares from Step 3.11
for _, capture_row in capture.iterrows():
    ax.annotate(f"{capture_row['shortfall_covered_pct']:.0%}",
                (capture_row["review_share"] * 100, capture_row["shortfall_covered_pct"] * 100),
                textcoords="offset points", xytext=(6, -12), fontsize=9)
# Say what each axis of the capture chart measures
ax.set(xlabel="Share of test-period loans reviewed, riskiest first (%)", ylabel="Share of funding shortfall covered (%)")
# The capture chart's title, with the total
ax.set_title(f"Funding shortfall covered by reviewing the riskiest loans (total ${total_shortfall / 1e6:.1f}M)",
             loc="left", fontsize=11)
# The capture chart's legend, bottom right
ax.legend(frameon=False, loc="lower right")
# Drop the capture chart's top and right borders
ax.spines[["top", "right"]].set_visible(False)
# Fit the capture chart inside the figure
plt.tight_layout()
# Save the capture chart for the README
plt.savefig("figs/shortfall_capture.png", dpi=150, bbox_inches="tight")
# Draw the capture chart
plt.show()
""")

check(r"""
# Reviewing every loan reaches the whole shortfall
assert abs(curve["shortfall_covered_pct"].iloc[-1] - 1.0) < 1e-9, "reviewing every loan must cover 100%"
# This run wrote the chart the README opens with
assert Path("figs/shortfall_capture.png").stat().st_mtime > RUN_STARTED, "figs/shortfall_capture.png was not written"
""")

md("""
Reviewing the riskiest 10% of loans reaches about three quarters of the shortfall, where a random 10% would reach about
10%, because the model concentrates the unfunded dollars at the top of its ranking. That list holds 62% of the loans
that went unfunded, at a precision of 28.7%: a little over one listed loan in four needed the help.

**What a funder or platform should do with this.** Use the ranking as a daily review queue, working down from the
riskiest loans as far as staff time allows; talk to field partners about term and size, the two drivers they control;
recalibrate on recent loans before turning scores into a budget; and before the list decides who receives help, check
the borrower-group gaps in Steps 2.20 and 2.21.
""")

# ---------------------------------------------------------------------
# 3C. Results file
# ---------------------------------------------------------------------
md("""
## 3C. Results file

Every headline number above is written to `outputs/results.json`, with the git commit of the code, the library versions
and the run time, so the README can be checked against it mechanically (`tests/test_readme_numbers.py`).
""")

step("3.13", "Record where the numbers came from",
     "Record the run date, run time, peak memory, git commit and library versions in a `provenance` block.",
     "what is provenance?",
     "A record of exactly what produced a result (which code, which libraries, when), so a reader can rerun the same "
     "version; `uncommitted_code_changes` is false only when no code file differs from that commit.")

code(r"""
# json: writes the results file
import json
# platform: the operating system and Python version
import platform
# subprocess: asks git for the commit
import subprocess
# datetime and timezone: the run date in UTC
from datetime import datetime, timezone
# version: reads an installed package's version
from importlib.metadata import PackageNotFoundError, version

# The commit the code is at, and whether any code file has changed since (outputs and the notebook excluded)
try:
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    changed_files = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)*.ipynb", ":(exclude)figs", ":(exclude)outputs"],
        capture_output=True, text=True, check=True).stdout.strip()
    uncommitted_code_changes = bool(changed_files)
except (subprocess.CalledProcessError, FileNotFoundError):
    git_commit = None
    uncommitted_code_changes = None

# Peak memory of this run: psutil on Windows, the resource module (kilobytes on Linux) elsewhere
try:
    if platform.system() == "Windows":
        import psutil
        peak_bytes = psutil.Process().memory_info().peak_wset
    else:
        import resource
        peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    peak_memory_gb = round(peak_bytes / 1e9, 1)
except Exception:
    peak_memory_gb = None

# package name -> installed version
package_versions = {}
# Look up each library the notebook uses
for package in ["numpy", "pandas", "scikit-learn", "lightgbm", "shap", "matplotlib", "seaborn", "plotly", "kaleido"]:
    try:
        package_versions[package] = version(package)
    except PackageNotFoundError:
        package_versions[package] = None

# The results file starts empty
results = {}
# Its first block: the provenance of this run
results["provenance"] = {
    "run_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "runtime_minutes": round((time.time() - RUN_STARTED) / 60),
    "peak_memory_gb": peak_memory_gb,
    "git_commit": git_commit,
    "uncommitted_code_changes": uncommitted_code_changes,
    "python": platform.python_version(),
    "packages": package_versions,
}
# Show the provenance block
print(json.dumps(results["provenance"], indent=2))
""")

check(r"""
# The run names a commit
assert results["provenance"]["git_commit"], "git did not report a commit"
# Nine library versions were recorded
assert len(results["provenance"]["packages"]) == 9, results["provenance"]["packages"]
""")

step("3.14", "Record the data and split figures",
     "Record the figures from sections 1A to 1F in two blocks, `data` and `split`.",
     "why convert to plain Python types?",
     "JSON knows numbers, text, lists and dictionaries, not numpy or pandas types, so every value becomes a plain "
     "`int` or `float` first.")

code(r"""
# {index label as text: value as a plain float}, the shape the results file uses
def to_float_dict(series):
    out = {}
    for label, value in series.items():
        out[str(label)] = float(value)
    return out


# The data block: loans, censoring and poverty coverage
results["data"] = {
    "loans": len(df),
    "snapshot_end": str(snapshot_end.date()),
    "censor_days": CENSOR_DAYS,
    "recent_loans_excluded": int((~settled).sum()),
    "recent_loans_excluded_share": float((~settled).mean()),
    "settled_loans": int(settled.sum()),
    "settled_not_funded_share": float(1 - df.loc[settled, "fully_funded"].mean()),
    "funded_rate_by_age_days": to_float_dict(censoring["funded_rate"]),
    "modeling_rows": len(model_df),
    "mpi_matched_loans": int(df["MPI"].notna().sum()),
    "mpi_coverage": float(coverage),
    "mpi_located_regions": n_located,
    "mpi_misplaced_coordinates": n_misplaced,
}
# The split block: the three periods and their unfunded shares
results["split"] = {
    "train_share": TRAIN_SHARE,
    "fit_slice_end": str(val_cut.date()),
    "train_end": str(split_date.date()),
    "fit_loans": int(is_fit.sum()),
    "validation_loans": int(is_val.sum()),
    "train_loans": int(is_train.sum()),
    "test_loans": int((~is_train).sum()),
    "test_not_funded_loans": int((y_test == 0).sum()),
    "not_funded_share": {
        "fit": float(1 - y[is_fit].mean()),
        "validation": float(1 - y[is_val].mean()),
        "test": float(1 - y_test.mean()),
    },
    "feature_columns": len(ALL_COLUMNS),
}
# The data and split blocks recorded so far
print(type(results).__name__, list(results))
""")

check(r"""
# The data block holds every loan
assert results["data"]["loans"] == 671_205, results["data"]["loans"]
# The split block holds the test period
assert results["split"]["test_loans"] == 129_017, results["split"]["test_loans"]
""")

step("3.15", "Record the outcome exploration",
     "Record the unfunded shares by band and by group from section 1C.",
     "why save exploratory tables?",
     "The README quotes some of them (0.3% for the shortest terms), and a quoted figure must be in the file the "
     "test reads.")

code(r"""
# The unfunded share per repayment schedule, borrower group and sector (Steps 1.42 to 1.44)
not_funded_by_group = {}
# Build each column's table once more and keep its shares
for col in ("repayment_interval", "borrower_group", "sector"):
    not_funded_by_group[col] = to_float_dict(unfunded_by(col)["not_funded_share"])
# The outcome block
results["outcome_eda"] = {
    "settled_not_funded_share": float(overall_unfunded),
    "not_funded_by_term_band": to_float_dict(by_term["not_funded_share"]),
    "not_funded_by_amount_band": to_float_dict(by_amount["not_funded_share"]),
    "not_funded_by_group": not_funded_by_group,
}
# The outcome block's keys
print(list(results["outcome_eda"]))
""")

check(r"""
# Six term bands were saved
assert len(results["outcome_eda"]["not_funded_by_term_band"]) == 6, results["outcome_eda"]["not_funded_by_term_band"]
""")

step("3.16", "Record the model results",
     "Record the candidates, the headline metrics, the precision-recall points and the drivers.",
     "why record the candidates that lost?",
     "The README compares them with the chosen model, and the test checks every number the README quotes.")

code(r"""
# Each candidate's validation and test scores
candidate_results = {}
# One entry per candidate
for name in CANDIDATES:
    candidate_results[name] = {
        "validation_pr_auc_at_risk": float(validation_pr_auc[name]),
        "test_pr_auc_at_risk": float(test_pr_auc[name]),
        "test_pr_auc_funded": float(test_pr_auc_majority[name]),
    }
# Precision at each recall level, with the recall written as text ("0.25")
precision_at_recall_results = {}
# One entry per recall level
for target_recall in precision_at_recall:
    precision_at_recall_results[f"{target_recall:.2f}"] = precision_at_recall[target_recall]
# Precision and recall for the not-funded class at the 0.5 cut-off
at_risk_report = report["0"]
# The classification block
results["classification"] = {
    "candidates": candidate_results,
    "chosen_model": best_model_name,
    "chosen_on": "validation PR-AUC, at-risk class",
    "test_pr_auc_at_risk": float(pr_auc_minority),
    "test_pr_auc_funded": float(pr_auc_majority),
    "test_roc_auc": float(roc_auc),
    "test_funded_share": float(y_test.mean()),
    "at_risk_recall_at_0_5": float(at_risk_report["recall"]),
    "at_risk_precision_at_0_5": float(at_risk_report["precision"]),
    "top12_mean_abs_shap": to_float_dict(mean_abs_shap.head(12)),
    "at_risk_precision_at_recall": precision_at_recall_results,
}
# The headline as saved
print("test PR-AUC (at-risk):", results["classification"]["test_pr_auc_at_risk"])
""")

check(r"""
# The saved headline is the one printed in Step 2.12
assert results["classification"]["test_pr_auc_at_risk"] == float(pr_auc_minority)
""")

step("3.17", "Record the poverty results",
     "Record the exact-join and linked-score results from section 1E.",
     "why keep both poverty answers?",
     "The README reports how the answer changed when the wider score replaced the narrow one, so both must be on "
     "file.")

code(r"""
# The exact-join block
results["regional_poverty"] = {
    "regions_min_20_settled_loans": len(region_summary),
    "settled_loans_in_regions": int(region_summary["n_loans"].sum()),
    "settled_loans_in_regions_share": float(region_loan_share),
    "pearson_mpi_vs_share_funded": float(mpi_corr),
    "spearman_mpi_vs_share_funded": float(mpi_corr_rank),
}
# The five fifths' funded shares as plain floats
shares_by_fifth = []
# Convert each fifth's share
for fifth_share in funded_by_fifth["share_fully_funded"]:
    shares_by_fifth.append(float(fifth_share))
# The linked-score block
results["regional_poverty_linked"] = {
    "loans_with_linked_mpi_share": float(linked_coverage),
    "provinces_min_20_settled_loans": len(province_summary),
    "countries": int(province_summary["country"].nunique()),
    "settled_loans_in_provinces_share": float(linked_share),
    "pearson_mpi_vs_share_funded": float(linked_corr),
    "spearman_mpi_vs_share_funded": float(linked_corr_rank),
    "share_fully_funded_by_poverty_fifth": shares_by_fifth,
    "provinces_mapped": len(map_df),
}
# The fifths as saved
print(results["regional_poverty_linked"]["share_fully_funded_by_poverty_fifth"])
""")

check(r"""
# Five fifths were saved
assert len(shares_by_fifth) == 5, shares_by_fifth
""")

step("3.18", "Record the walk-forward and challenger results",
     "Record each walk-forward window and the challenger's decision.",
     "why save the decision rule with the result?",
     "A reader can then see that the rule was set before the test, not chosen to fit the outcome.")

code(r"""
# One entry per walk-forward window
windows = []
# Convert each window's row to plain types
for _, window_row in walk_forward.iterrows():
    windows.append({
        "test_window": window_row["test_window"],
        "train_loans": int(window_row["train_loans"]),
        "test_loans": int(window_row["test_loans"]),
        "not_funded_share": float(window_row["not_funded_share"]),
        "pr_auc_at_risk": float(window_row["pr_auc_at_risk"]),
        "lift_over_random": float(window_row["lift_over_random"]),
    })
# The walk-forward block
results["walk_forward"] = {
    "model": best_model_name,
    "windows": windows,
    "pr_auc_at_risk_min": float(walk_forward["pr_auc_at_risk"].min()),
    "pr_auc_at_risk_max": float(walk_forward["pr_auc_at_risk"].max()),
    "lift_over_random_min": float(walk_forward["lift_over_random"].min()),
    "lift_over_random_max": float(walk_forward["lift_over_random"].max()),
}
# The challenger block
results["linked_mpi_challenger"] = {
    "rule": "adopt if validation PR-AUC (at-risk) improves by at least 0.005",
    "validation_pr_auc_exact": float(validation_pr_auc[best_model_name]),
    "validation_pr_auc_linked": float(linked_val_pr_auc),
    "gain": float(linked_gain),
    "adopted": bool(adopt_linked),
}
# How many windows were saved, and the decision
print(len(windows), "windows | linked score adopted:", results["linked_mpi_challenger"]["adopted"])
""")

check(r"""
# Four windows were saved
assert len(results["walk_forward"]["windows"]) == 4, results["walk_forward"]["windows"]
# The challenger was not adopted
assert results["linked_mpi_challenger"]["adopted"] is False
""")

step("3.19", "Record days to fund, the drivers' direction, the segments and calibration",
     "Record the results of sections 2C, 2D and 3A not yet on file.",
     "what does `to_dict(orient=\"records\")` do?",
     "It turns a table into a list of dictionaries, one per row, which is how JSON stores a table.")

code(r"""
# The days-to-fund block
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
# The SHAP-direction block starts empty
results["shap_direction"] = {}
# One banded table per driver
for col in shap_direction:
    results["shap_direction"][col] = to_float_dict(shap_direction[col])
# The segment block starts empty
results["segments"] = {}
# One list of rows per segment column
for col in segments:
    results["segments"][col] = segments[col].to_dict(orient="records")
# The calibration block
results["calibration"] = {
    "brier_constant": float(brier_constant),
    "brier_raw": float(brier_raw),
    "brier_calibrated": float(brier_calibrated),
    "mean_predicted_raw": float(mean_predicted_raw),
    "mean_predicted_calibrated": float(mean_predicted_calibrated),
    "actual_unfunded_share": float(actual_unfunded_share),
    "pr_auc_raw": float(pr_auc_raw),
    "pr_auc_calibrated": float(pr_auc_calibrated),
    "mean_predicted_final_model": float(mean_predicted_final_model),
    "rank_corr_selection_vs_final": float(rank_corr_selection_vs_final),
}
# Every block recorded so far, in order
print(list(results))
""")

check(r"""
# Four segment tables were saved
assert list(results["segments"]) == ["borrower_group", "repayment_interval", "sector", "country"], list(results["segments"])
""")

step("3.20", "Record the priority list and the business impact",
     "Record the top five provinces and the shortfall table from section 3B.",
     "why save only the top five provinces?",
     "The README and the summary quote the top of the list; the full table is reproduced by the notebook itself.")

code(r"""
# The five highest-priority provinces
top5 = []
# Convert each of the first five rows to plain types
for _, province_row in priority.head(5).iterrows():
    top5.append({
        "country": province_row["country"],
        "province": province_row["mpi_region"],
        "n_test_loans": int(province_row["n_test_loans"]),
        "MPI": float(province_row["MPI"]),
        "mean_model_score": float(province_row["mean_model_score"]),
        "actual_not_funded_share": float(province_row["actual_not_funded_share"]),
        "priority_index": float(province_row["priority_index"]),
    })
# The priority block
results["priority_regions"] = {
    "poverty_score": "Kiva region link (province)",
    "min_test_loans": 30,
    "provinces": len(priority),
    "countries": int(priority["country"].nunique()),
    "risk_vs_actual_spearman": float(risk_agreement),
    "top5": top5,
}
# The shortfall block
results["shortfall"] = {
    "test_total_usd": float(total_shortfall),
    "flagged_loans": int(flagged.sum()),
    "flagged_share": float(flagged.mean()),
    "flagged_shortfall_usd": float(flagged_cover),
    "flagged_shortfall_share": float(flagged_cover / total_shortfall),
    "capture": capture.to_dict(orient="records"),
}
# All the blocks, in order
print(list(results))
""")

check(r"""
# Fifteen blocks in all
assert len(results) == 15, f"expected 15 blocks, got {len(results)}"
# The top province is Moyamba
assert results["priority_regions"]["top5"][0]["province"] == "Moyamba, Sierra Leone"
""")

step("3.21", "Write the results file",
     "Write `results` to `outputs/results.json`.",
     "what does `default=` do in `json.dump`?",
     "It is called for any value JSON cannot write on its own; here it turns a numpy number into a plain Python "
     "number.")

code(r"""
# numpy numbers are not JSON types; .item() turns one into a plain Python number
def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


# Make sure the outputs folder exists
Path("outputs").mkdir(exist_ok=True)
# Write the file
with open("outputs/results.json", "w", encoding="utf-8") as results_file:
    json.dump(results, results_file, indent=2, default=json_default)

# The file's size on disk
print("Wrote outputs/results.json,", Path("outputs/results.json").stat().st_size, "bytes")
""")

check(r"""
# This run wrote the file
assert Path("outputs/results.json").stat().st_mtime > RUN_STARTED, "outputs/results.json was not written by this run"
""")

step("3.22", "Read the file back",
     "Reload `outputs/results.json` and compare headline values with the variables that produced them.",
     "why read back what you just wrote?",
     "It proves the file on disk, the one the README test reads, holds the numbers this run computed.")

code(r"""
# Read the results file from disk
with open("outputs/results.json", encoding="utf-8") as saved_file:
    saved_results = json.load(saved_file)

# What was read and two of its headline values
print(type(saved_results).__name__, "with", len(saved_results), "blocks | test PR-AUC (at-risk):",
      saved_results["classification"]["test_pr_auc_at_risk"], "| total shortfall:", saved_results["shortfall"]["test_total_usd"])
""")

check(r"""
# The headline PR-AUC survived the round trip
assert saved_results["classification"]["test_pr_auc_at_risk"] == float(pr_auc_minority), "PR-AUC differs on disk"
# So did the total shortfall
assert saved_results["shortfall"]["test_total_usd"] == float(total_shortfall), "shortfall differs on disk"
# And the loan count
assert saved_results["data"]["loans"] == 671_205, "loan count differs on disk"
""")

# =====================================================================
# CLOSING
# =====================================================================
md("""
---
## What you built

- **A clean loan table** from three raw files, with tested borrower parsing, a funding target, two poverty scores and
  a censoring rule that keeps loans still raising money out of the outcome (Part 1).
- **A leakage guard** that stops the four post-funding columns reaching the model, run before and after encoding, and
  shown catching a deliberate leak (section 1F).
- **A baseline and four candidate models**, chosen on a validation slice, and a challenger judged by a rule fixed in
  advance (sections 2A and 2B).
- **A fair evaluation**: one score on later, unseen loans, the full precision-recall trade-off, a walk-forward check
  over time, a check by segment and by borrower group on the review list, and a calibration test that showed the
  scores should be used as a ranking (section 2C).
- **A second model** for days to fund, compared with its own baseline (section 2D).
- **Results for a decision-maker**: the drivers, a province priority list, a shortfall table with precision next to
  recall, and a results file that every README number is tested against (Part 3).

### The answers

**Can the loans that will fall short be spotted when they are posted?** Yes, well enough to prioritise. The chosen
model, LightGBM with a time trend, reaches a test PR-AUC of 0.374 on the at-risk class against a 4.6% base rate, about
eight times a random ranking, and holds between 0.394 and 0.503 across four later walk-forward windows. Reviewing the
riskiest 10% of new loans would reach 75.9% of the dollars that went unfunded, with 28.7% of the reviewed loans needing
help. The risk comes mainly from the loan's own terms: longer and larger loans are much more likely to fall short.

**Does the risk fall hardest on the poorest regions?** No. On the province link, which covers most loans, poverty and
funding are only weakly related, and both the least poor and the poorest fifth of loans are funded less often than the
middle. The province priority list still picks out places where deep poverty and high risk meet, such as Moyamba in
Sierra Leone and Mopti in Mali, as a starting point for review rather than an allocation rule.

### Limitations

- **Censoring is handled, not removed.** Loans posted within 60 days of the snapshot are excluded because their
  outcome was not yet known (Step 1.35). A few older loans may still have been fundraising.
- **One test window for the headline, checked by walk-forward.** The walk-forward test (Step 2.17) re-scores the
  chosen specification; it does not repeat the model choice in each window.
- **Trend and season.** A tree model cannot extend a trend beyond the training period, so the trend column can only
  hold the latest level steady.
- **Scores are not probabilities, and calibration did not transfer** (Steps 2.22 to 2.27): the unfunded share halved
  between the validation slice and the test period.
- **The model is weaker for some groups.** All-male and bullet loans, and some countries, are ranked less well
  (Step 2.20), and the review list reaches their unfunded loans at different rates (Step 2.21). Check this before the
  score decides who receives help.
- **Segment results are noisy at the edges.** Segments just above the 50-unfunded-loan floor carry wide uncertainty,
  and lift is not comparable across segments with different unfunded shares.
- **Days to fund is truncated.** Only loans funded by the snapshot have a value, so the test target leans towards
  fast funding.
- **The exact poverty join covers few loans** (Step 1.49), so the `MPI` model feature describes a small, non-random
  subset; the wider link is approximate, assigned by Kiva by nearest point.
- **The poverty correlations are ecological.** They are measured across regions or provinces, so they say nothing
  about whether any individual poorer borrower is more or less likely to be funded (the ecological fallacy).
- **Correlational, not causal.** Nothing here shows that any feature causes funding success or delay.
- **English-only text mining.** Non-English descriptions add noise to the term list.

### Next steps

For each idea, decide the rule for success **before** looking at the result, as in Step 2.9, and judge it on the
validation slice, never the test period.

1. **Tune LightGBM.** Try a few values of `learning_rate`, `n_estimators` and `num_leaves` on the validation slice.
2. **Recalibrate on recent loans.** Fit the calibrator on the last two months of the training period and compare the
   Brier score with Step 2.24.
3. **Drop the time trend.** Refit plain LightGBM and repeat Steps 2.20 and 2.21: do the weak groups change?
4. **Add a posting-time feature,** such as the number of loans posted the same week; it must be knowable on the day a
   loan is posted, and the leakage guard must still pass.
5. **Review each borrower group's riskiest loans separately** and compare recall and precision with Step 2.21.

## Glossary

| Term | Meaning | Where |
|---|---|---|
| Field partner | The local lender that posts a loan on Kiva and disburses it | Opening |
| Wrangle function | A function that reads a raw file and returns clean data | Section 1A |
| MPI | Multidimensional Poverty Index (OPHI and UNDP), 0 to 1, higher is poorer | Step 1.7 |
| Right-censoring | An outcome cut off by the end of the data | Step 1.35 |
| Settled loan | A loan posted more than 60 days before the end of the data, so its outcome is known | Step 1.35 |
| Base rate | The share of loans with the outcome you are looking for | Step 1.36 |
| Leakage | Training on information not available when the model is used | Step 1.58 |
| Time-based split | Train on earlier loans, test on later ones | Step 1.61 |
| One-hot encoding | One 0/1 column per category value | Step 1.63 |
| Precision, recall, PR-AUC | Share of flagged loans that went unfunded; share of unfunded loans flagged; the trade-off in one number | Step 2.1 |
| Baseline | A model that makes one prediction for every loan | Step 2.3 |
| Validation slice | Recent training loans used to choose the model | Step 2.6 |
| ROC-AUC | Chance an unfunded loan is ranked riskier than a funded one; random is 0.5 | Step 2.12 |
| Walk-forward | Retrain and score window after window through time | Step 2.17 |
| Fairness gap | A difference in how well a model works for different groups | Step 2.20 |
| Calibration, Brier score | Whether scores read as probabilities; mean squared error of those probabilities | Steps 2.22 and 2.24 |
| MAE, R-squared | Average error in days; share of variation explained | Step 2.31 |
| SHAP | Each feature's contribution to one prediction | Step 3.2 |
| Shortfall | The part of a loan's request that was never funded | Step 3.9 |
| Ecological fallacy | Reading an area-level pattern as true of individuals | Limitations |
""")

# =====================================================================
# WRITE THE NOTEBOOK
# =====================================================================
# Number the cells so a rebuild gives the same ids and a clean git diff
for cell_number, cell in enumerate(cells):
    cell["id"] = f"cell-{cell_number:03d}"
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK}: {len(cells)} cells")
