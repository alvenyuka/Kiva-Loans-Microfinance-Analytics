# Kiva Loans Microfinance Analytics: full methodology

> The detailed write-up behind the short [README](../README.md): method, every result, tests, and known limitations.

Funding-risk model on 671,205 Kiva microloans: which loans are at risk of not being funded, and does that risk fall hardest on the poorest regions? On an out-of-time test set of the 129,017 most recent loans, LightGBM reaches 0.389 PR-AUC on the at-risk minority class, explained with SHAP.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#tech-stack)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)](#tech-stack)
[![tests](https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/actions/workflows/ci.yml)

## Question

**Which loans are at risk of not getting fully funded, and does that risk fall hardest on the poorest regions?** This notebook analyzes 671k+ real microloans from Kiva's public dataset (Kaggle's "Data Science for Good: Kiva Crowdfunding"). It runs exploratory analysis, text mining of loan-use descriptions, and a geospatial join against a region-level poverty index, then builds a funding-risk classification model explained with SHAP, a days-to-fund regression, and a final synthesis flagging regions that are both poverty-deep and funding-at-risk. All findings are correlational, not causal; this is a single-snapshot dataset.

## Project Structure

```
Kiva-Loans-Microfinance-Analytics/
├── Kiva_Loans_Microfinance_Analytics.ipynb   # the notebook, run end to end
├── build_notebook.py                         # generates the notebook, edit this not the .ipynb
├── src/
│   └── features.py                           # leakage guard, funding target, gender parsing
├── tests/
│   └── test_features.py                      # leakage guard, target, gender parsing; no dataset needed
├── figs/
│   ├── geo_funding_vs_poverty.png            # Section 5 geospatial/MPI figure
│   └── shap_summary.png                      # Section 7 SHAP summary plot
├── docs/METHODOLOGY.md                       # this file
├── .github/workflows/ci.yml                  # runs the tests on every push
├── conftest.py                               # puts src/ on sys.path for the tests
├── pytest.ini
├── requirements.txt
├── LICENSE
└── README.md
```

The notebook is generated, so `build_notebook.py` is the file to edit, never the
`.ipynb`. The logic the conclusions rest on has been moved into `src/features.py`
so the notebook and the test suite exercise the same code rather than two copies
that drift apart.

## Quick Start

1. Download the two source CSVs (see Dataset below) into `./data`, or put them anywhere and set `KIVA_DATA_DIR` to that folder.
2. `pip install -r requirements.txt`
3. `python -m pytest` to check the leakage guard and feature logic. This needs no data and takes a few seconds.
4. Run `build_notebook.py`, then execute the generated notebook end to end (full commands under Running it below). Note that step 4 overwrites the committed notebook with a fresh, output-free copy, so its stored outputs are gone until you execute it.
5. Section 7 (funding-risk model + SHAP) has the headline result; Section 9 has the region-priority synthesis.

## Features

- **Exploratory analysis** across loan amount, sector, country, and borrower gender composition (Section 3).
- **Text mining** of the free-text `use` field via TF-IDF, producing coherent per-sector vocabulary (Section 4).
- **Geospatial join** against Kiva's region-level Multidimensional Poverty Index, with coverage reported and mis-keyed coordinates removed (Section 5).
- **Leakage-checked feature engineering**: outcome-dependent columns (`funded_time`, `disbursed_time`, `lender_count`, `funded_amount`) explicitly excluded because they aren't knowable at posting time (Section 6).
- **Censoring-aware target and out-of-time evaluation**: loans whose outcome was not yet known are excluded, and models are scored on loans posted after the training period (Section 6).
- **Funding-risk classification** (Logistic Regression, Random Forest, LightGBM compared), evaluated on minority-class PR-AUC, because the majority-class number looks good regardless of whether the model works (Section 7).
- **Days-to-fund regression** on funded loans (Section 8).
- **Region-priority synthesis** combining predicted funding risk with MPI poverty depth (Section 9).

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| ML | scikit-learn, LightGBM |
| Explainability | SHAP |
| Text mining | scikit-learn TF-IDF |
| Geospatial / visualization | Plotly, Kaleido, Matplotlib, Seaborn |
| Notebook | Jupyter, nbformat, nbclient |

## Dataset

Kaggle's "Data Science for Good: Kiva Crowdfunding" dataset: 671,205 real microloans
(`kiva_loans.csv`), joined against Kiva's region-level Multidimensional Poverty Index
(`kiva_mpi_region_locations.csv`). Not included in this repo (see `.gitignore`).
Download from:
https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding

## Methodology

- **EDA (Section 3):** loan amounts are right-skewed (median $500, mean $842, max
  $100,000); the three largest sectors are Agriculture (180,302 loans), Food (136,657),
  and Retail (124,494); the top countries by volume are the Philippines (160,441),
  Kenya (75,825), and El Salvador (39,875); borrower gender composition skews female
  (median 100% female borrowers per loan among loans with at least one parsed borrower).
- **Text mining (Section 4):** TF-IDF (30 terms, 1-2 grams, `min_df=50`) on the free-text
  `use` field produces term-presence features later fed into the model; per-sector
  vocabulary is coherent and non-degenerate (Agriculture -> farm/fertilizer/seeds,
  Food -> fish/rice/ingredients, Personal Use -> drinking/filter/solar/water).
- **Geospatial + MPI join (Section 5):** each loan's `country` + `region` is joined
  against Kiva's region-to-MPI lookup table. Coverage is low: only **7.6%** of loans
  matched (50,955 / 671,205), so every MPI-dependent result describes only that
  matched subset, not the full dataset (see Known Limitations). Before the join, 94
  of the 892 located regions are flagged as mis-keyed in the upstream file: a region
  whose latitude or longitude sits more than five robust standard deviations (and at
  least 10 degrees) from its country's median is dropped from the map, for example
  Kenya / Rift Valley at -18.4, 47.3, which is in Madagascar. Their MPI values, which
  the model uses, are kept.
- **Censoring-aware target (Section 6.1):** the snapshot ends 2017-07-26 and Kiva
  loans fundraise for weeks, so a loan posted near that date with no `funded_time`
  is "not funded yet", not "not funded". The notebook prints the funded rate by
  posting age: 93.5% for loans posted more than 90 days before the snapshot, 89.3% at
  46 to 60 days, 74.0% at 31 to 45 days, 34.8% at 15 to 21 days and 16.4% in the
  final week. Loans posted within 60 days of the snapshot (26,122, or 3.9%) are
  excluded, leaving 645,083 loans with a settled outcome, 6.4% of them not funded.
- **Feature engineering with an explicit leakage check (Section 6.1):** `funded_time`,
  `disbursed_time`, `lender_count`, and `funded_amount` are excluded because they are
  consequences of a loan being funded, not knowable at posting time. The guard runs
  before and after one-hot encoding. The model matrix has 304 columns: posting-time
  loan and borrower fields, the MPI join, and TF-IDF term-presence flags.
- **Time-based split, preprocessing fitted on train (Section 6.2):** the earliest 80%
  of settled loans by posting date (516,066, up to 2016-10-27) train the models and the
  most recent 20% (129,017) test them. The TF-IDF vocabulary and the `MPI` and
  `pct_female` medians used for missing values are fitted on the training period only.
- **Funding-risk model + SHAP (Section 7):** Logistic Regression, Random Forest, and
  LightGBM are compared on `fully_funded` using **both** majority-class and
  minority-class PR-AUC, since the notebook's actual question, which loans are at
  risk of *not* being funded, is about the minority class. LightGBM wins on both and
  is explained with SHAP.
- **Days-to-fund regression (Section 8):** among settled loans that did get funded, a
  Random Forest regresses `days_to_fund = funded_time - posted_time` on the same
  posting-time features, with the same date cut-off separating train and test.
- **Synthesis (Section 9):** the funding-risk model's predicted funding probability on
  the test set is aggregated to region level and combined with each region's MPI into
  a `priority_score = MPI * (1 - mean predicted funding probability)`, ranking regions
  that are both poverty-deep and funding-at-risk.

## Results

All figures are from the executed notebook, on the out-of-time test set (129,017
loans posted after 2016-10-27, of which 5,957, or 4.6%, were not funded).

| Model | PR-AUC, at-risk (minority) class | PR-AUC, funded (majority) class |
|---|---:|---:|
| **LightGBM** | **0.3890** | 0.9958 |
| Random Forest | 0.2838 | 0.9942 |
| Logistic Regression | 0.2366 | 0.9874 |

- **Funding-risk model:** LightGBM is the best model, with ROC-AUC 0.9210. On the
  at-risk class at the default threshold it has precision 0.24 and recall 0.76. The
  majority-class PR-AUC sits close to its trivial floor (0.954, the funded share) and
  is reported for completeness only.
- **Effect of the stricter evaluation:** an earlier version of this notebook, with a
  random split, preprocessing fitted on all rows and the censored loans included,
  reported 0.4889. The drop to 0.3890 is the cost of an honest test, and 0.389 is
  the figure to use.
- **MPI join coverage: 7.6%** (50,955 / 671,205 loans matched to a region-level MPI
  score). Across the 76 regions with at least 20 loans and a known MPI, the
  correlation between MPI and the share of loans fully funded is 0.253.
- **Top SHAP drivers of funding risk** (LightGBM, ranked by mean |SHAP value|):
  `term_in_months`, `loan_amount`, `post_month`, `pct_female`, `n_female`,
  `sector_Education`, `sector_Retail`, `country_Cambodia`,
  `repayment_interval_monthly`, `country_Kenya`, `sector_Arts`, `sector_Food`.
- **Days-to-fund regression:** MAE = 7.04 days, R² = 0.248, on 604,019 settled funded
  loans (mean days-to-fund 14.7, standard deviation 14.5). Scored out of time, the
  regression explains a quarter of the variance, so it gives a rough expected
  fundraising time rather than a forecast to plan around.
- **Priority-regions finding:** 57 regions qualify for the priority table (at least 10
  test loans with a known MPI). The highest-priority regions are Timor-Leste /
  Viqueque (MPI 0.410, priority score 0.134), Timor-Leste / Baucau, Nigeria / Kaduna
  (1,239 test loans, the only large sample near the top), Sierra Leone / Bo and
  Guatemala / Quiche. Most top regions rest on 14 to 40 test loans and cover only
  the 7.6% of loans with an MPI match, so the list illustrates the method rather than
  a reliable targeting list.

## Known Limitations

**Censoring is handled, not eliminated.** Excluding loans posted within 60 days of
the snapshot removes the loans whose outcome was plainly unknown (the funded rate
recovers to 96.4% at 61 to 90 days). A few older loans may still have been
fundraising, so the at-risk class can hold a small residue of them.

**The leakage guard checks names, not meaning.** `assert_no_leakage` is a name check
over feature columns. It would not catch a renamed derived feature or a groupby
target encoding, and it never inspects the target definition, which is why the
censoring above had to be measured separately.

**The test period is one stretch of time.** Scoring on the latest 20% is the honest
form for a model that will score future loans, but it is a single out-of-time
window. Walk-forward windows would show how stable the 0.389 figure is.

**94 region coordinates in the upstream file are wrong.** The rule in Section 5
removes them from the map; their MPI values, which come from the same rows, are
kept on the assumption that the coordinates rather than the poverty scores were
mis-keyed. Sierra Leone / Port Loko and Timor-Leste / Aileu are among the dropped
points.

**MPI join coverage sits at just 7.6%** (50,955 / 671,205 loans). Most loans
could not be matched to a region-level MPI score, most likely because
`kiva_loans.csv`'s free-text `region` field (entered inconsistently by field
partners) doesn't standardize against the MPI lookup table's `region` field
well enough for an exact string join. Every MPI-dependent result, the Section 5
map and correlation, the `MPI` feature in the Section 7 model, and the Section
9 priority-regions table, describes only that small, non-random subset.

**The Section 5 MPI correlation is an ecological one.** The 0.253 figure is
computed across 76 regions, between a region's MPI and its percent funded. It
says nothing about whether any individual poorer borrower is less likely to be
funded.

**Correlational, not causal**, throughout: nothing here establishes that any
feature *causes* funding success or delay. And the text mining is English-only:
TF-IDF was fit on the raw `use` field without language detection, so
non-English descriptions contribute noise to the term list.

## Running it

Requires: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`,
`shap`, `plotly` (with `kaleido` for static map export), and `jupyter`/`nbconvert`.

```bash
pip install -r requirements.txt

# the tests: no dataset, a few seconds
python -m pytest

# the notebook
export KIVA_DATA_DIR=/path/to/kiva/csvs      # or put them in ./data
python build_notebook.py
jupyter nbconvert --to notebook --execute Kiva_Loans_Microfinance_Analytics.ipynb \
  --output Kiva_Loans_Microfinance_Analytics.ipynb
```

The tests take seconds and need nothing. The notebook takes roughly 15 minutes
end to end and needs about 2GB of free memory, most of it the 671,205-row loan
table and the TF-IDF matrix built from the `use` field.

`kiva_loans.csv` and `kiva_mpi_region_locations.csv` (see Dataset above) go in
`./data`, or anywhere you point `KIVA_DATA_DIR` at. The setup cell raises a named
error if it cannot find them, rather than failing several cells later with
something that looks like a data problem.

An earlier version of this repo hardcoded `../Data/Kiva`, which stopped resolving
once the folder moved. The notebook's stored outputs still looked fine, so nothing
surfaced the breakage until someone tried to re-run it. That is the reason the path
is configurable and checked now.

## Tests

```bash
python -m pytest        # about 2 seconds
```

The tests do not re-check the model's score. They check the three things whose
failure would leave the score looking perfectly reasonable:

| What is tested | Why it can break silently |
|---|---|
| **The leakage guard** (`assert_no_leakage`) | `funded_time`, `disbursed_time`, `lender_count` and `funded_amount` all exist only because a loan was funded. A model using them to predict funding is reading the answer off the back of the card, and the symptom is a *better* PR-AUC, not an error. Each leaky column is tested individually, so a newly added one cannot pass by hiding behind one already caught, and suffixed derivatives like `lender_count_log` are caught too, since one-hot encoding and binning rename columns. The guard runs twice in the notebook, before and after encoding. |
| **The funding target** (`mark_fully_funded`) | One line, and every figure in the project hangs off it. The test that matters is the overfunded loan: Kiva loans occasionally close slightly above the amount requested, and an equality test rather than `>=` would label those as failures. |
| **Borrower gender parsing** (`parse_gender_counts`) | `borrower_genders` is a comma-separated list because group loans are normal on Kiva, so the parsing has to count borrowers rather than rows. Missing values and unrecognised labels are counted as neither, not folded into one side, because the female share is a headline figure. A row that cannot be parsed gets `pct_female = NaN`, never 0, since 0 would assert an all-male loan and the model reads that column. |

This is not a hypothetical concern. An uncaught leak of exactly
this shape once inflated a model's R-squared from 0.906 to 0.996 in an earlier,
unpublished project of mine, which is why the guard is executed in the notebook
rather than described in a comment.

Three more tests cover a bug this repo actually hit. The feature-name constants
were tuples at first, and pandas reads a tuple as one compound key, so
`df[POSTING_TIME_FEATURES]` raised `KeyError` with all fourteen names as the key
instead of selecting fourteen columns. The message looks like missing data rather
than a type mistake, and it surfaced part-way through a full notebook run. The
constants are lists now, and the three call sites that depend on that are each
tested: column selection, list concatenation for the model matrix, and
`pd.get_dummies(columns=...)`.

Every test builds its own small frame, so CI runs them without the ~200MB of Kiva
CSVs. The analysis itself stays a local step.

## Roadmap

- [x] EDA, text mining, geospatial/MPI join
- [x] Leakage-checked funding-risk model (LightGBM, SHAP)
- [x] Days-to-fund regression
- [x] Region-priority synthesis
- [x] Drop loans posted within 60 days of the snapshot and restate the PR-AUC
- [x] Fit imputation and TF-IDF on the training period only
- [x] Time-based train/test split
- [x] Remove mis-keyed MPI coordinates before mapping
- [ ] Walk-forward evaluation over several later windows
- [ ] Improve MPI join coverage beyond 7.6% (fuzzy/normalized region matching)
- [ ] Language detection for non-English `use` text before TF-IDF

## License

MIT. See [`LICENSE`](../LICENSE).

## Credits

Author: **Alven Yuka**, CPA Finalist. Built on the [Kiva Loans / MPI dataset](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) (Kaggle).

