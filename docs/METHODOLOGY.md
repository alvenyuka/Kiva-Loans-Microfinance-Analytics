# Kiva Loans Microfinance Analytics: full methodology

> The detailed write-up behind the short [README](../README.md): method, every result, tests, and known limitations.

Funding-risk model on 671,205 Kiva microloans: which loans are at risk of not being funded, and does that risk fall hardest on the poorest regions? The model is chosen on a validation slice at the end of the training period; on an out-of-time test set of the 129,017 most recent loans, the chosen LightGBM reaches 0.374 PR-AUC on the at-risk minority class, explained with SHAP.

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
│   ├── features.py                           # leakage guard, funding target, gender parsing
│   └── impact.py                             # funding-shortfall capture (Section 9.1)
├── tests/
│   ├── test_features.py                      # leakage guard, target, gender parsing; no dataset needed
│   ├── test_impact.py                        # shortfall arithmetic behind the business-impact table
│   └── test_readme_numbers.py                # every README number appears in outputs/results.json
├── figs/
│   ├── geo_funding_vs_poverty.png            # Section 5 geospatial/MPI figure
│   ├── shap_summary.png                      # Section 7 SHAP summary plot
│   └── shortfall_capture.png                 # Section 9.1 shortfall captured by risk ranking
├── outputs/
│   └── results.json                          # every headline number, with package versions and git commit
├── docs/METHODOLOGY.md                       # this file
├── .github/workflows/ci.yml                  # runs the tests on every push
├── conftest.py                               # puts src/ on sys.path for the tests
├── pytest.ini
├── requirements.txt
├── LICENSE
└── README.md
```

The notebook is generated, so `build_notebook.py` is the file to edit, never the
`.ipynb`. It is written to be followed and recreated: three parts (prepare the
data, build the models, communicate the results), each section broken into small
numbered tasks with one short code cell each, a baseline before every model, and
"check your work" assertions that stop the run at the step that went wrong. The logic the conclusions rest on lives in `src/features.py` and
`src/impact.py`, so the notebook and the test suite exercise the same code rather
than two copies that drift apart.

## Quick Start

1. Download the two source CSVs (see Dataset below) into `./data`, or put them anywhere and set `KIVA_DATA_DIR` to that folder.
2. `pip install -r requirements.txt`
3. `python -m pytest` to check the leakage guard and feature logic. This needs no data and takes a few seconds.
4. Run `build_notebook.py`, then execute the generated notebook end to end (full commands under Running it below). Note that step 4 overwrites the committed notebook with a fresh, output-free copy, so its stored outputs are gone until you execute it.
5. Section 7 (funding-risk model + SHAP) has the headline result; Section 9 has the region-priority synthesis; `outputs/results.json` holds every headline number with the package versions and git commit of the run.

## Features

- **Exploratory analysis** across loan amount, sector, country, and borrower gender composition (Section 3).
- **Text mining** of the free-text `use` field via TF-IDF, producing coherent per-sector vocabulary (Section 4).
- **Geospatial join** against Kiva's region-level Multidimensional Poverty Index, with coverage reported and mis-keyed coordinates removed (Section 5).
- **Leakage-checked feature engineering**: outcome-dependent columns (`funded_time`, `disbursed_time`, `lender_count`, `funded_amount`) explicitly excluded because they aren't knowable at posting time (Section 6).
- **Censoring-aware target and out-of-time evaluation**: loans whose outcome was not yet known are excluded, and models are scored on loans posted after the training period (Section 6).
- **Funding-risk classification** (Logistic Regression, Random Forest, LightGBM, and LightGBM with a time trend), chosen on a validation slice and evaluated on minority-class PR-AUC, because the majority-class number looks good regardless of whether the model works (Section 7).
- **Days-to-fund regression** on funded loans (Section 8).
- **Region-priority synthesis** combining funding-risk and poverty ranks into a relative priority index (Section 9).
- **Results file** with provenance, checked against the README by a test (Section 11).

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
  `use` field shows the common vocabulary. This exploratory vocabulary is fitted on
  all loans and is not a model input; the model's term-presence flags are refitted
  on training loans only (Section 6.2). Per-sector vocabulary is coherent and
  non-degenerate (Agriculture -> farm/fertilizer/seeds,
  Food -> fish/rice/ingredients, Personal Use -> drinking/filter/solar/water).
- **Geospatial + MPI join (Section 5):** each loan's `country` + `region` is joined
  against Kiva's region-to-MPI lookup table. Coverage is low: only **7.6%** of loans
  matched (50,955 / 671,205), so every MPI-dependent result describes only that
  matched subset, not the full dataset (see Known Limitations). Before the join, 94
  of the 892 located regions are flagged as mis-keyed in the upstream file: a region
  whose latitude or longitude sits more than five robust standard deviations (and at
  least 10 degrees) from its country's median is dropped from the map, for example
  Brazil / Rondônia at -17.8, 31.0, which is in Zimbabwe. Their MPI values, which
  the model uses, are kept. The same file stores some names mis-encoded upstream
  ("Maranhðo", "Rondðnia" for Maranhão and Rondônia); the file is valid UTF-8, so
  this is a defect in the source, and the names are printed as stored.
- **Does funding risk fall hardest on the poorest regions? (Section 5.1):** across the
  75 regions with at least 20 settled loans and a known MPI (48,590 loans, 7.5% of
  settled loans), the correlation between MPI and the share of loans fully funded is
  0.303 (Pearson) and 0.375 (Spearman). Both are positive: in this matched subset,
  poorer regions were funded slightly more often, not less. The correlation is
  ecological, so it does not show that poorer borrowers are favoured.
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
  before and after one-hot encoding. The model matrix has 306 columns: posting-time
  loan and borrower fields, the MPI join with an `MPI_missing` flag, TF-IDF
  term-presence flags, and `months_since_start`, a time-trend column (months since
  the first loan in the data) used only by the time-trend candidate.
- **Time-based split, preprocessing fitted on earlier loans (Section 6.2):** the
  earliest 80% of settled loans by posting date (516,066, up to 2016-10-27) form the
  training period and the most recent 20% (129,017) the test period. The training
  period is split again by date: a fit slice (412,853 loans, up to 2016-04-23) and a
  validation slice (103,213 loans). The TF-IDF vocabulary and the `MPI` and
  `pct_female` medians are fitted on the fit slice for model selection and on the
  whole training period for the final model.
- **Funding-risk model + SHAP (Section 7):** Logistic Regression, Random Forest,
  LightGBM, and LightGBM with the time-trend column are fitted on the fit slice and
  compared on the validation slice by minority-class PR-AUC, since the notebook's
  actual question, which loans are at risk of *not* being funded, is about the
  minority class. The winner is refitted on the whole training period and scored
  once on the test period; the other candidates are refitted and scored for
  reference only. All use balanced class weights, so their outputs are ranking
  scores, not calibrated probabilities. The chosen model is explained with SHAP.
- **Days-to-fund regression (Section 8):** among settled loans that did get funded, a
  Random Forest regresses `days_to_fund = funded_time - posted_time` on the same
  posting-time features, with the same date cut-off separating train and test, and is
  compared with a naive forecast that gives every test loan the training-period
  median.
- **Synthesis (Section 9):** the chosen model's test-period scores are averaged by
  region. Because the scores are not calibrated, they are not multiplied by MPI.
  Each region gets a percentile rank on poverty (higher MPI ranks higher) and on
  funding risk (lower mean score ranks higher), and the relative priority index is
  the product of the two: 1 is the poorest and riskiest region in the table.

## Results

All figures are from the executed notebook and `outputs/results.json`, on the
out-of-time test set (129,017 loans posted after 2016-10-27, of which 5,957, or 4.6%,
were not funded). The validation slice had 9.6% not funded.

| Model | Validation PR-AUC, at-risk | Test PR-AUC, at-risk | Test PR-AUC, funded |
|---|---:|---:|---:|
| **LightGBM + time trend (chosen)** | **0.4902** | **0.3742** | 0.9958 |
| LightGBM | 0.4860 | 0.3859 | 0.9958 |
| Random Forest | 0.3951 | 0.2857 | 0.9941 |
| Logistic Regression | 0.3293 | 0.2366 | 0.9875 |

- **Funding-risk model:** LightGBM with the time-trend column is chosen on the
  validation slice, ahead of plain LightGBM by 0.004. On the test period it has
  PR-AUC 0.3742 and ROC-AUC 0.9196; at the default score threshold of 0.5 its
  precision on the at-risk class is 0.19 and its recall 0.86. Plain LightGBM scores
  0.3859 on the test period; the order of the two swaps between validation and test,
  so the difference between them is within the noise of one window. The published
  figure stays with the model chosen before the test set was scored. The
  majority-class PR-AUC sits close to its floor (0.954, the funded share of the test
  set) and is reported for completeness only.
- **MPI join coverage: 7.6%** (50,955 / 671,205 loans matched to a region-level MPI
  score). Across the 75 regions with at least 20 settled loans and a known MPI, the
  correlation between MPI and the share of loans fully funded is 0.303 (rank
  correlation 0.375).
- **Top SHAP drivers of funding risk** (chosen model, mean |SHAP value| on 1,000 test
  loans, in log-odds): `term_in_months` 1.552, `loan_amount` 1.341,
  `months_since_start` 0.675, `post_month` 0.392, `pct_female` 0.224, `n_female`
  0.188, `sector_Education` 0.160, `sector_Retail` 0.150, `country_Cambodia` 0.090,
  `sector_Arts` 0.076, `country_Kenya` 0.071, `sector_Food` 0.066.
- **Days-to-fund regression:** MAE 7.04 days against 8.08 days for the naive
  training-median forecast (10.2 days), R² = 0.248, on 604,019 settled funded loans
  (123,059 in the test period; mean days-to-fund 14.7, standard deviation 14.5).
  Scored out of time, the regression explains a quarter of the variance, so it gives
  a rough expected fundraising time rather than a forecast to plan around.
- **Business impact (Section 9.1):** the test period's loans fell $5,797,550 short of
  full funding. Reviewing loans riskiest first reaches 57.5% of that shortfall at 5% of
  loans, 75.9% at 10%, 92.9% at 20% and 97.7% at 30%; the loans flagged at the default
  threshold (26,388, or 20.5%) cover 93.2%.
- **Priority-regions finding:** 57 regions qualify for the priority table (at least 10
  test loans with a known MPI). The highest relative priority index goes to
  Timor-Leste / Viqueque (MPI 0.410, index 0.90), then Timor-Leste / Baucau, Nigeria /
  Kaduna (1,239 test loans, the only large sample near the top), Guatemala / Quiche
  and Sierra Leone / Bo. Most top regions rest on 14 to 38 test loans and cover only
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
window, and the validation slice before it had twice the unfunded share. Walk-forward
windows would show how stable the 0.374 figure is.

**Trend and season.** Under a time split, month of posting can stand in for a
platform-wide trend. The time-trend candidate gives the model that trend directly;
it ranks third in the SHAP table, behind term and amount, with posting month fourth.
A tree model cannot extrapolate the trend past the training period, so the column
can only hold the latest level steady for test loans.

**Scores are not probabilities.** Balanced class weights move every score towards
0.5. The scores rank loans and regions, which is how they are used; a probability
reading would need calibration on later data.

**Days-to-fund is truncated in the test period.** Only loans funded by the snapshot
have a days-to-fund value, and the most recent loans had the least time to fund
slowly, so the test target is biased towards short fundraising times. The
regression and its baseline share that bias.

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

**The Section 5 MPI correlation is an ecological one.** The 0.303 figure is
computed across 75 regions, between a region's MPI and its share of settled loans
fully funded. It says nothing about whether any individual poorer borrower is more
or less likely to be funded.

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
  --output Kiva_Loans_Microfinance_Analytics.ipynb --ExecutePreprocessor.timeout=3600
```

The tests take seconds and need nothing. The last recorded run of the notebook
took 26 minutes end to end and peaked at 4.1 GB of memory (both written to
`outputs/results.json`), most of it the 671,205-row loan table, the two encoded
model matrices and the scaled copy the logistic regression needs.

`kiva_loans.csv` and `kiva_mpi_region_locations.csv` (see Dataset above) go in
`./data`, or anywhere you point `KIVA_DATA_DIR` at. The setup cell raises a named
error if it cannot find them, rather than failing several cells later with
something that looks like a data problem.

## Tests

```bash
python -m pytest        # about 2 seconds
```

The tests do not re-check the model's score. They check the things whose failure
would leave the score looking perfectly reasonable:

| What is tested | Why it can break silently |
|---|---|
| **The leakage guard** (`assert_no_leakage`) | `funded_time`, `disbursed_time`, `lender_count` and `funded_amount` all exist only because a loan was funded. A model using them to predict funding is reading the answer off the back of the card, and the symptom is a *better* PR-AUC, not an error. Each leaky column is tested individually, so a newly added one cannot pass by hiding behind one already caught, and suffixed derivatives like `lender_count_log` are caught too, since one-hot encoding and binning rename columns. The guard runs twice in the notebook, before and after encoding. |
| **The funding target** (`mark_fully_funded`) | One line, and every figure in the project hangs off it. The test that matters is the overfunded loan: Kiva loans occasionally close slightly above the amount requested, and an equality test rather than `>=` would label those as failures. |
| **Borrower gender parsing** (`parse_gender_counts`) | `borrower_genders` is a comma-separated list because group loans are normal on Kiva, so the parsing has to count borrowers rather than rows. Missing values and unrecognised labels are counted as neither, not folded into one side, because the female share is a headline figure. A row that cannot be parsed gets `pct_female = NaN`, never 0, since 0 would assert an all-male loan and the model reads that column. |

A leaked column does not break a model; it improves the score, so nothing
downstream would flag it. The guard is therefore executed in the notebook, not
described in a comment.

Three more tests cover a bug this repo actually hit. The feature-name constants
were tuples at first, and pandas reads a tuple as one compound key, so
`df[POSTING_TIME_FEATURES]` raised `KeyError` with all fourteen names as the key
instead of selecting fourteen columns. The message looks like missing data rather
than a type mistake, and it surfaced part-way through a full notebook run. The
constants are lists now, and the three call sites that depend on that are each
tested: column selection, list concatenation for the model matrix, and
`pd.get_dummies(columns=...)`.

`tests/test_impact.py` checks the shortfall arithmetic behind the business-impact
table, and `tests/test_readme_numbers.py` checks that every number quoted in the
README appears in `outputs/results.json`, which the notebook writes at the end of
each run, so a stale README figure fails the build.

Every test builds its own small frame or reads the committed results file, so CI
runs them without the ~200MB of Kiva CSVs. The analysis itself stays a local step.

## Roadmap

- [x] EDA, text mining, geospatial/MPI join
- [x] Leakage-checked funding-risk model (LightGBM, SHAP)
- [x] Days-to-fund regression
- [x] Region-priority synthesis
- [x] Drop loans posted within 60 days of the snapshot and restate the PR-AUC
- [x] Fit imputation and TF-IDF on the training period only
- [x] Time-based train/test split
- [x] Remove mis-keyed MPI coordinates before mapping
- [x] Choose the model on a validation slice; write results.json with provenance
- [ ] Walk-forward evaluation over several later windows
- [ ] Improve MPI join coverage beyond 7.6% (fuzzy/normalized region matching)
- [ ] Language detection for non-English `use` text before TF-IDF

## License

MIT. See [`LICENSE`](../LICENSE).

## Credits

Author: **Alven Yuka**, CPA Finalist. Built on the [Kiva Loans / MPI dataset](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) (Kaggle).

