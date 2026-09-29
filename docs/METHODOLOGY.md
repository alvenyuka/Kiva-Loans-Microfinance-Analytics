# Kiva Loans Microfinance Analytics: full methodology

> The detailed write-up behind the short [README](../README.md): method, every result, tests, and known limitations. Moved here unchanged on 2026-09-29 when the README was shortened.

> Funding-risk model on 671,205 real Kiva microloans: which loans are at risk of not being funded, and does that risk fall hardest on the poorest regions? LightGBM reaches 0.4889 PR-AUC on the at-risk minority class, explained with SHAP.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#tech-stack)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)](#tech-stack)
[![PR-AUC (at-risk)](https://img.shields.io/badge/PR--AUC%20(at--risk)-0.4889-success)](#results)
[![tests](https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/actions/workflows/ci.yml)

## Why?

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
- **Geospatial join** against Kiva's region-level Multidimensional Poverty Index, with honest coverage reporting (Section 5).
- **Leakage-checked feature engineering**: outcome-dependent columns (`funded_time`, `disbursed_time`, `lender_count`, `funded_amount`) explicitly excluded because they aren't knowable at posting time (Section 6).
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
  matched subset, not the full dataset (see Known Limitations).
- **Feature engineering with an explicit leakage check (Section 6):** `funded_time`,
  `disbursed_time`, `lender_count`, and `funded_amount` are excluded because they are
  consequences of a loan being funded, not knowable at posting time. The model uses
  44 posting-time features (including the MPI join and TF-IDF term-presence columns),
  one-hot encoded to 305 columns.
- **Funding-risk model + SHAP (Section 7):** Logistic Regression, Random Forest, and
  LightGBM are compared on `fully_funded` (92.8% funded / 7.2% not funded) using
  **both** majority-class and minority-class PR-AUC, since the notebook's actual
  question, which loans are at risk of *not* being funded, is about the minority
  class; LightGBM wins on both metrics and is explained with SHAP.
- **Days-to-fund regression (Section 8):** among loans that did get funded, a Random
  Forest regresses `days_to_fund = funded_time - posted_time` on the same posting-time
  feature set.
- **Synthesis (Section 9):** the funding-risk model's predicted funding probability is
  aggregated to region level and combined with each region's MPI into a
  `priority_score = MPI * (1 - mean predicted funding probability)`, ranking regions
  that are both poverty-deep and funding-at-risk.

## Results

- **Funding-risk model:** LightGBM is the best model. On the headline metric,
  **PR-AUC (at-risk, minority class) = 0.4889**, it substantially outperforms Random
  Forest (0.3580) and Logistic Regression (0.2731) on this metric. (Majority-class
  PR-AUC is 0.9937 and ROC-AUC is 0.9241, but both sit close to the majority class's
  trivial-baseline floor and are reported for completeness, not as the headline;
  the minority-class number is the one that answers the "at risk of not
  being funded" question.) On the minority class specifically: precision 0.25,
  recall 0.91.
- **MPI join coverage: 7.6%** (50,955 / 671,205 loans matched to a region-level MPI
  score).
- **Top SHAP drivers of funding risk** (LightGBM, ranked by mean |SHAP value|):
  `term_in_months`, `loan_amount`, `post_month`, `pct_female`, `sector_Retail`,
  `sector_Education`, `n_female`, `country_Kenya`, `repayment_interval_irregular`,
  `repayment_interval_monthly`, `country_Philippines`, `country_Peru`.
- **Days-to-fund regression:** MAE = 7.43 days, R² = 0.435, on 622,873 funded loans
  (mean days-to-fund 14.6, std 14.4).
- **Priority-regions finding:** 68 regions qualify for the priority table (>=10
  held-out test loans with a known MPI). The highest-priority region is
  Timor-Leste/Aileu (MPI 0.379, priority score 0.1996), followed by
  Timor-Leste/Viqueque, Timor-Leste/Ermera, Sierra Leone/Kenema, and Sierra
  Leone/Bo, so Timor-Leste and Sierra Leone regions dominate the top of the
  list, combining high poverty depth with lower model-predicted funding
  probability. Given the 7.6% MPI coverage, this list should be read as
  illustrative of the method, not as a reliable region-targeting list for the
  92.4% of loan volume that couldn't be matched to an MPI score.

## Known Limitations

Read these before trusting any result above too far. The first one moves the
headline number.

**About one in eight of the minority-class labels is a censoring artifact.**
Loans with no `funded_time` when the dataset was captured are labelled "not
fully funded", but Kiva loans fundraise for weeks after posting, so a loan
posted near the snapshot boundary is not "not funded", it is "not funded yet".
The snapshot ends 2017-07-26, and the funded rate collapses as that date
approaches (figures in this paragraph were measured offline against the source CSV and are not yet
reproducible from this repo; no notebook cell computes them): 93.6% for loans posted more than 90 days before it, 89.3% at 46 to
60 days, 74.0% at 31 to 45 days, 34.8% at 15 to 21 days, 16.4% in the final
week. **12.7% of all 48,328 not-funded labels are loans posted within the final
45 days.** The headline metric is minority-class PR-AUC, so roughly one in
eight of the positives it scores on is this artifact rather than a funding
failure. Filtering out loans posted within the last 60 days would cost about
3.8% of rows and give a defensible target. It is not done here, and 0.4889
should be read with that in mind.

**The leakage guard cannot catch that, by design.** `assert_no_leakage` is a
name check over feature columns. The problem above is in the target definition,
which the guard never inspects. The same blind spot covers a renamed derived
feature and any groupby target encoding. The guard is good at what it does; it
is worth knowing what it does not look at.

**Preprocessing is fit before the split.** The `MPI` and `pct_female` medians
and the TF-IDF vocabulary and IDF weights are all computed over train and test
together. A median barely moves for a handful of extra rows, so the effect is
small, but the correct form is a pipeline fit on train only.

**The split is random on data that has a time dimension.** `posted_time`
exists, `post_month` and `post_dow` are features, and the target has the strong
time trend documented above. A train-before-a-cutoff, test-after split would be
the honest form and would report a lower number.

**Two mapped regions are plotted on the wrong continent.** In the upstream
`kiva_mpi_region_locations.csv`, Sierra Leone / Port Loko carries 5.557, 23.763
(Central African Republic) and Timor-Leste / Aileu carries 3.428, -76.487
(Colombia), while every other Timor-Leste region in that file sits near -8.x,
125 to 127. The join is exact and `many_to_one`, so this is a defect in the
Kaggle file rather than in the notebook, but `figs/geo_funding_vs_poverty.png`
publishes it and Aileu is the top row of the priority table. A bounding-box
check after the merge would catch it.

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

The tests take seconds and need nothing. The notebook takes roughly 10 minutes
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
- [ ] Improve MPI join coverage beyond 7.6% (fuzzy/normalized region matching)
- [ ] Language detection for non-English `use` text before TF-IDF
- [ ] Drop loans posted within 60 days of the snapshot and restate the PR-AUC
- [ ] Fit imputation and TF-IDF inside a pipeline, on train only
- [ ] Temporal train/test split, reported alongside the random one
- [ ] Bounding-box check on the MPI coordinates after the merge

## License

MIT. See [`LICENSE`](../LICENSE).

## Credits

Author: **Alven Yuka**, CPA Finalist. Built on the [Kiva Loans / MPI dataset](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) (Kaggle).

