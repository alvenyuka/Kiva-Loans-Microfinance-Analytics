# Kiva Loans Microfinance Analytics

> Funding-risk model on 671,205 real Kiva microloans: which loans are at risk of not being funded, and does that risk fall hardest on the poorest regions? LightGBM reaches 0.4889 PR-AUC on the at-risk minority class, explained with SHAP.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#tech-stack)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)](#tech-stack)
[![PR-AUC (at-risk)](https://img.shields.io/badge/PR--AUC%20(at--risk)-0.4889-success)](#results)

## Why?

**Which loans are at risk of not getting fully funded, and does that risk fall hardest on the poorest regions?** This notebook analyzes 671k+ real microloans from Kiva's public dataset (Kaggle's "Data Science for Good: Kiva Crowdfunding"). It runs exploratory analysis, text mining of loan-use descriptions, and a geospatial join against a region-level poverty index, then builds a funding-risk classification model explained with SHAP, a days-to-fund regression, and a final synthesis flagging regions that are both poverty-deep and funding-at-risk. All findings are correlational, not causal; this is a single-snapshot dataset.

## Project Structure

```
Kiva-Loans-Microfinance-Analytics/
├── Kiva_Loans_Microfinance_Analytics.ipynb   # the notebook, run end to end
├── build_notebook.py                         # generates the notebook, edit this not the .ipynb
├── figs/
│   ├── geo_funding_vs_poverty.png            # Section 5 geospatial/MPI figure
│   └── shap_summary.png                      # Section 7 SHAP summary plot
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

1. Download the two source CSVs (see Dataset below) and place them in `../Data/Kiva/` relative to this folder.
2. `pip install -r requirements.txt`
3. Run `build_notebook.py` then execute the generated notebook end to end (full commands under Running it below).
4. Section 7 (funding-risk model + SHAP) has the headline result; Section 9 has the region-priority synthesis.

## Features

- **Exploratory analysis** across loan amount, sector, country, and borrower gender composition (Section 3).
- **Text mining** of the free-text `use` field via TF-IDF, producing coherent per-sector vocabulary (Section 4).
- **Geospatial join** against Kiva's region-level Multidimensional Poverty Index, with honest coverage reporting (Section 5).
- **Leakage-checked feature engineering** -- outcome-dependent columns (`funded_time`, `disbursed_time`, `lender_count`, `funded_amount`) explicitly excluded because they aren't knowable at posting time (Section 6).
- **Funding-risk classification** (Logistic Regression, Random Forest, LightGBM compared), evaluated on minority-class PR-AUC -- the majority-class number looks good regardless of whether the model works, so it's minority-class PR-AUC that actually answers the question (Section 7).
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

Kaggle's "Data Science for Good: Kiva Crowdfunding" dataset -- 671,205 real microloans
(`kiva_loans.csv`), joined against Kiva's region-level Multidimensional Poverty Index
(`kiva_mpi_region_locations.csv`). Not included in this repo (see `.gitignore`) --
download from:
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
  against Kiva's region-to-MPI lookup table. Coverage is low -- only **7.6%** of loans
  matched (50,955 / 671,205) -- so every MPI-dependent result describes only that
  matched subset, not the full dataset (see Known Limitations).
- **Feature engineering with an explicit leakage check (Section 6):** `funded_time`,
  `disbursed_time`, `lender_count`, and `funded_amount` are excluded because they are
  consequences of a loan being funded, not knowable at posting time. The model uses
  44 posting-time features (including the MPI join and TF-IDF term-presence columns),
  one-hot encoded to 305 columns.
- **Funding-risk model + SHAP (Section 7):** Logistic Regression, Random Forest, and
  LightGBM are compared on `fully_funded` (92.8% funded / 7.2% not funded) using
  **both** majority-class and minority-class PR-AUC, since the notebook's actual
  question -- which loans are at risk of *not* being funded -- is about the minority
  class; LightGBM wins on both metrics and is explained with SHAP.
- **Days-to-fund regression (Section 8):** among loans that did get funded, a Random
  Forest regresses `days_to_fund = funded_time - posted_time` on the same posting-time
  feature set.
- **Synthesis (Section 9):** the funding-risk model's predicted funding probability is
  aggregated to region level and combined with each region's MPI into a
  `priority_score = MPI * (1 - mean predicted funding probability)`, ranking regions
  that are both poverty-deep and funding-at-risk.

## Results

- **Funding-risk model:** LightGBM is the best model. Headline metric --
  **PR-AUC (at-risk, minority class) = 0.4889** -- substantially outperforms Random
  Forest (0.3580) and Logistic Regression (0.2731) on this metric. (Majority-class
  PR-AUC is 0.9937 and ROC-AUC is 0.9241, but both sit close to the majority class's
  trivial-baseline floor and are reported for completeness, not as the headline --
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
  Leone/Bo -- i.e. Timor-Leste and Sierra Leone regions dominate the top of the
  list, combining high poverty depth with lower model-predicted funding
  probability. Given the 7.6% MPI coverage, this list should be read as
  illustrative of the method, not as a reliable region-targeting list for the
  92.4% of loan volume that couldn't be matched to an MPI score.

## Known Limitations

Four things worth flagging before trusting any result above too far.

Loans with no `funded_time` at the moment this dataset was captured are treated
as "not fully funded" in the model, but some may simply not have expired yet;
that's a modeling assumption, not a certainty. MPI join coverage sits at just
7.6% (50,955 / 671,205 loans) -- most loans couldn't be matched to a
region-level MPI score, most likely because `kiva_loans.csv`'s free-text
`region` field (entered inconsistently by field partners) doesn't standardize
against the MPI lookup table's `region` field well enough for an exact string
join. That means every MPI-dependent result in this notebook, the Section 5
map/correlation, the `MPI` feature in the Section 7 model, and the Section 9
priority-regions table, describes only that small, non-random 7.6% subset of
loans.

Nothing here establishes that any feature *causes* funding success or delay,
only that it's predictive or associated with it. And the text mining is
English-only: TF-IDF was fit on the raw `use` field without language
detection, so non-English descriptions contribute noise to the term list.

## Running it

Requires: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`,
`shap`, `plotly` (with `kaleido` for static map export), and `jupyter`/`nbconvert`.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm shap plotly kaleido jupyter nbconvert
python build_notebook.py
jupyter nbconvert --to notebook --execute Kiva_Loans_Microfinance_Analytics.ipynb --output Kiva_Loans_Microfinance_Analytics.ipynb
```

The two source CSVs (`kiva_loans.csv` and `kiva_mpi_region_locations.csv`, see Dataset
above) must be placed in `../Data/Kiva/` relative to this folder before running.

## Roadmap

- [x] EDA, text mining, geospatial/MPI join
- [x] Leakage-checked funding-risk model (LightGBM, SHAP)
- [x] Days-to-fund regression
- [x] Region-priority synthesis
- [ ] Improve MPI join coverage beyond 7.6% (fuzzy/normalized region matching)
- [ ] Language detection for non-English `use` text before TF-IDF

## License

MIT. See [`LICENSE`](LICENSE).

## Credits

Author: **Alven Yuka**, CPA Finalist (Kenya). Built on the [Kiva Loans / MPI dataset](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) (Kaggle).

## Connect

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · 🐙 [GitHub](https://github.com/alvenyuka)
