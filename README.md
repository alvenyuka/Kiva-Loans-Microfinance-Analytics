# Kiva Loans Microfinance Analytics

**Which loans are at risk of not getting fully funded -- and does that risk fall hardest
on the poorest regions?** This notebook analyzes 671k+ real microloans from Kiva's public
dataset (Kaggle's "Data Science for Good: Kiva Crowdfunding"), combining exploratory
analysis, text mining of loan-use descriptions, a geospatial join against a
region-level poverty index, a funding-risk classification model explained with SHAP,
a days-to-fund regression, and a synthesis identifying regions that are both
poverty-deep and funding-at-risk. All findings are correlational, not causal --
this is a single-snapshot dataset.

## Data

Kaggle's "Data Science for Good: Kiva Crowdfunding" dataset -- 671,205 real microloans
(`kiva_loans.csv`), joined against Kiva's region-level Multidimensional Poverty Index
(`kiva_mpi_region_locations.csv`). Not included in this repo (see `.gitignore`) --
download from:
https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding

## Method

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
  matched subset, not the full dataset (see Limitations).
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
  the minority-class number is the one that actually reflects the "at risk of not
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

## Limitations

- **Single data snapshot.** Loans with no `funded_time` at the moment this dataset
  was captured are treated as "not fully funded" -- some may simply not have expired
  yet. This is a modeling assumption, not a certainty.
- **MPI join coverage is only 7.6% (50,955 / 671,205 loans).** Most loans could not be
  matched to a region-level MPI score, most likely because `kiva_loans.csv`'s
  free-text `region` field (entered inconsistently by field partners) doesn't
  standardize against the MPI lookup table's `region` field well enough for an exact
  string join. Every MPI-dependent result -- the Section 5 map/correlation, the `MPI`
  feature in the Section 7 model, and the Section 9 priority-regions table -- describes
  only that small, non-random 7.6% subset of loans, not the full dataset.
- **Correlational, not causal.** Nothing here establishes that any feature *causes*
  funding success or delay -- only that it's predictive/associated.
- **English-only text mining.** TF-IDF was fit on the raw `use` field without language
  detection; non-English descriptions contribute noise to the term list.

## Running it

Requires: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`,
`shap`, `plotly` (with `kaleido` for static map export), and `jupyter`/`nbconvert`.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm shap plotly kaleido jupyter nbconvert
python build_notebook.py
jupyter nbconvert --to notebook --execute Kiva_Loans_Microfinance_Analytics.ipynb --output Kiva_Loans_Microfinance_Analytics.ipynb
```

The two source CSVs (`kiva_loans.csv` and `kiva_mpi_region_locations.csv`, see Data
above) must be placed in `../Data/Kiva/` relative to this folder before running.

## License

MIT — see [`LICENSE`](LICENSE).

## Credits

Author: **Alven Yuka** — CPA Finalist (Kenya). Built on the [Kiva Loans / MPI dataset](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) (Kaggle).

## Connect

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · 🐙 [GitHub](https://github.com/alvenyuka)
