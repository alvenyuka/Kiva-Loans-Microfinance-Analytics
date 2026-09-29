# Kiva Loans Microfinance Analytics

> Which microloans are at risk of never being funded, and does that risk fall hardest on the poorest regions? A funding-risk model on 671,205 real Kiva loans that catches 91% of at-risk loans at posting time (PR-AUC 0.4889 on the at-risk class).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#how-it-works)
[![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat)](#how-it-works)
[![tests](https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/actions/workflows/ci.yml)

## The problem

On Kiva, microfinance field partners post loans for small borrowers and lenders fund them in small amounts.
About 7% of loans never reach full funding. For a borrower that can mean a missed planting season or lost
stock; for a field partner it means a loan they may have to cover or cancel. If the loans at risk could be
spotted **when they are posted**, a platform or impact funder could step in early, for example with
featuring or matching funds.

The development-finance question sits on top of that: are the poorest regions the ones left unfunded?

## What I found

| Result | Value |
|---|---:|
| At-risk loans the model catches (recall) | **91%** |
| Flagged loans that are truly at risk (precision) | 25% |
| PR-AUC on the at-risk class (LightGBM; Random Forest 0.358, logistic regression 0.273) | **0.4889** |
| Days-to-fund prediction error, funded loans (MAE) | 7.43 days |

- **How a loan is structured matters most.** The strongest predictors of funding risk are the loan term, the
  amount requested and the month it is posted, ranked ahead of borrower gender mix, sector and country.

  ![Top features by mean absolute SHAP value: loan term, loan amount and posting month lead](figs/shap_summary.png)

- **Poverty data covers only 7.6% of loans** (50,955 of 671,205), because region names do not match the
  poverty index cleanly. Within that subset, regions in Timor-Leste and Sierra Leone combine the deepest
  poverty with the lowest predicted funding, but the list illustrates the method rather than a targeting
  list.
- **The model is a screening tool, not a verdict.** It flags about four loans for every one truly at risk,
  which suits early, low-cost interventions such as featuring a loan, not decisions that exclude borrowers.

**What I would recommend to a platform or impact funder:** use the score to queue loans for early support at
posting time, review how loan term and size are set with field partners, and invest in cleaning region data
before using poverty to target funds.

## How it works

1. **Data.** 671,205 loans from Kaggle's Kiva dataset, joined to Kiva's regional Multidimensional Poverty
   Index.
2. **Posting-time features only.** Columns that exist only because a loan was funded (funded time, lender
   count, funded amount) are excluded, and a tested leakage guard enforces it.
3. **Text features** from each loan's free-text purpose, using TF-IDF.
4. **Three models compared** on the at-risk class, because the funded majority looks good whatever the
   model does. LightGBM wins and SHAP explains its drivers.
5. **Regional synthesis**: a priority score combining poverty depth with predicted funding risk.

## Run it

```bash
pip install -r requirements.txt
python -m pytest          # tests, no dataset needed, a few seconds
# put kiva_loans.csv and kiva_mpi_region_locations.csv in ./data (or set KIVA_DATA_DIR), then:
python build_notebook.py
jupyter nbconvert --to notebook --execute Kiva_Loans_Microfinance_Analytics.ipynb --output Kiva_Loans_Microfinance_Analytics.ipynb
```

## Limitations

- **Some "not funded" labels are loans still fundraising** when the dataset snapshot was taken, which
  inflates the at-risk class. Dropping recently posted loans and restating the score is the next fix.
- **Random, not time-based, split**, and some preprocessing was fitted before the split; a stricter setup
  would report a lower score.
- **Correlational only.** Nothing here shows that any factor causes funding success, and the poverty link is
  measured across regions, not individual borrowers.

## More detail

The full write-up, with every result, all limitations and the tests, is in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

## License

MIT. See [`LICENSE`](LICENSE). Data: [Data Science for Good: Kiva Crowdfunding](https://www.kaggle.com/datasets/kiva/data-science-for-good-kiva-crowdfunding) (Kaggle).

## Connect

Built by Alven Yuka, CPA Finalist and Accounting Specialist at GIZ, Nairobi.

📫 [alvenyuka2@gmail.com](mailto:alvenyuka2@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/alven-yuka-610b78174/) · 🐙 [GitHub](https://github.com/alvenyuka)
