# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Every figure quoted
below is in, or worked out from, `outputs/results.json`.

## [1.1.0] - 2026-10-25

The notebook becomes a guided build a newcomer can follow cell by cell. No
result changes: every value in `outputs/results.json` equals 1.0.0 except the
run record (`provenance`), and the test PR-AUC is still 0.374 against a 4.6%
base rate.

### Changed

- `Kiva_Loans_Microfinance_Analytics.ipynb` is now written by
  `build_notebook.py`. It has three parts, Prepare Data (1A to 1F), Build Model
  (2A to 2D) and Communicate Results (3A to 3C), of 122 numbered steps. Each step
  has one instruction, a one-line Concept, one small code cell with a
  plain-words comment above every top-level statement, printed output, and a
  "Check your work" cell whose expected values come from the executed run.
- The notebook opens with the question, the data, who acts on the ranking and
  the headline, and closes with what was built, the answers, limitations, next
  steps and a glossary.
- New steps: the majority-class trap shown on a constant score (Step 2.2), the
  review list's recall and precision for each borrower group (Step 2.21), and
  each review list's precision next to its recall in the shortfall table (Step
  3.11). `find_misplaced` and the text-flag helper are tried on made-up inputs
  before they run on the data.
- README and `docs/METHODOLOGY.md` point at the new section and step labels and
  name the builder as the source; the README states the new run.

## [1.0.0] - 2026-10-22

The first tagged version: the state of the project after the work from July to
October 2026.

### Added

- Funding-risk model: four candidates (logistic regression, random forest,
  LightGBM, LightGBM with a time trend) chosen on a validation slice and scored
  once on the most recent 20% of settled loans. Test PR-AUC for the at-risk
  class is 0.374 against a 4.6% base rate.
- Funding-shortfall impact (`src/impact.py`): reviewing the riskiest 10% of test
  loans reaches 76% of the $5.8M shortfall.
- Walk-forward check over four later windows, a segment table ranked by ROC-AUC,
  a calibration test, SHAP drivers and a province priority list.
- Days-to-fund regression: MAE 7.04 days against 8.08 for the median guess.
- Poverty analysis on the exact region join and on Kiva's region link.
- Leakage guard, funding target and gender parsing in `src/features.py`, with
  tests, including one that lists the four post-funding columns (funded time,
  disbursed time, lender count, funded amount) the guard keeps out of the model.
- `tests/test_readme_numbers.py`: every number in the README's prose, tables and
  image descriptions is read in order and compared with an expected list. Each
  entry is formatted from `outputs/results.json` at the printed precision, found
  in the executed notebook's output, or listed as a fixed design or wording
  number with its reason, so a mistyped figure anywhere fails. The test also
  checks that no sentence quotes a recall figure without its precision, that the
  fairness gap stays in Limits, and that the run named in the README is the one
  in the results file.
- `outputs/results.json` with the run's git commit, package versions and run time.
- CI on Python 3.11 and 3.13, installing pinned test dependencies from
  `requirements-test.txt`.
- This changelog and a pull request template.

### Fixed

- Loans posted within 60 days of the end of the data are excluded, because
  their outcome was not yet known.
- PR-AUC is reported for the at-risk class; the default scored the funded class.
- The README's calibration and PR-AUC explanations, and the poverty-score
  definitions.
- The README says who acts on the ranking (a partner-support team or a
  funder's matching-fund desk working down the list) and quotes the precision
  next to every recall figure: the riskiest 10% of test loans hold 62% of the
  unfunded loans at 28.7% precision, and the shortfall table gains a precision
  column.
- The README's opening line names the partner-support team or impact funder as
  the user of the ranking (on Kiva, lenders are members of the public), and the
  leakage sentences name all four post-funding columns, disbursed time included.

[1.1.0]: https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/releases/tag/v1.1.0
[1.0.0]: https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/releases/tag/v1.0.0
