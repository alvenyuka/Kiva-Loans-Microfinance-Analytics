# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Every figure quoted
below is in `outputs/results.json`.

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
  tests, and `tests/test_readme_numbers.py`, which fails when the README quotes a
  number the results file does not hold.
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

[1.0.0]: https://github.com/alvenyuka/Kiva-Loans-Microfinance-Analytics/releases/tag/v1.0.0
