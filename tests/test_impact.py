"""The shortfall arithmetic behind the README's business-impact table."""
import numpy as np
import pytest

from impact import funding_shortfall, shortfall_capture


def test_shortfall_is_the_unfunded_part_and_never_negative():
    s = funding_shortfall([1000, 500, 300], [400, 500, 325])
    assert list(s) == [600, 0, 0]  # the overfunded loan counts as no shortfall


def test_reviewing_the_riskiest_loans_first():
    risk = np.array([0.9, 0.8, 0.1, 0.05])
    loan = np.array([1000, 400, 500, 500])
    funded = np.array([0, 400, 300, 500])  # shortfalls: 1000, 0, 200, 0
    row = shortfall_capture(risk, loan, funded, review_shares=(0.5,)).iloc[0]
    assert row["loans_reviewed"] == 2
    assert row["shortfall_covered_usd"] == 1000
    assert row["shortfall_covered_pct"] == pytest.approx(1000 / 1200)
    assert row["unfunded_loans_covered"] == 1
    assert row["unfunded_loans_covered_pct"] == pytest.approx(0.5)


def test_reviewing_everything_covers_everything():
    risk = np.array([0.2, 0.4])
    row = shortfall_capture(risk, [100, 100], [0, 50], review_shares=(1.0,)).iloc[0]
    assert row["shortfall_covered_pct"] == pytest.approx(1.0)
