"""Tests for the leakage guard, the funding target, and gender parsing.

The leakage tests are the ones that matter. A leaked column does not produce an
error or an obviously wrong number; it produces a *better* score, which reads as
success and passes every downstream check. This file is the only place that
failure mode is visible, which is why it is tested first and hardest.
"""
import numpy as np
import pandas as pd
import pytest

from features import (
    CATEGORICAL_COLUMNS,
    LEAKY_COLUMNS,
    POSTING_TIME_FEATURES,
    LeakageError,
    add_borrower_features,
    assert_no_leakage,
    mark_fully_funded,
    parse_gender_counts,
)


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def test_posting_time_features_and_leaky_columns_never_overlap():
    """The two lists are the whole design. If a name appears in both, the guard
    is meaningless no matter what the runtime check does."""
    assert set(POSTING_TIME_FEATURES).isdisjoint(set(LEAKY_COLUMNS))


def test_clean_feature_set_passes():
    assert_no_leakage(list(POSTING_TIME_FEATURES))


def test_the_four_known_leaky_columns_are_all_listed():
    """Pin the membership of LEAKY_COLUMNS by name.

    This looks redundant next to the parametrised test below, and it is not. That
    test draws its cases from LEAKY_COLUMNS itself, so deleting an entry deletes
    the test case along with it and the suite still goes green while the guard
    has stopped watching that column. Verified by deleting `funded_time` and
    watching the parametrised test quietly stop covering it.

    Each of these four exists only because a loan was funded:
      funded_time      when it finished funding
      disbursed_time   when the money went out
      lender_count     how many people chipped in
      funded_amount    how much was raised, which is the target in disguise
    """
    assert set(LEAKY_COLUMNS) == {
        "funded_time",
        "disbursed_time",
        "lender_count",
        "funded_amount",
    }


@pytest.mark.parametrize("leak", LEAKY_COLUMNS)
def test_each_leaky_column_is_caught(leak):
    """Every post-outcome column, one at a time. Parametrised rather than
    checked in a single list so a newly added leaky column cannot pass by
    hiding behind one that is already caught."""
    with pytest.raises(LeakageError) as exc:
        assert_no_leakage(list(POSTING_TIME_FEATURES) + [leak])
    assert leak in str(exc.value)


def test_derived_leaky_column_is_caught():
    """One-hot encoding and binning append suffixes, so an exact-name check
    alone would wave `lender_count_log` straight through."""
    with pytest.raises(LeakageError) as exc:
        assert_no_leakage(["loan_amount", "lender_count_log", "funded_amount_bucket"])
    msg = str(exc.value)
    assert "lender_count_log" in msg
    assert "funded_amount_bucket" in msg


def test_error_names_every_offender_not_just_the_first():
    """A guard that reports one problem at a time turns one fix into several
    round trips."""
    with pytest.raises(LeakageError) as exc:
        assert_no_leakage(["loan_amount", "funded_time", "lender_count"])
    assert "funded_time" in str(exc.value)
    assert "lender_count" in str(exc.value)


def test_guard_accepts_any_iterable():
    """It is called with a DataFrame's .columns as often as with a list."""
    df = pd.DataFrame(columns=["loan_amount", "term_in_months"])
    assert_no_leakage(df.columns)


def test_categoricals_are_all_posting_time_features():
    """Every column that gets one-hot encoded has to be known at posting time,
    or the encoding quietly introduces leakage the guard would then have to
    catch by suffix."""
    assert set(CATEGORICAL_COLUMNS).issubset(set(POSTING_TIME_FEATURES))


def test_constants_can_select_dataframe_columns():
    """Regression test. These constants were tuples once, and pandas reads a
    tuple as a single compound key, so `df[POSTING_TIME_FEATURES]` raised
    KeyError with the whole 14-name tuple as the key instead of selecting 14
    columns. The message looked like missing data rather than a type mistake,
    and it only surfaced part-way through a full notebook run."""
    for const in (POSTING_TIME_FEATURES, LEAKY_COLUMNS, CATEGORICAL_COLUMNS):
        frame = pd.DataFrame({name: [0] for name in const})
        assert list(frame[const].columns) == list(const)


def test_constants_concatenate_with_a_plain_list():
    """`POSTING_TIME_FEATURES + ["fully_funded"]` is how the model matrix is
    built. A tuple raises TypeError on that line instead."""
    combined = POSTING_TIME_FEATURES + ["fully_funded"]
    assert combined[-1] == "fully_funded"
    assert len(combined) == len(POSTING_TIME_FEATURES) + 1


def test_get_dummies_accepts_the_categorical_constant():
    """pd.get_dummies(columns=...) is the other call site that needs a list."""
    frame = pd.DataFrame({c: ["a", "b"] for c in CATEGORICAL_COLUMNS})
    encoded = pd.get_dummies(frame, columns=CATEGORICAL_COLUMNS, drop_first=True)
    assert all(c not in encoded.columns for c in CATEGORICAL_COLUMNS)


# ---------------------------------------------------------------------------
# The funding target
# ---------------------------------------------------------------------------

def test_fully_funded_is_one_when_target_is_met_exactly():
    df = pd.DataFrame({"funded_amount": [500.0], "loan_amount": [500.0]})
    assert mark_fully_funded(df).iloc[0] == 1


def test_fully_funded_is_one_when_overfunded():
    """Kiva loans occasionally close slightly above the requested amount. An
    equality test would label those unfunded, which is backwards."""
    df = pd.DataFrame({"funded_amount": [525.0], "loan_amount": [500.0]})
    assert mark_fully_funded(df).iloc[0] == 1


def test_fully_funded_is_zero_when_short():
    df = pd.DataFrame({"funded_amount": [499.99], "loan_amount": [500.0]})
    assert mark_fully_funded(df).iloc[0] == 0


def test_fully_funded_returns_a_compact_integer_type():
    df = pd.DataFrame({"funded_amount": [500.0, 100.0], "loan_amount": [500.0, 500.0]})
    result = mark_fully_funded(df)
    assert result.dtype == np.int8
    assert result.tolist() == [1, 0]


# ---------------------------------------------------------------------------
# Borrower gender parsing
# ---------------------------------------------------------------------------

def test_single_borrower():
    assert parse_gender_counts("female") == {"n_male": 0, "n_female": 1}


def test_group_loan_counts_every_borrower():
    """Group loans are the normal case on Kiva, not an edge case. Counting the
    row once instead of once per borrower would understate group lending
    across the whole dataset."""
    assert parse_gender_counts("female, female, male, female") == {"n_male": 1, "n_female": 3}


def test_whitespace_and_case_do_not_change_the_count():
    assert parse_gender_counts("  Female ,MALE,  male  ") == {"n_male": 2, "n_female": 1}


def test_missing_value_counts_as_neither():
    assert parse_gender_counts(np.nan) == {"n_male": 0, "n_female": 0}
    assert parse_gender_counts(None) == {"n_male": 0, "n_female": 0}


def test_unrecognised_label_is_not_guessed_at():
    """An unexpected label must not land in either count. Folding it into one
    would skew the female share, which is a headline figure in the analysis."""
    assert parse_gender_counts("female, nonbinary, male") == {"n_male": 1, "n_female": 1}


def test_add_borrower_features_computes_share_and_totals():
    df = pd.DataFrame({"borrower_genders": ["female, female, male", "male", "female"]})
    out = add_borrower_features(df)
    assert out["n_borrowers"].tolist() == [3, 1, 1]
    assert out["pct_female"].tolist() == pytest.approx([2 / 3, 0.0, 1.0])


def test_unparseable_row_gets_nan_share_not_zero():
    """NaN says "we could not tell". Zero says "all borrowers were male". The
    model reads this column, so the difference is not cosmetic."""
    df = pd.DataFrame({"borrower_genders": [np.nan, "female"]})
    out = add_borrower_features(df)
    assert np.isnan(out["pct_female"].iloc[0])
    assert out["pct_female"].iloc[1] == pytest.approx(1.0)
    assert out["n_borrowers"].iloc[0] == 0


def test_add_borrower_features_does_not_mutate_the_caller():
    df = pd.DataFrame({"borrower_genders": ["female, male"]})
    before = df.copy()
    add_borrower_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_vectorised_counts_match_the_scalar_parser():
    """add_borrower_features counts in bulk; parse_gender_counts is the rule.
    The two must agree on every edge case, including labels that merely contain
    the word "male"."""
    cells = ["female", "female, female, male", "  Female ,MALE,  male  ", np.nan, None,
             "female, nonbinary, male", "", "male-ish, female", ",,male"]
    out = add_borrower_features(pd.DataFrame({"borrower_genders": cells}, index=[10, 3, 3, 7, 8, 1, 2, 5, 4]))
    for i, cell in enumerate(cells):
        expected = parse_gender_counts(cell)
        assert out["n_male"].iloc[i] == expected["n_male"], cell
        assert out["n_female"].iloc[i] == expected["n_female"], cell
