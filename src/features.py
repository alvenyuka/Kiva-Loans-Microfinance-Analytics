"""Feature engineering and the leakage guard for the Kiva funding-risk model.

Three things live here so that the tests can check them:

1. The leakage guard. `funded_time`, `disbursed_time`, `lender_count` and
   `funded_amount` only exist because a loan was funded. A model that uses them
   to predict "will this loan be funded" is reading the answer. A leaked column
   makes the score better, so nothing downstream would complain. That is why
   `assert_no_leakage` is run, not only described.
2. Borrower gender parsing. `borrower_genders` is a comma-separated list with one
   entry per borrower, because many Kiva loans are group loans.
3. The funding target: a loan is fully funded when the amount raised reaches the
   amount asked for.

Nothing here reads a file, so the tests run without the dataset.
"""
import numpy as np
import pandas as pd

# These must be lists, not tuples. pandas reads a tuple as one compound key, so
# df[some_tuple] raises a KeyError that looks like missing data.

# Columns that only exist because a loan was funded. Never model inputs.
LEAKY_COLUMNS = [
    "funded_time",
    "disbursed_time",
    "lender_count",
    "funded_amount",
]

# Everything known at the moment a loan is posted. These are the model inputs.
POSTING_TIME_FEATURES = [
    "loan_amount",
    "sector",
    "activity",
    "country",
    "term_in_months",
    "repayment_interval",
    "n_male",
    "n_female",
    "n_borrowers",
    "pct_female",
    "post_month",
    "post_dow",
    "MPI",
    "use_len",
]

# Text columns that get one-hot encoded before modelling.
CATEGORICAL_COLUMNS = [
    "sector",
    "activity",
    "country",
    "repayment_interval",
]


class LeakageError(AssertionError):
    """Raised when a post-outcome column reaches the model's feature set."""


def assert_no_leakage(feature_names):
    """Fail if any post-outcome column is in the feature set, and name every one."""
    exact_matches = []
    derived_columns = []

    for name in feature_names:
        if name in LEAKY_COLUMNS:
            if name not in exact_matches:
                exact_matches.append(name)
            continue

        # One-hot encoding and text features add a suffix, so an exact-name check
        # alone would miss "funded_amount_bucket" or "lender_count_log".
        for leaky in LEAKY_COLUMNS:
            if name.startswith(leaky + "_"):
                if name not in derived_columns:
                    derived_columns.append(name)

    exact_matches.sort()
    derived_columns.sort()

    if exact_matches or derived_columns:
        raise LeakageError(
            "Post-outcome columns reached the model's feature set. These are "
            "consequences of funding, not inputs to it, so any score computed "
            "with them is meaningless.\n"
            f"  exact matches:    {exact_matches}\n"
            f"  derived columns:  {derived_columns}"
        )


def mark_fully_funded(df):
    """1 when a loan raised at least what it asked for, else 0.

    Uses >= rather than ==: a few Kiva loans close slightly over the requested
    amount, and those are funded.
    """
    return (df["funded_amount"] >= df["loan_amount"]).astype("int8")


def parse_gender_counts(genders_str):
    """Count male and female borrowers in one loan's `borrower_genders` cell.

    A group loan reads like "female, female, male". A missing value or an
    unrecognised label counts as neither, so it cannot skew the female share.
    """
    if genders_str is None or pd.isna(genders_str):
        return {"n_male": 0, "n_female": 0}

    parts = str(genders_str).split(",")
    cleaned = []
    for part in parts:
        cleaned.append(part.strip().lower())

    return {"n_male": cleaned.count("male"), "n_female": cleaned.count("female")}


def add_borrower_features(df):
    """Add n_male, n_female, n_borrowers and pct_female. Returns a copy.

    pct_female is NaN, not 0, when no borrower could be parsed. 0 would claim the
    loan had only male borrowers, and the model reads this column.
    """
    out = df.copy()

    n_male = []
    n_female = []
    for cell in out["borrower_genders"]:
        counts = parse_gender_counts(cell)
        n_male.append(counts["n_male"])
        n_female.append(counts["n_female"])

    # Plain arrays, not Series: the index of df may have repeats, and a Series would try to line up on it
    out["n_male"] = np.array(n_male, dtype="int16")
    out["n_female"] = np.array(n_female, dtype="int16")
    out["n_borrowers"] = (out["n_male"] + out["n_female"]).astype("int16")

    # Dividing by NaN instead of 0 gives NaN for loans with no parsed borrower
    borrowers_or_nan = out["n_borrowers"].replace(0, np.nan)
    out["pct_female"] = out["n_female"] / borrowers_or_nan
    return out
