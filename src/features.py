"""Feature engineering and the leakage guard for the Kiva funding-risk model.

This module holds the logic the notebook's conclusions actually rest on, so that
it can be tested rather than asserted. Three things live here:

1. **The leakage guard.** `funded_time`, `disbursed_time`, `lender_count` and
   `funded_amount` are consequences of a loan being funded. They do not exist at
   the moment a loan is posted, so a model that uses them to predict "will this
   loan be funded" is reading the answer off the back of the card. A leaked
   column does not break a model; it improves the score, so nothing downstream
   would flag it. `assert_no_leakage` is therefore executed, not described: it
   turns the rule into a check that fails the build.

2. **Borrower gender parsing.** `borrower_genders` is a comma-separated list with
   one entry per borrower, because Kiva loans are often group loans. Parsing it
   is small, fiddly, and easy to get subtly wrong on the edge cases (missing
   values, unexpected labels, stray whitespace), which is exactly the profile of
   code that deserves tests.

3. **The funding target itself.** A loan is fully funded when the amount raised
   reaches the amount asked for. One line, and every number in the project hangs
   off it.

Nothing here reads a file or needs the dataset, so the tests run in CI in
seconds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The leakage guard
# ---------------------------------------------------------------------------

# These are lists, not tuples, on purpose. They are used to select columns
# (`df[POSTING_TIME_FEATURES]`), and pandas treats a tuple as one compound key,
# so a tuple raises KeyError with the entire tuple as the key. The failure
# surfaces far from the cause and reads like missing data rather than a type
# mistake.

#: Columns that only exist because a loan was funded. Never model inputs.
LEAKY_COLUMNS: list[str] = [
    "funded_time",
    "disbursed_time",
    "lender_count",
    "funded_amount",
]

#: Everything known at the moment a loan is posted. These are the model inputs.
POSTING_TIME_FEATURES: list[str] = [
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

#: Categoricals that get one-hot encoded before modelling.
CATEGORICAL_COLUMNS: list[str] = [
    "sector",
    "activity",
    "country",
    "repayment_interval",
]


class LeakageError(AssertionError):
    """Raised when a post-outcome column reaches the model's feature set."""


def assert_no_leakage(feature_names) -> None:
    """Fail if any post-outcome column has reached the feature set.

    Called before the model is fitted. A leaked column does not announce itself:
    it raises the score, which looks like success, so nothing downstream will
    catch it. This is the only place it can be caught cheaply.

    Raises:
        LeakageError: naming every offending column.
    """
    names = list(feature_names)
    leaked = sorted({c for c in names if c in set(LEAKY_COLUMNS)})

    # One-hot encoding and text features append suffixes, so an exact-match check
    # alone would miss `funded_amount_bucket` or `lender_count_log`.
    prefixed = sorted(
        {
            c
            for c in names
            if c not in leaked and any(c.startswith(f"{leak}_") for leak in LEAKY_COLUMNS)
        }
    )

    if leaked or prefixed:
        raise LeakageError(
            "Post-outcome columns reached the model's feature set. These are "
            "consequences of funding, not inputs to it, so any score computed "
            "with them is meaningless.\n"
            f"  exact matches:    {leaked}\n"
            f"  derived columns:  {prefixed}"
        )


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------

def mark_fully_funded(df: pd.DataFrame) -> pd.Series:
    """1 when a loan raised at least what it asked for, else 0.

    Uses >= rather than ==: Kiva loans occasionally close slightly over the
    requested amount, and those are funded, not unfunded.
    """
    return (df["funded_amount"] >= df["loan_amount"]).astype("int8")


# ---------------------------------------------------------------------------
# Borrower gender composition
# ---------------------------------------------------------------------------

def parse_gender_counts(genders_str) -> dict[str, int]:
    """Count male and female borrowers in one loan's `borrower_genders` cell.

    The field is a comma-separated list with one entry per borrower, so a group
    loan reads like "female, female, male". Missing values and unrecognised
    labels count as neither rather than being guessed at, which keeps an
    unexpected label out of the female share instead of silently skewing it.
    """
    if genders_str is None or pd.isna(genders_str):
        return {"n_male": 0, "n_female": 0}

    parts = [p.strip().lower() for p in str(genders_str).split(",")]
    return {
        "n_male": sum(1 for p in parts if p == "male"),
        "n_female": sum(1 for p in parts if p == "female"),
    }


def add_borrower_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add n_male, n_female, n_borrowers and pct_female. Returns a copy.

    `pct_female` is NaN, not 0, when no borrower could be parsed. Zero would
    claim the loan had only male borrowers, which is a different statement from
    "we could not tell", and the difference matters because the model reads this
    column.
    """
    out = df.copy()
    # Vectorised form of parse_gender_counts: split on commas, strip and lower-case
    # each entry, then count exact matches per loan. The same rules as the scalar
    # function (tested against it), without a Python call per row over 671k loans.
    genders = pd.Series(out["borrower_genders"].to_numpy(), dtype="string")
    tokens = genders.str.lower().str.split(",").explode().str.strip()
    for label in ("male", "female"):
        hits = tokens.eq(label).fillna(False).astype("int16")
        out[f"n_{label}"] = hits.groupby(level=0).sum().reindex(range(len(out)), fill_value=0).to_numpy().astype("int16")
    out["n_borrowers"] = (out["n_male"] + out["n_female"]).astype("int16")
    out["pct_female"] = np.where(
        out["n_borrowers"] > 0,
        out["n_female"] / out["n_borrowers"].replace(0, np.nan),
        np.nan,
    )
    return out
