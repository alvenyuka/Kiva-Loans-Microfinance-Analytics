"""
Business impact of the funding-risk score: how much of the funding shortfall it points at.

A loan's shortfall is the part of its requested amount that lenders never funded
(loan_amount - funded_amount, never below zero), in US dollars as Kiva reports them.
If a platform reviewed only the loans the model ranks riskiest, what share of the total
shortfall, and of the loans that went unfunded, would those reviews cover?
"""
import numpy as np
import pandas as pd

REVIEW_SHARES = (0.05, 0.10, 0.20, 0.30)


def funding_shortfall(loan_amount, funded_amount):
    """Unfunded part of each loan, floored at zero (a few loans close slightly overfunded)."""
    shortfall = np.asarray(loan_amount, float) - np.asarray(funded_amount, float)
    return np.clip(shortfall, 0, None)


def shortfall_capture(risk, loan_amount, funded_amount, review_shares=REVIEW_SHARES):
    """One row per review share: review the riskiest loans first, report what they cover.

    risk is higher for loans more likely to go unfunded (for example 1 - predicted funding probability).
    """
    risk = np.asarray(risk, float)
    shortfall = funding_shortfall(loan_amount, funded_amount)
    is_unfunded = shortfall > 0
    n_loans = len(risk)
    total_shortfall = shortfall.sum()
    total_unfunded = is_unfunded.sum()

    # Loan positions from riskiest to least risky
    riskiest_first = np.argsort(-risk, kind="stable")

    rows = []
    for share in review_shares:
        n_reviewed = int(round(share * n_loans))

        # True for the loans the platform would review
        reviewed = np.zeros(n_loans, dtype=bool)
        reviewed[riskiest_first[:n_reviewed]] = True

        shortfall_covered = shortfall[reviewed].sum()
        unfunded_covered = (reviewed & is_unfunded).sum()

        # With nothing to cover the share is reported as 0, not a division by zero
        if total_shortfall == 0:
            shortfall_pct = 0.0
        else:
            shortfall_pct = float(shortfall_covered / total_shortfall)
        if total_unfunded == 0:
            unfunded_pct = 0.0
        else:
            unfunded_pct = float(unfunded_covered / total_unfunded)

        rows.append({
            "review_share": share,
            "loans_reviewed": n_reviewed,
            "shortfall_covered_usd": float(shortfall_covered),
            "shortfall_covered_pct": shortfall_pct,
            "unfunded_loans_covered": int(unfunded_covered),
            "unfunded_loans_covered_pct": unfunded_pct,
        })
    return pd.DataFrame(rows)
