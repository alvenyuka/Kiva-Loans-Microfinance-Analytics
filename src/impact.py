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


def funding_shortfall(loan_amount, funded_amount) -> np.ndarray:
    """Unfunded part of each loan, floored at zero (a few loans close slightly overfunded)."""
    return np.clip(np.asarray(loan_amount, float) - np.asarray(funded_amount, float), 0, None)


def shortfall_capture(risk, loan_amount, funded_amount, review_shares=REVIEW_SHARES) -> pd.DataFrame:
    """One row per review share: review the riskiest loans first, report what they cover.

    risk  higher = more likely to go unfunded (for example 1 - predicted funding probability)
    """
    risk = np.asarray(risk, float)
    shortfall = funding_shortfall(loan_amount, funded_amount)
    unfunded = shortfall > 0
    order = np.argsort(-risk, kind="stable")
    n, total = len(risk), shortfall.sum()
    rows = []
    for share in review_shares:
        k = int(round(share * n))
        reviewed = np.zeros(n, bool)
        reviewed[order[:k]] = True
        rows.append({
            "review_share": share,
            "loans_reviewed": k,
            "shortfall_covered_usd": float(shortfall[reviewed].sum()),
            "shortfall_covered_pct": float(shortfall[reviewed].sum() / total) if total else 0.0,
            "unfunded_loans_covered": int((reviewed & unfunded).sum()),
            "unfunded_loans_covered_pct": float((reviewed & unfunded).sum() / unfunded.sum()) if unfunded.any() else 0.0,
        })
    return pd.DataFrame(rows)
