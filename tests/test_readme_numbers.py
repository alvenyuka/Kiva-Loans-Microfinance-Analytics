"""Every number in the README, in order, checked against where it comes from.

The README is typed by hand. This file reads every number in its prose, tables
and image descriptions, one occurrence at a time, in the order they appear, and
compares that sequence with an expected list. Each expected number is one of
three kinds:

- a result, formatted from outputs/results.json (which the notebook writes) at
  the precision the README prints it (0.374, not 0.3742193424722034), or worked
  out from two of its values (the precision of a review list is the unfunded
  loans it reaches divided by the loans reviewed);
- a figure the executed notebook prints, checked against the notebook's saved
  output;
- a fixed number that is part of the design or the wording (the 60-day rule, a
  section or step number), listed with the reason it is fixed.

Code blocks, inline code and link targets are skipped. A re-run that moves a
result, or an edit that mistypes any number, changes the sequence and fails
here. Adding or rewording a number in the README means adding it here too,
with its source.

No data is needed: the results file and the executed notebook are committed, so
these tests run in CI.
"""
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
NOTEBOOK_FILE = ROOT / "Kiva_Loans_Microfinance_Analytics.ipynb"

with open(ROOT / "outputs" / "results.json", encoding="utf-8") as results_file:
    RESULTS = json.load(results_file)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def notebook_printed_text():
    # Join every text output the executed notebook saved, so a figure the
    # README takes from the notebook can be found in what the notebook printed
    notebook = json.loads(NOTEBOOK_FILE.read_text(encoding="utf-8"))
    pieces = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            text = output.get("text") or output.get("data", {}).get("text/plain", "")
            pieces.append("".join(text))
    return "\n".join(pieces)


NOTEBOOK_TEXT = notebook_printed_text()


# ---------------------------------------------------------------------------
# Reading the numbers out of the README
# ---------------------------------------------------------------------------

# "26 Jul 2017" is read as one token, so a date cannot pass as two loose numbers
DATE = re.compile(r"\d{1,2} (?:%s) \d{4}" % "|".join(MONTHS))
NUMBER = re.compile(r"(?<![\w.,/-])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)|(?<=/)\d+(?!\w)")


def readme_prose(source=README):
    # Drop what is not prose: fenced code, inline code, link targets, HTML tags
    # and the "1." markers of the numbered build steps
    text = re.sub(r"```.*?```", "", source, flags=re.S)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(?m)^\s*\d+\.\s", "", text)
    return text


def readme_numbers(source=README):
    # Every number in reading order, with its line for the error message
    found = []
    for line_no, line in enumerate(readme_prose(source).splitlines(), start=1):
        position = 0
        while position < len(line):
            date_match = DATE.match(line, position)
            if date_match and (position == 0 or not line[position - 1].isalnum()):
                found.append((date_match.group(0), line_no, line))
                position = date_match.end()
                continue
            number = NUMBER.match(line, position)
            if number:
                found.append((number.group(0), line_no, line))
                position = number.end()
                continue
            position += 1
    return found


# ---------------------------------------------------------------------------
# The expected sequence, one entry per occurrence
# ---------------------------------------------------------------------------

NOTEBOOK_FIGURES = []   # (README text, text the notebook prints)


def result(text):
    # A figure already formatted from outputs/results.json
    return text


def from_notebook(text, printed):
    # A figure the executed notebook prints
    NOTEBOOK_FIGURES.append((text, printed))
    return text


def fixed(text, why):
    # A number that is part of the design or the wording, not a result
    assert why
    return text


def pct(x, places=0):
    # A share written as a percentage, without the % sign
    return f"{x * 100:.{places}f}"


def money(x):
    # A dollar amount with thousands separators, without the $ sign
    return f"{x:,.0f}"


def count(x):
    # A count with thousands separators
    return f"{x:,}"


def day_month_year(iso_text):
    # "2017-07-26" written the README's way: "26 Jul 2017"
    year, month, day = (int(part) for part in iso_text.split("-"))
    written = date(year, month, day)
    return f"{written.day} {MONTHS[written.month - 1]} {written.year}"


def capture_row(review_share):
    # The row of the shortfall table for one review share
    for row in RESULTS["shortfall"]["capture"]:
        if abs(row["review_share"] - review_share) < 1e-9:
            return row
    raise AssertionError(f"no capture row for review share {review_share}")


def list_precision(row):
    # Precision of a review list: the unfunded loans it reaches over the loans reviewed
    return row["unfunded_loans_covered"] / row["loans_reviewed"]


def segment(column, value):
    # One row of a segment table, by its label
    for row in RESULTS["segments"][column]:
        if row[column] == value:
            return row
    raise AssertionError(f"no segment {value!r} in {column}")


def expected_numbers():
    data = RESULTS["data"]
    split = RESULTS["split"]
    clf = RESULTS["classification"]
    candidates = clf["candidates"]
    shortfall = RESULTS["shortfall"]
    calibration = RESULTS["calibration"]
    walk = RESULTS["walk_forward"]
    linked = RESULTS["regional_poverty_linked"]
    challenger = RESULTS["linked_mpi_challenger"]
    days = RESULTS["days_to_fund"]
    terms = list(RESULTS["outcome_eda"]["not_funded_by_term_band"].values())
    provenance = RESULTS["provenance"]
    test_share = split["not_funded_share"]["test"]
    validation_share = split["not_funded_share"]["validation"]
    c5, c10, c20, c30 = capture_row(0.05), capture_row(0.1), capture_row(0.2), capture_row(0.3)
    first_week = data["funded_rate_by_age_days"]["0-7"]
    oldest = data["funded_rate_by_age_days"][">90"]
    baseline = from_notebook("0.0962", "Baseline validation PR-AUC (at-risk): 0.0962")
    chosen = candidates["LightGBM + time trend"]
    plain = candidates["LightGBM"]
    forest = candidates["Random Forest"]
    logistic = candidates["Logistic Regression"]
    precision_at = clf["at_risk_precision_at_recall"]
    male = segment("borrower_group", "all male")
    female = segment("borrower_group", "all female")

    seq = []
    # Hero image description: share of the shortfall reached at 5%, 10% and 20% of loans
    seq += [result(pct(c5["shortfall_covered_pct"])), result(pct(c5["review_share"])),
            result(pct(c10["shortfall_covered_pct"])), result(pct(c10["review_share"])),
            result(pct(c20["shortfall_covered_pct"])), result(pct(c20["review_share"]))]
    # What this project does: the data line
    seq += [result(count(data["loans"])), result(day_month_year(data["snapshot_end"]))]

    # Results: test period size
    seq += [result(count(split["test_loans"]))]
    # Bullet 1: the riskiest 10%
    seq += [result(pct(c10["review_share"])), result(pct(c10["shortfall_covered_pct"])),
            result(money(shortfall["test_total_usd"])), result(pct(c10["review_share"])),
            result(count(c10["loans_reviewed"])), result(money(c10["shortfall_covered_usd"])),
            result(pct(c10["unfunded_loans_covered_pct"])), result(pct(list_precision(c10), 1)),
            result(pct(c10["review_share"])), fixed("10", "a random 10% review reaches 10% of the shortfall")]
    # Bullet 2: PR-AUC against the base rate
    seq += [result(f"{clf['test_pr_auc_at_risk']:.3f}"), result(pct(test_share, 1)),
            fixed("0", "PR-AUC range"), fixed("1", "PR-AUC range"),
            result(pct(test_share, 1)), result(f"{test_share:.3f}")]
    # Bullet 3: the default cut-off
    seq += [fixed("0.5", "default cut-off"), result(pct(clf["at_risk_recall_at_0_5"])),
            result(pct(clf["at_risk_precision_at_0_5"], 1)),
            result(count(shortfall["flagged_loans"])), result(pct(shortfall["flagged_share"], 2)),
            result(pct(shortfall["flagged_shortfall_share"], 1)), fixed("0.5", "default cut-off"),
            result(pct(test_share, 1))]
    # Bullet 4: days to fund
    seq += [result(f"{days['mae_model_days']:.2f}"), result(f"{days['mae_baseline_days']:.2f}"),
            result(f"{days['r2_model']:.2f}")]
    # Shortfall table: one row per review share
    for row in (c5, c10, c20, c30):
        seq += [result(pct(row["review_share"])), result(count(row["loans_reviewed"])),
                result(money(row["shortfall_covered_usd"])), result(pct(row["shortfall_covered_pct"])),
                result(pct(row["unfunded_loans_covered_pct"])), result(pct(list_precision(row), 1))]
    # Model table
    for scores in (chosen, plain, forest, logistic):
        seq += [result(f"{scores['validation_pr_auc_at_risk']:.3f}"), result(f"{scores['test_pr_auc_at_risk']:.3f}")]

    # How I built it, step 1
    seq += [fixed("1", "notebook section"), fixed("2", "notebook section"), fixed("4", "README step number")]
    # Step 2: censoring
    seq += [fixed("3", "notebook section"), result(pct(first_week)), result(pct(oldest)),
            result(str(data["censor_days"])), result(count(data["recent_loans_excluded"])),
            result(pct(data["recent_loans_excluded_share"], 1)), result(count(data["settled_loans"])),
            result(pct(data["settled_not_funded_share"], 1)), result(pct(test_share, 1))]
    # Step 3: the baseline
    seq += [fixed("7", "notebook section"), baseline]
    # Step 4: features and the two poverty scores
    seq += [fixed("4", "notebook section"), fixed("6", "notebook section"),
            fixed("0", "MPI range"), fixed("1", "MPI range"),
            result(count(data["mpi_matched_loans"])), result(pct(data["mpi_coverage"], 1)),
            result(pct(linked["loans_with_linked_mpi_share"], 1)), fixed("12", "README step number")]
    # Step 5: the date split
    seq += [fixed("6", "notebook section"), result(pct(split["train_share"])), result(count(split["train_loans"])),
            result(pct(1 - split["train_share"])), result(count(split["test_loans"])),
            result(count(split["validation_loans"]))]
    # Step 6: four models chosen on validation
    seq += [fixed("8", "notebook section"), result(f"{chosen['validation_pr_auc_at_risk']:.3f}"),
            result(f"{plain['test_pr_auc_at_risk']:.3f}")]
    # Step 7: precision at three recall levels
    seq += [fixed("9", "notebook section"), result(pct(precision_at["0.25"], 1)),
            result(pct(precision_at["0.50"], 1)), result(pct(precision_at["0.75"], 1))]
    seq += [result(pct(test_share, 1))]                          # PR curve description
    seq += [result(pct(test_share, 1))]                          # PR curve caption
    # Step 8: walk-forward
    seq += [fixed("9", "notebook section"), result(count(walk["windows"][0]["test_loans"])),
            result(f"{walk['pr_auc_at_risk_min']:.3f}"), result(f"{walk['pr_auc_at_risk_max']:.3f}"),
            result(f"{walk['lift_over_random_min']:.1f}"), result(f"{walk['lift_over_random_max']:.1f}"),
            result(pct(walk["windows"][0]["not_funded_share"], 1)),
            result(pct(walk["windows"][-1]["not_funded_share"], 1))]
    # Step 9: calibration
    seq += [fixed("9", "notebook section"), result(f"{calibration['brier_raw']:.4f}"),
            result(f"{calibration['brier_calibrated']:.4f}"), result(f"{calibration['brier_constant']:.4f}"),
            result(pct(calibration["mean_predicted_raw"], 1)), result(pct(calibration["actual_unfunded_share"], 1)),
            result(pct(calibration["mean_predicted_calibrated"], 1)), result(day_month_year(split["train_end"])),
            result(f"{calibration['rank_corr_selection_vs_final']:.3f}"),
            result(pct(calibration["mean_predicted_final_model"], 1)),
            result(pct(calibration["actual_unfunded_share"], 1))]
    # Step 10: drivers
    seq += [fixed("11", "notebook section"), result(pct(terms[0], 1)), result(pct(terms[-1], 1))]
    # Step 11: money impact
    seq += [fixed("12", "notebook section")]
    # Step 12: days to fund
    seq += [fixed("10", "notebook section"), result(f"{days['mae_model_days']:.2f}"),
            result(f"{days['mae_baseline_days']:.2f}")]
    # Poverty paragraph
    seq += [result(pct(linked["loans_with_linked_mpi_share"], 1)), result(pct(data["mpi_coverage"], 1))]
    for share in linked["share_fully_funded_by_poverty_fifth"]:
        seq += [result(pct(share, 1))]
    seq += [result(f"{challenger['validation_pr_auc_exact']:.4f}"),
            result(f"{challenger['validation_pr_auc_linked']:.4f}")]
    seq += [fixed("90", "map caption: funded more than 90% of the time")]

    # Problems I hit
    seq += [result(pct(first_week)), result(pct(oldest)), result(str(data["censor_days"]))]
    seq += [fixed("1", "label value"), fixed("1", "label value"), result(pct(clf["test_funded_share"])),
            result(f"{clf['test_pr_auc_funded']:.4f}"), result(f"{clf['test_pr_auc_at_risk']:.4f}")]
    seq += [fixed("0", "0/1 column"), fixed("1", "0/1 column")]
    seq += [result(pct(validation_share, 1)), result(pct(test_share, 1)),
            result(pct(calibration["mean_predicted_calibrated"], 1)),
            result(f"{calibration['brier_raw']:.4f}"), result(f"{calibration['brier_calibrated']:.4f}"),
            fixed("0", "probability range"), fixed("1", "probability range")]
    seq += [result(pct(data["mpi_coverage"], 1)), result(str(data["mpi_misplaced_coordinates"]))]

    # Data table and the paragraph under it
    seq += [result(count(data["loans"])), result(day_month_year(data["snapshot_end"]))]
    seq += [result(count(split["test_loans"])), result(day_month_year(split["train_end"])),
            result(pct(test_share, 1)), result(count(data["mpi_matched_loans"])),
            result(pct(data["mpi_coverage"], 1))]

    # Limits
    seq += [result(pct(validation_share, 1)), result(pct(test_share, 1))]
    seq += [result(f"{male['roc_auc']:.3f}"), result(f"{female['roc_auc']:.3f}")]
    seq += [fixed("9", "README step number")]
    seq += [result(str(data["censor_days"]))]
    seq += [result(day_month_year(provenance["run_date_utc"])), result(str(provenance["runtime_minutes"])),
            result(f"{provenance['peak_memory_gb']:.1f}")]

    # How to run: the README test's own description
    seq += [fixed("60", "the 60-day rule, named as a fixed number")]
    return seq


def test_every_readme_number_in_order():
    expected = expected_numbers()
    found = readme_numbers()
    for index in range(min(len(expected), len(found))):
        text, line_no, line = found[index]
        assert text == expected[index], (
            f"number {index + 1} in the README is {text!r}, expected {expected[index]!r}; "
            f"line {line_no}: {line.strip()[:160]}"
        )
    if len(found) > len(expected):
        extra_text, extra_line, _ = found[len(expected)]
        raise AssertionError(f"README has a number the test does not expect: {extra_text!r} on line {extra_line}")
    assert len(found) == len(expected), f"README ends before the expected number {expected[len(found)]!r}"


def test_notebook_sourced_numbers_are_printed_by_the_notebook():
    expected_numbers()
    assert NOTEBOOK_FIGURES
    for text, printed in NOTEBOOK_FIGURES:
        assert printed in NOTEBOOK_TEXT, f"README figure {text} is not in the notebook's output ({printed!r})"


def number_texts(source):
    # Only the number strings, in reading order
    texts = []
    for text, _, _ in readme_numbers(source):
        texts.append(text)
    return texts


def test_the_reader_catches_a_mistyped_number():
    # Guard the guard: one mistyped figure, a swapped pair or a dropped figure
    # must each change the sequence the main test compares
    expected = expected_numbers()
    assert number_texts(README) == expected
    headline = f"{RESULTS['classification']['test_pr_auc_at_risk']:.3f}"
    mistyped = README.replace(f"PR-AUC is {headline}", "PR-AUC is 0.375", 1)
    swapped = README.replace("from 0.0345 to 0.0371", "from 0.0371 to 0.0345", 1)
    dropped = README.replace("(12,902 loans) reaches", "reaches", 1)
    for changed in (mistyped, swapped, dropped):
        assert changed != README, "the sentence this check edits has moved; update this test"
        assert number_texts(changed) != expected


def sentences(text):
    # Prose sentences. A blank line or a new bullet ends a block, so one bullet
    # can never borrow a word from the next; table rows are left out (a table is
    # checked by its header)
    blocks = []
    current = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            continue
        if not stripped or stripped.startswith("- "):
            if current:
                blocks.append(" ".join(current))
            current = []
        if stripped:
            current.append(stripped)
    if current:
        blocks.append(" ".join(current))
    found = []
    for block in blocks:
        for sentence in re.split(r"(?<=[.!?])\**\s+(?=[A-Z*(])", block):
            found.append(sentence)
    return found


RECALL_WORDS = ("recall", "catches", "catching", "of the loans that went unfunded", "of unfunded loans")


def test_no_recall_figure_without_its_precision():
    # A recall figure alone hides how many reviewed loans did not need help, so
    # every sentence that quotes one must also quote the precision
    offenders = []
    for sentence in sentences(readme_prose()):
        lowered = sentence.lower()
        quotes_recall = False
        for word in RECALL_WORDS:
            if word in lowered:
                quotes_recall = True
        if quotes_recall and re.search(r"\d", sentence) and "precision" not in lowered:
            offenders.append(sentence.strip()[:160])
    assert not offenders, f"recall quoted without precision: {offenders}"


def test_capture_table_shows_precision_next_to_recall():
    assert "| Unfunded loans reached (recall) | Reviewed loans that went unfunded (precision) |" in README


def test_headline_sentences():
    # The headline sentences, pinned whole so a stray copy elsewhere cannot stand in
    clf = RESULTS["classification"]
    shortfall = RESULTS["shortfall"]
    c10 = capture_row(0.1)
    test_share = RESULTS["split"]["not_funded_share"]["test"]
    assert f"{clf['test_pr_auc_at_risk']:.3f}" == "0.374" and f"{test_share:.1%}" == "4.6%"
    assert f"**The riskiest 10% of loans hold {c10['shortfall_covered_pct']:.0%} of the unfunded dollars.**" in README
    assert f"loans fell ${shortfall['test_total_usd']:,.0f} short" in README
    assert (f"**The model's PR-AUC is {clf['test_pr_auc_at_risk']:.3f}, against a base rate of "
            f"{test_share:.1%}.**") in README
    assert (f"it catches {clf['at_risk_recall_at_0_5']:.0%} of unfunded loans at a precision of "
            f"{clf['at_risk_precision_at_0_5']:.1%}") in README
    assert f"| **LightGBM + time trend (chosen)** | **0.490** | **{clf['test_pr_auc_at_risk']:.3f}** |" in README


def test_fairness_gap_is_disclosed():
    # The weaker ranking for all-male loans stays in Limits with both figures
    male = segment("borrower_group", "all male")
    female = segment("borrower_group", "all female")
    assert male["roc_auc"] < female["roc_auc"]
    assert (f"the model scores {male['roc_auc']:.3f} on all-male loans against {female['roc_auc']:.3f} on "
            f"all-female loans") in README
    assert "Check this before the" in README


def test_decision_framing_names_who_acts():
    assert "The ranking is for whoever decides, each day, which new loans get that help" in README


def test_post_funding_fields_are_named_as_excluded():
    # The README names the columns the leakage guard keeps out of the model
    assert "funded time, lender count, funded amount" in README


def test_run_provenance_matches_results_file():
    # The README names the run that wrote outputs/results.json. After a re-run
    # this fails until the Limits sentence is updated to the new run.
    provenance = RESULTS["provenance"]
    assert provenance["uncommitted_code_changes"] is False
    sentence = (f"**The recorded run is from {day_month_year(provenance['run_date_utc'])}**, "
                f"{provenance['runtime_minutes']} minutes and {provenance['peak_memory_gb']:.1f} GB")
    assert sentence in README
