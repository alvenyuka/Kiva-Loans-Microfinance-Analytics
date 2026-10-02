"""Every number quoted in README.md must appear in outputs/results.json.

The notebook writes results.json at the end of each run. A README figure that
cannot be traced to it is either stale (the model changed and the prose did not)
or was never produced by the code. Both are caught here without the dataset.

A README number matches when some value in results.json, formatted the way the
README writes numbers (thousands separators, rounding to 0 to 4 decimals,
percentages, dollar amounts in millions, dates), gives the same text.
"""
import json
import re
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "results.json"
README = ROOT / "README.md"

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def _leaves(obj):
    """Every scalar value in the results tree."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _leaves(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _leaves(v)
    else:
        yield obj


def _renderings(value):
    """The ways the README may write one value."""
    out = set()
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, str):
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
        if m:
            d = date(int(m[1]), int(m[2]), int(m[3]))
            out.add(f"{d.day} {MONTHS[d.month - 1]} {d.year}")
        return out
    v = float(value)
    for scaled, suffix in ((v, ""), (v * 100, "%"), (v / 1e6, "M"), (v / 1e9, "B")):
        for dp in range(0, 5):
            s = f"{scaled:,.{dp}f}"
            out.add(s + suffix)
            out.add(s.replace(",", "") + suffix)
    return out


def _allowed():
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    allowed = set()
    for leaf in _leaves(results):
        allowed |= _renderings(leaf)
    return allowed


def _readme_numbers(text):
    """Numbers in the README prose and tables, without code, links, badges or list markers."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # code blocks and the mermaid chart
    text = re.sub(r"\]\([^)]*\)", "]", text)                     # link and image targets
    text = re.sub(r"^\[!\[.*$", " ", text, flags=re.M)           # badge lines
    text = re.sub(r"^\s*\d+\.\s", " ", text, flags=re.M)         # numbered list markers
    dates = re.findall(r"\b\d{1,2} (?:%s) \d{4}\b" % "|".join(MONTHS), text)
    text = re.sub(r"\b\d{1,2} (?:%s) \d{4}\b" % "|".join(MONTHS), " ", text)
    numbers = re.findall(r"(?<![\w.])\$?(\d[\d,]*(?:\.\d+)?)(%|M\b|B\b)?", text)
    return dates + [n.rstrip(",") + (suffix or "") for n, suffix in numbers]


def test_results_file_exists():
    assert RESULTS.exists(), "outputs/results.json is missing: execute the notebook to write it"


def test_every_readme_number_is_in_results_json():
    if not RESULTS.exists():
        pytest.skip("results.json not written yet")
    allowed = _allowed()
    missing = sorted({n for n in _readme_numbers(README.read_text(encoding="utf-8")) if n not in allowed})
    assert not missing, f"README numbers not found in outputs/results.json: {missing}"


def test_the_check_catches_a_stale_number():
    """Guard the guard: a number the results cannot produce must fail."""
    allowed = _allowed()
    assert "0.987654" not in allowed
    assert [n for n in _readme_numbers("PR-AUC 0.987654 on 123,456,789 loans") if n not in allowed]
