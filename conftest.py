"""Puts src/ on sys.path for the tests.

src/features.py is imported by the generated notebook as `from features import ...`,
so the tests import it the same way rather than through a package alias that would
give the notebook and the tests two separate module objects with two separate
copies of LEAKY_COLUMNS.
"""
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
