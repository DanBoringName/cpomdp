"""Pytest configuration.

Put ``examples/`` on the import path so the acceptance tests can import the models that
are built *with* cpomdp (e.g. the chemotaxis network) rather than shipped inside the
installable library. The examples stay plain, runnable scripts; this only affects how
the test process resolves them.
"""

import sys
from pathlib import Path

_examples = Path(__file__).parent / "examples"
sys.path.insert(0, str(_examples))
sys.path.insert(0, str(_examples / "ffg"))
