"""`warrantlib` stands alone, and cpomdp's old import paths still reach it.

The vocabulary is published as its own distribution with no dependencies, so a check
suite can label its findings without a numerical stack. That property is invisible in
ordinary use: every test in this repository imports cpomdp anyway, so a fresh import of
`cpomdp` inside `warrantlib` would go unnoticed until someone installed it on its own.
The first test below runs the import in a clean interpreter and fails if any `cpomdp`
module is reachable afterwards.

The rest pin the re-exports. `cpomdp.warrant` and `cpomdp.enumeration` name objects that
are defined in `warrantlib`, and a copy rather than a re-export would break every
`isinstance` check that crosses the two.
"""

import subprocess
import sys

import cpomdp
import cpomdp.warrant
import warrantlib
from cpomdp.enumeration import CompletenessCertificate, SearchWarrant

_NO_CPOMDP = """
import sys
import warrantlib

leaked = sorted(name for name in sys.modules if name.split(".")[0] == "cpomdp")
if leaked:
    raise SystemExit("warrantlib pulled in " + ", ".join(leaked))
"""


def test_warrantlib_imports_nothing_from_cpomdp():
    completed = subprocess.run(
        [sys.executable, "-c", _NO_CPOMDP], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_cpomdp_warrant_re_exports_every_name():
    for name in warrantlib.__all__:
        assert getattr(cpomdp.warrant, name) is getattr(warrantlib, name), name
    assert set(cpomdp.warrant.__all__) == set(warrantlib.__all__)


def test_top_level_names_are_the_warrantlib_objects():
    for name in ("CheckReport", "Outcome", "SymbolicReduction", "Tier", "Warrant"):
        assert getattr(cpomdp, name) is getattr(warrantlib, name), name
    assert cpomdp.check_summary is warrantlib.check_summary


def test_enumeration_re_exports_the_certificate_and_the_alias():
    assert CompletenessCertificate is warrantlib.CompletenessCertificate
    assert SearchWarrant is warrantlib.Warrant
