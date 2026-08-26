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

One more pins the façade. The definitions live in `warrantlib._vocabulary` and
`warrantlib._serialise`, and `warrantlib/__init__.py` re-exports them by hand, so a new
public name added to one and forgotten in the other is invisible: it imports fine from
its own module and is absent from the published surface. The source is read rather than
the imported module, because the imported module cannot tell a definition from an
import.

`warrantlib.pytest_plugin` is outside that rule, also deliberately. Its public names
are pytest hooks and a fixture, which pytest finds through the entry point, and putting
them in `__all__` would publish an import path nobody should use.

`cpomdp.warrant` is held to a weaker rule, and deliberately. It exists so import paths
that predate the split keep working, so it does not track warrantlib name for name. A
name with no old path to preserve stays out, because carrying it would build a second
public surface to maintain past 1.0 (ADR-039).

The exception is a name `Evidence` admits. That alias is re-exported here, so widening
the union widens what this path accepts whether or not the new members are named on it,
and a caller annotating against it could be handed a type it had no import for. The
completeness leaves are carried for that reason and not as a general policy of tracking
warrantlib. The set below is the whole surface, floor and ceiling both, so either kind
of drift shows up as a failure rather than as a name nobody meant to publish.
"""

import ast
import subprocess
import sys
import typing
from pathlib import Path

import cpomdp
import cpomdp.warrant
import warrantlib
from cpomdp.enumeration import CompletenessCertificate, SearchWarrant
from warrantlib import _serialise, _vocabulary

_NO_CPOMDP = """
import sys
import warrantlib

leaked = sorted(name for name in sys.modules if name.split(".")[0] == "cpomdp")
if leaked:
    raise SystemExit("warrantlib pulled in " + ", ".join(leaked))
"""

_NO_PYTEST = """
import sys
import warrantlib

if "pytest" in sys.modules:
    raise SystemExit("warrantlib pulled in pytest")
"""


def test_warrantlib_imports_nothing_from_cpomdp():
    completed = subprocess.run(
        [sys.executable, "-c", _NO_CPOMDP], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_warrantlib_imports_no_pytest():
    # The plugin module imports pytest and is reached through the `pytest11` entry
    # point, never from `__init__`. An import added there would put pytest on the
    # dependency list of a distribution whose list is empty on purpose (ADR-039), and
    # nothing else in the suite would notice, because pytest is always installed here.
    completed = subprocess.run(
        [sys.executable, "-c", _NO_PYTEST], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_cpomdp_warrant_re_exports_every_name_it_carries():
    for name in cpomdp.warrant.__all__:
        assert getattr(cpomdp.warrant, name) is getattr(warrantlib, name), name
    assert set(cpomdp.warrant.__all__) <= set(warrantlib.__all__)


def test_the_shim_carries_the_vocabulary_cpomdp_once_exported():
    # The subset rule above is satisfied by a shim that has lost a name, which is the
    # breakage it exists to prevent. This is the floor it may not fall below, and the
    # ceiling too: growing it is a decision about cpomdp's public surface, so it is made
    # by editing this set rather than by an import going unnoticed.
    assert set(cpomdp.warrant.__all__) == {
        "AxisDeclaration",
        "CheckReport",
        "CompletenessCertificate",
        "CompletenessEvidence",
        "Evidence",
        "Outcome",
        "ProductCompletenessCertificate",
        "Provenance",
        "SymbolicReduction",
        "Tier",
        "Warrant",
        "check_summary",
    }


def test_every_member_of_the_evidence_union_is_nameable_from_the_shim():
    # The reason the completeness leaves are here at all. `Evidence` is re-exported, so
    # a caller annotating against it can be handed any member; one it cannot import from
    # the same path is a type it can receive and not name.
    for member in typing.get_args(cpomdp.warrant.Evidence):
        assert member.__name__ in cpomdp.warrant.__all__, member.__name__


def test_top_level_names_are_the_warrantlib_objects():
    for name in ("CheckReport", "Outcome", "SymbolicReduction", "Tier", "Warrant"):
        assert getattr(cpomdp, name) is getattr(warrantlib, name), name
    assert cpomdp.check_summary is warrantlib.check_summary


def test_enumeration_re_exports_the_certificate_and_the_alias():
    assert CompletenessCertificate is warrantlib.CompletenessCertificate
    assert SearchWarrant is warrantlib.Warrant


def _defined_public_names() -> set[str]:
    """Top-level public names the private modules define, read from their source."""
    names = set()
    sources = (Path(_vocabulary.__file__), Path(_serialise.__file__))
    for node in [n for path in sources for n in ast.parse(path.read_text()).body]:
        if isinstance(node, ast.ClassDef | ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
    return {name for name in names if not name.startswith("_")}


def test_the_facade_exports_every_public_vocabulary_name():
    defined = _defined_public_names()
    # A guard on the guard: an empty set would pass the comparison below by asking
    # nothing, exactly as a renamed module would.
    assert len(defined) >= 9
    assert defined == set(warrantlib.__all__)
