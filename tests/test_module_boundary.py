import ast
from pathlib import Path

import pytest

# The seam PR-3 built. Paper 3 reuses it and does not import the three-term evaluator,
# so neither may these, however many modules deep the route runs.
SEAM = ("harness", "constructors")

# The evaluator's module, named here before it exists. A test that waits for the
# import to be possible is a test that arrives after the import has been written.
EVALUATOR = "scoring"

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "cpomdp"


def _path_of(module: str) -> Path | None:
    """The file backing a module named relative to ``cpomdp``, if there is one."""
    for candidate in (
        PACKAGE.joinpath(*module.split(".")).with_suffix(".py"),
        PACKAGE.joinpath(*module.split("."), "__init__.py"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _imports_of(path: Path) -> set[str]:
    """The cpomdp submodules a file imports, named relative to the package.

    Covers the three forms the tree uses: ``import cpomdp.x``, ``from cpomdp.x import
    y``, and ``from cpomdp import x``. A relative import would be missed, and none
    exists here — ``test_the_tree_imports_absolutely`` is what keeps that true.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {
                alias.name.removeprefix("cpomdp.")
                for alias in node.names
                if alias.name.startswith("cpomdp.")
            }
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "cpomdp":
                found |= {alias.name for alias in node.names}
            elif node.module.startswith("cpomdp."):
                found.add(node.module.removeprefix("cpomdp."))
    return found


def _closure(*modules: str) -> set[str]:
    """Every cpomdp module reachable from ``modules`` by following imports.

    ``cpomdp/__init__.py`` is not followed. It re-exports the public surface, so
    following it would make every module depend on every other and the boundary would
    be untestable rather than unbroken.
    """
    seen: set[str] = set()
    queue = list(modules)
    while queue:
        module = queue.pop()
        if module in seen or module == "":
            continue
        seen.add(module)
        path = _path_of(module)
        if path is not None:
            queue.extend(_imports_of(path))
    return seen


# --- the walk finds things, before it is asked to find nothing ----------------------


def test_the_walk_finds_a_direct_import():
    assert "types" in _closure("harness")


def test_the_walk_follows_an_import_of_an_import():
    # harness imports backends.kalman, which is where _validation enters. Nothing in
    # harness names it, so finding it proves the walk went past the first hop.
    assert "backends.kalman" in _closure("harness")
    assert "_validation" in _closure("harness")


def test_the_walk_stops_where_the_first_party_imports_do():
    # cpomdp.warrant re-exports warrantlib and reaches nothing else in the tree, so its
    # closure is itself. A walk that over-collected would not come back with one name.
    assert _closure("warrant") == {"warrant"}


def test_a_module_outside_the_seam_is_not_dragged_in():
    assert "ffg.chain" not in _closure("constructors")


# --- the boundary ------------------------------------------------------------------


@pytest.mark.parametrize("module", SEAM)
def test_the_seam_does_not_reach_the_evaluator(module):
    assert EVALUATOR not in _closure(module)


def test_the_seam_does_not_reach_an_action_selector():
    # A ScoredAgent is driven, never choosing. Reaching selection from here would put
    # the machinery for it one import away from an object that must not have it.
    assert "selection" not in _closure(*SEAM)
    assert "enumeration" not in _closure(*SEAM)


def test_the_tree_imports_absolutely():
    relative = [
        path.relative_to(PACKAGE)
        for path in PACKAGE.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert not relative
