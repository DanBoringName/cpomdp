# Contributing to cpomdp

Thanks for taking a look. Here's how to get set up and what the tooling expects.

## The quality bar

The code here follows Martin Fowler's refactoring practices and the SOLID
principles. Test Driven Development is encouraged. If you develop something in
the linear-Gaussian space try to also use an exact oracle. Naming conventions
ideally spell out their function with a comment of their single letter
counter-part (if they have one) e.g. `observation_matrix` followed by `C`.
Where a few variables are parts of one thing, prefix them all with the name of
that thing: `observation_matrix` and `observation_noise` are both parts of the
observation channel, and while they're flat fields the prefix is the only thing
saying so. That's the default rather than a hard rule. If it doesn't fit, it doesn't fit, no problem. This is as much a note to myself as anyone else.

A PR has to clear the same bar. Code that is obviously machine-generated and
dumped in without consideration of the previous points will be rejected.
However, I appreciate not everyone is a software engineer and I am more than
happy to help anyone who wishes to contribute but is unsure on how to meet
any of the above.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). Once you've cloned it:

```bash
uv sync                  # install the package + dev tooling
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

That second line wires up the git hooks. You only do it once. After that the
checks run automatically every time you commit, so you find problems before CI
does rather than after.

## The workspace

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
with three members, one lockfile and one `.venv`, all at the root:

| member | what it is |
| --- | --- |
| `.` | `cpomdp`, the library. The workspace root. |
| `packages/warrantlib` | the warrant vocabulary. Publishes to PyPI on its own, depends on the standard library alone. |
| `research` | `research.checks`, the symbolic suites, plus the registrations they gate. Never published. |

`uv sync` installs all three. You don't need `--all-packages`: the research member is
a `dev` dependency of the root, so a plain `uv sync` or `uv run` cannot leave you with
an environment where `python -m research.checks.gap_series` fails to import.

If you use VSCode, open `cpomdp.code-workspace` (File > Open Workspace from File)
rather than the directory. It puts the three members at the top of the explorer
instead of buried, and points the Python extension at the root `.venv`. It's committed
so a git worktree gets its own copy: check the worktree out, run `uv sync --locked` in
it, open its `.code-workspace`, and it resolves against that worktree's venv.

## How the rules are enforced

There's one source of truth for style and linting: the `[tool.ruff]` section of
`pyproject.toml`. Nothing depends on which editor you use. `.vscode/settings.json`
is gitignored, and `cpomdp.code-workspace` is a convenience, not a gate. The config is
enforced in two places that both read it:

- **pre-commit**, locally, on every commit (see `.pre-commit-config.yaml`).
- **CI**, on every push and PR, running the exact same hooks.

So if it's green locally, it's green in CI. If you want your editor to format on
save, point it at ruff yourself. Just don't rely on it. The hooks are what count.

## What the hooks check

- **ruff** lints and formats the code. Line length is 88. Formatting isn't a
  matter of taste here, ruff decides and that's that.
- **markdown** is linted by [markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2)
  (config in `.markdownlint-cli2.yaml`). It auto-fixes the mechanical stuff:
  blank lines around headings, fenced blocks and lists. So most of the time it
  just tidies your edit and asks you to re-stage. The only things it can't fix for
  you are ones it can't guess, like naming a code fence's language. Prose lines
  aren't wrapped, so line length isn't checked.
- **spelling** in markdown and in `src/` docstrings is checked by
  [cspell](https://cspell.org) (`cspell.json`). It's British English and seeded
  with the project's vocabulary (cpomdp, Kalman, pytree, ...). If it flags a real
  term it doesn't know yet, add it to the `words` list in that file rather than
  working around it. It splits `snake_case`/`camelCase` into real words, so it
  reads your docstrings without choking on most identifiers.
- **docstrings** are required on public modules, classes, functions and methods in
  `src/` (Google style). Tests are exempt, because their names are the documentation.
  Constructors can be documented at the class level instead of in `__init__`.
- **commit messages** follow [Conventional Commits](https://www.conventionalcommits.org):
  `feat:`, `fix:`, `docs:`, `test:`, `chore:`, and so on. The commit-msg hook will
  bounce anything that doesn't.
- a few **hygiene** checks: no trailing whitespace, files end in a newline, YAML and
  TOML parse, no leftover merge-conflict markers.

## The docs site

Four pages under `docs/` are one-line stubs: `index.md`, `changelog.md`,
`examples.md` and `examples-ffg.md`. Each is a `--8<--` snippet include of a file
that lives outside `docs/`, because GitHub and PyPI expect `README.md` and
`CHANGELOG.md` at the repo root while MkDocs only serves what's under `docs_dir`.
Edit the source file, not the stub. `mkdocs_hooks.py` pastes the content in at
build time and repoints the relative links, so one file reads correctly in both
places. The stubs are excluded from markdownlint and cspell.

```bash
uv run --group docs mkdocs build --strict # what CI runs, and it fails on a bad link
uv run --group docs mkdocs serve          # live preview at localhost:8000
```

## Running things by hand

```bash
uv run pytest -m "not rxinfer and not slow"   # the fast, pure-Python suite
uv run pytest -m "not rxinfer"                # + the slow crossover gate (H*=7)
uv run --extra examples ty check # type checking (examples are in the checked tree)
uv run pre-commit run --all-files
```

The symbolic suites are deliberately not in `testpaths`. `gap_series` derives `c₂` and
`c₄` symbolically and costs about half a minute, which nobody wants on every run. They
have their own command, and their own CI job:

```bash
uv run pytest research/registered_checks.toml   # every registered check, reconciled
uv run pytest research/registered_checks.toml --warrant-detail  # + each check's reason (or -vv)
uv run python -m warrantlib.manifest --check research/registered_checks.toml
```

The first turns each check the manifest declares into an item, so a check that stops
reporting fails by name. The second says whether the manifest itself is current. Rewrite
it after adding or renaming a check:

```bash
uv run python -m warrantlib.manifest research/registered_checks.toml
```

The new ids land in the diff, which is where they get reviewed. Each suite is also still
runnable on its own, printing its summary and exiting non-zero if a check fires:

```bash
uv run python -m research.checks.series_kernel --check
```

The `rxinfer` tests boot a Julia runtime (the RxInfer backend is an independent
oracle the native filter is checked against). They're slow and need the extra:

```bash
uv run --extra rxinfer pytest -m rxinfer
```

You don't need Julia for normal work. The pure-Python suite covers the core, and
the rxinfer job runs separately in CI.
