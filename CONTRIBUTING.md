# Contributing to cpomdp

Thanks for taking a look. Here's how to get set up and what the tooling expects.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). Once you've cloned it:

```bash
uv sync                  # install the package + dev tooling
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

That second line wires up the git hooks. You only do it once. After that the
checks run automatically every time you commit, so you find problems before CI
does rather than after.

## How the rules are enforced

There's one source of truth for style and linting: the `[tool.ruff]` section of
`pyproject.toml`. Editor settings aren't checked in on purpose, so nothing depends
on which editor you use. The config is enforced in two places that both read it:

- **pre-commit**, locally, on every commit (see `.pre-commit-config.yaml`).
- **CI**, on every push and PR, running the exact same hooks.

So if it's green locally, it's green in CI. If you want your editor to format on
save, point it at ruff yourself; just don't rely on it, the hooks are what count.

## What the hooks check

- **ruff** lints and formats the code. Line length is 88. Formatting isn't a
  matter of taste here, ruff decides and that's that.
- **docstrings** are required on public modules, classes, functions and methods in
  `src/` (Google style). Tests are exempt; their names are the documentation.
  Constructors can be documented at the class level instead of in `__init__`.
- **commit messages** follow [Conventional Commits](https://www.conventionalcommits.org):
  `feat:`, `fix:`, `docs:`, `test:`, `chore:`, and so on. The commit-msg hook will
  bounce anything that doesn't.
- a few **hygiene** checks: no trailing whitespace, files end in a newline, YAML and
  TOML parse, no leftover merge-conflict markers.

## Running things by hand

```bash
uv run pytest -m "not rxinfer"   # the fast, pure-Python suite
uv run ty check                  # type checking
uv run pre-commit run --all-files
```

The `rxinfer` tests boot a Julia runtime (the RxInfer backend is an independent
oracle the native filter is checked against). They're slow and need the extra:

```bash
uv run --extra rxinfer pytest -m rxinfer
```

You don't need Julia for normal work. The pure-Python suite covers the core, and
the rxinfer job runs separately in CI.
