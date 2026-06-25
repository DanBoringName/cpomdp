## Summary

<!-- Briefly describe what this PR does and why. -->

## Checklist

- [ ] `uv run pytest -m "not rxinfer"` passes
- [ ] `uv run ty check` is clean
- [ ] `uv run pre-commit run --all-files` is clean (ruff, markdownlint, cspell, docstrings)
- [ ] New public API in `src/` has Google-style docstrings
- [ ] PR title follows Conventional Commits (`feat:` / `fix:` / `docs:` / …)
- [ ] `CHANGELOG.md` updated if user-facing
- [ ] Relevant ADR/RFC linked if this touches a recorded decision
