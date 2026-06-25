## bug_report.md

```markdown
---
name: Bug report
about: Create a report to help us fix an issue
title: "[BUG] "
labels: bug
assignees: ""
---

## Expected vs. Actual Behaviour

- **Expected:** What you expected to happen.
- **Actual:** What actually happened.

## Environment

- **cpomdp version** (run `pip show cpomdp` or `print(cpomdp.__version__)`):
- **Python version:**
- **Operating System:**
- **JAX version:**
- **Is JAX x64 enabled?** (Yes/No):
- **Is the optional `rxinfer` extra installed?** (Yes/No):

## Minimal Reproducible Example

Please provide a minimal script that reproduces the bug.

```python
# Your code here
```

## Full Traceback

If an exception was raised, paste the full traceback here.

```text
# Traceback here
```
```

---

## feature_request.md

```markdown
---
name: Feature request
about: Propose a new feature, API, or behaviour
title: "[FEAT] "
labels: enhancement
assignees: ""
---

## The Problem / Use-Case

A clear and concise description of the problem or the context for the requested feature.

## Proposed API or Behaviour

Describe how you envision this feature working. Include pseudo-code or an example of how the new API would look if applicable.

## Relevant ADR / RFC

If this idea touches on any existing Architecture Decision Records (in `DECISIONS.md`) or previous discussions, mention them here.
```