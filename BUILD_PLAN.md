# cpomdp build plan / progress tracker

Tracked, surviving replacement for the old `.claude/cpomdp_v0.4_build_plan.md`,
which was gitignored and didn't transfer between sittings. Authoritative
decisions still live in `DECISIONS.md` (ADRs); this file is the running
checklist of what's built and what's next.

Conventions: `[x]` done, `[ ]` open, `[~]` partial. Phases follow ADR-012.

---

## v0.4 — FFG message passing (ADR-012)

Generalise the Kalman/EFE machinery to a Forney factor graph so the E. coli
chemotaxis network — with its shared `CheA` node feeding a fast (CheY-P/motor)
and a slow (CheR/CheB methylation) branch — is representable. Canonical
(information) form; from-scratch JAX; RxInfer narrowed to oracle-only;
hand-authored schedule.

### Phase 0 — scaffolding decisions — DONE (commit `7a2713c`)

- [x] The four ADR-012 choices settled: from-scratch-JAX, RxInfer-as-oracle,
      canonical-form messages, hand-authored schedule.

### Phase 1 — `CanonicalGaussian` message algebra — DONE (2026-06-25, pending commit)

The FFG wire payload: `src/cpomdp/ffg/message.py`, spec in
`tests/test_ffg_message.py`. 255 tests green, `ty` clean, `ruff` clean.

- [x] Scaffold — construct/validate `(Λ, h)`, `ndim`, pytree flatten/unflatten.
- [x] `__add__` — factor product as elementwise sum; jit-safe shape guard;
      builds via the no-validate seam (no inversion on this path).
- [x] `to_moment` — `(mean, cov)` readout via solve/inv; positive-definite
      guard; `h` = potential = information-vector naming pinned in the docstring.
- [x] `marginalize` — Schur-complement elimination; kept indices ascending;
      positive-definite guard on the eliminated block; only that block inverted.
- [x] `_unchecked` — shared non-validating constructor (hot-path-lean,
      tracer-clean); `tree_unflatten` de-duplicated onto it.
- [x] Supporting: `_validation.py` symmetry check made trace-safe (latent bug
      that blocked construction under `jit`); associativity oracle relaxed to
      `allclose` (IEEE addition isn't associative); `cspell` dict += `elim`,
      `Schur`.

Parked open question: a `from_moment` / moment-form constructor (none in v0.4;
moment form is readout-only via `to_moment`).

### Phase 2 — factor nodes + chain = Kalman byte-identity gate — NEXT

- [ ] Factor nodes that *produce* messages (emit `CanonicalGaussian`s from
      incoming ones) — the grammar over the Phase 1 alphabet.
- [ ] Hand-authored schedule for the fixed chemotaxis graph (no reactive
      scheduler — out of scope).
- [ ] **KEYSTONE GATE:** a linear chain of factor nodes is *byte-identical* to
      the existing Kalman path (not mere agreement) — the chain is the filter's
      degenerate case.
- [ ] RxInfer oracle check on small graphs (behind the `rxinfer` marker).
- [ ] jit/grad/vmap smoke tests as gates on every new public inference entry.

### Out of scope (ADR-012 — say no on sight)

General `@model` frontend; tier-2 conjugate-exponential engine (seam stubbed,
deferred to v0.5+); reactive scheduling / automatic conjugacy; constrained
Bethe Free Energy as a general objective; structure *learning*.
