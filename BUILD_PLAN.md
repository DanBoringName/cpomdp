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

### Phase 2 — factor nodes + chain = Kalman numerical-identity gate — IN PROGRESS (RxInfer oracle open)

Tier-1 linear-Gaussian factor nodes (`src/cpomdp/ffg/factors/linear_gaussian.py`)
plus the chain backend (`src/cpomdp/ffg/chain.py`); specs in
`tests/test_ffg_factors.py` and `tests/test_ffg_chain.py`. Registered as JAX
pytrees. 290 tests green, `ty`/`ruff` clean.

- [x] **Observation factor** — `GaussianObservation.message(y)` = the likelihood's
      information form `(CᵀR⁻¹C, CᵀR⁻¹y)`; the update is `belief + message`. Oracle:
      moment-form measurement update.
- [x] **Transition factor** — `GaussianTransition.predict(message, control_term)`:
      build the joint over `[x, x']`, fold the message into the x block, marginalize
      x out. Oracle: moment-form predict `AΣAᵀ+Q` / `Aμ+b`. PD-Q only — the
      information form inverts Q, so no deterministic (`Q=0`) transition.
- [x] **Chain backend** — `src/cpomdp/ffg/chain.py`: `ChainBackend` wires
      `lift → predict → update → to_moment` into `infer_states` (satisfies the
      `InferenceBackend` protocol). Factors front-loaded in `__init__`; the moment→
      canonical lift builds via `_unchecked`, so the eager loop's only validation is
      the output `Belief` — same per-step cost as `KalmanBackend`. Tier-1 fixed only
      (state-dependent R(x)/Q(x) rejected → Phase 2.5; `Q=0` rejected as the
      info-form divergence).
- [x] **KEYSTONE GATE** — `tests/test_ffg_chain.py` (18 tests): numerical identity
      (atol 1e-7) vs `KalmanBackend` over sequences, dims (1,1)→(4,3), with/without
      control; plus an independent NumPy scalar-filter oracle. (Tolerance note below.)
- [ ] RxInfer oracle check on small graphs (behind the `rxinfer` marker).
- [x] jit/grad/vmap smoke tests as gates on every new public inference entry
      (`TestChainBackendTransforms`).

Tolerance note: the keystone is *numerical* identity (atol 1e-7), not literal
bit-for-bit — info-vs-moment form inverts/re-inverts. ADR-012's "byte-identity"
wording amended accordingly (2026-06-26).

### Phase 2.5 — `ChainBackend` R(x)/Q(x) parity — PLANNED

Before v0.4 ships, the FFG chain path reaches feature parity with `KalmanBackend`
on state-dependent noise (decided 2026-06-26; recorded in ADR-012). Phase 2 ships
fixed-matrix only (rejected at construction) to keep the keystone clean; this phase
lifts that restriction via the same *linearize-at-μ⁻ plug-in* Kalman already uses.

- [ ] After `predict`, read μ⁻ from the predicted message; build the observation
      factor from `observation.linearize(μ⁻)` and the transition's Q from
      `process_noise.noise_at(μ⁻)` that step (per-step factors on this path only —
      the fixed path keeps front-loaded factors).
- [ ] Drop the Phase 2 scope rejection; gate against `KalmanBackend`'s R(x)/Q(x)
      path (the existing `test_kalman.py` R(x)/Q(x) oracles, ported to the chain).

### Extras

- [ ] Demos: A comparison v0.3 kalman demo with v0.4 ffg demo with R(x) revealing goal position, not state agent precision.
- [ ] Update contribution section of the docs that explicitly state code blatantly wrote by AI with zero regard for quality cpomdp tries to upkeep will result in PR being closed.

### Out of scope (ADR-012 — say no on sight)

General `@model` frontend; tier-2 conjugate-exponential engine (seam stubbed,
deferred to v0.5+); reactive scheduling / automatic conjugacy; constrained
Bethe Free Energy as a general objective; structure *learning*.
