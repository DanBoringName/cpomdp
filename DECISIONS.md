# Architecture Decisions

Decisions are append-only. Each records the choice, the evidence, and the date.

---

## ADR-002 — v0.1 inference engine: **native fixed-gain fast path; RxInfer as oracle + general fallback**

**Date:** 2026-06-12
**Status:** Accepted
**Phase:** 2 (the abstraction wall)
**Amends:** ADR-001 (does not revoke it — re-roles RxInfer rather than removing it)

### Decision

v0.1's *default* inference is a **native, front-loaded steady-state Kalman
filter** (Option 1 in the build plan), exposed as a backend behind the
`InferenceBackend` Protocol. **RxInfer (via juliacall, per ADR-001) is retained
as a second backend** — serving now as the *correctness oracle* and later as the
*general engine* for the cases the native fast path cannot handle (nonlinear,
non-stationary, intermittent observations, structure learning, hierarchical).

### Why this changes ADR-001's emphasis

ADR-001 made RxInfer "the engine." The front-loading analysis (RESEARCH.md) shows
that for the **LTI-Gaussian** v0.1 scope the inference loop reduces to a fixed-gain
filter so cheap that RxInfer would never run in the hot path — it would be a Julia
dependency carried for nothing. We arrive at the native path *not* because the
bridge failed (it worked, ADR-001 stands as evidence) but because front-loading
removes the only reason the bridge was load-bearing. The Phase-2 abstraction wall
is exactly what lets both coexist as swappable backends instead of a fork.

### The principle being implemented (RESEARCH.md)

**Front-load the *structure* of the computation, never the *values*.** For an LTI
Gaussian model the covariance/gain sequence is data-independent: solve the
discrete algebraic Riccati equation (DARE) **once at agent construction** to get
the steady-state gain `K∞`, then run a fixed-gain update in the loop. No
inversion, no covariance update, no O(n³) op in the hot path.

### Scope guards (resisting the doc's own scope creep)

- **In for v0.1:** `Belief` (plain covariance, scalar), `InferenceBackend`
  Protocol, native fixed-gain backend (DARE → `K∞` + warmup), RxInfer oracle
  backend, 2D point-mass reaching demo validated against a full per-step Kalman.
- **Deferred (named seams only, no impl):** `CovarianceRep` strategy/Protocol
  (YAGNI until a 2nd representation exists — scalar is the trivial 1×1 case of
  all three), BMR outer loop, LQR/control side.
- **JAX:** not adopted reflexively. v0.1 scalar fixed-gain is instant in NumPy;
  JAX is revisited when autodiff (EFE gradients, param learning) or vmap/GPU
  actually pays. Core stays NumPy-only until then.

### Boundaries where the native fast path is INVALID (fall back to RxInfer)

- Nonlinear models — EKF/UKF gains depend on the linearisation point → the
  estimate → the data → gains become data-dependent → not front-loadable.
- Non-stationary `A,Q,R` — `K∞` goes stale; needs drift detection + re-solve.
- **Intermittent / irregularly-sampled / varying-`R` observations** — breaks the
  "regular complete observations" assumption that makes `K` constant.

### Validation strategy

The native filter's posterior is checked against (a) a plain NumPy RTS
smoother / full per-step Kalman (analytic oracle) and (b) the RxInfer backend.
The Phase-0 spike (`spike/`) is re-roled from "shipping engine prototype" to
"oracle harness."

---

## ADR-001 — Backend bridge shape: **Shape A (juliacall, in-process)**

**Date:** 2026-06-12
**Status:** Accepted (emphasis amended by ADR-002)
**Phase:** 0 (verification spike — the gate)

### Decision

cpomdp's v0.1 inference engine is **RxInfer.jl, reached in-process via `juliacall`**
(Shape A). Not the HTTP `RxInferClient.py` → `RxInferServer.jl` route (Shape B).

### Evidence from the spike (`spike/`, throwaway)

A scalar linear-Gaussian state-space model was the test vehicle:

    xₜ = A·xₜ₋₁ + 𝒩(0,Q),   yₜ = B·xₜ + 𝒩(0,R),   x₀ ~ 𝒩(m0,v0)

1. **Julia-only ground truth** (`lgssm_groundtruth.jl`): RxInfer runs, posteriors
   read out cleanly. **Validated correct** against an independent NumPy RTS
   smoother (`rts_oracle.py`) — agreement to **5e-13** (machine precision).
2. **juliacall bridge** (`juliacall_driver.py`): the *same* model driven from
   Python — NumPy array in, array out — reproduced the Julia-only posteriors to
   **5e-13**. The bridge introduces no numerical error.
3. **Shape B not deeply evaluated.** The decision rule in the build plan defaults
   to Shape A unless it proves unworkable. It held on the first real attempt, so
   the default stands. Shape B remains a documented fallback, not a need.

### Consequences / things learned (carry into Phase 1+)

- **Toolchain that worked:** Julia **1.12.6** (via juliaup), **RxInfer v5.4.0**,
  **juliacall 0.9.35**, on **CPython 3.14.5**. The feared Python-3.14
  incompatibility did **not** materialise — 3.14 is fine.
- **juliacall needs PythonCall.jl in the active Julia project.** It's juliacall's
  Julia-side counterpart. The real backend must ensure both PythonCall.jl and
  RxInfer.jl are present — juliacall ships a `juliapkg.json` mechanism for
  declaring Julia deps; cpomdp should ship its own `juliapkg.json` declaring
  RxInfer so `pip install cpomdp[rxinfer]` auto-provisions the Julia side.
- **Startup cost is real but acceptable.** First-ever run paid a one-time
  ~70s (registry update + PythonCall add + precompile). Steady-state startup is
  the `import juliacall` + `using RxInfer` load (tens of seconds, JIT warmup),
  paid once per process — not per inference. Not prohibitive for a library used
  in a session; worth a note in user docs.
- **Inference convention — SMOOTHER, not filter.** Handing RxInfer a whole
  observation sequence at once yields the *smoothed* posterior p(xₜ|y₁..y_T)
  (message passing flows both directions). The Phase-4 correctness oracle must
  therefore be an **RTS smoother** (already written: `rts_oracle.py`), not a bare
  Kalman filter. For an online agent acting in real time we will likely want the
  *filter* instead — drive RxInfer in streaming/one-observation-at-a-time mode.
  Decide this when building `agent.py`.

### The wall (unchanged, restated)

juliacall, PythonCall, RxInfer, and the `@model` DSL all live behind
`backends/base.py`'s Protocol. None of it appears in any public signature, return
type, exception, or docstring. Shape A vs B is an implementation detail the wall
makes swappable.

## On changing the matrices names
To explicitly name the matricices to avoid further confusion and collision within the space.
An example:

LinearGaussianModel(
    dynamics=...,        # A: state → next state
    control=...,         # B: action → state
    observation=...,     # C: state → observation
    process_noise=...,   # Q
    observation_noise=...,# R
)

The letters can survive as aliases/internal attributes and definitely in the docstrings but the primary interface is role-named.
