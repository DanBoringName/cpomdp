# Changelog

Everything worth noting lands here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [semantic versioning](https://semver.org). While we're pre-1.0, treat the minor version as the place breaking changes can show up.

## [Unreleased]

### Added

- `examples/crossover_horizon_figure.py` — a showcase of the epistemic/pragmatic crossover on
  an open plane, and the first demo of it on the *flat* Kalman/EFE route rather than the factor
  graph. An augmented state `x = (p, g)` carries a position the control moves and a static
  latent goal it does not. `R(p)` makes the channel that reads `g` sharp only near a beacon off
  the direct path. The animation runs the same world once per planning horizon: at short `H`
  the agent settles where its prior said without ever checking, and past a crossing it detours
  to the beacon, finds out it was wrong, and goes to the real goal. Nothing changes but `H`.
  The margin `ΔG(H) = G(detour, H) − G(direct, H)`, in nats, crosses zero exactly once, because
  the epistemic pull is flat while the pragmatic gradient decays under it. A frozen-R twin re-runs
  the whole sweep with `R` pinned and never crosses, so the behaviour is attributable to the
  state-dependent sensor and the horizon together rather than to the beacon-and-goal geometry.
  Under a fixed sensor the epistemic term is a constant at every horizon (ADR-003). Renders
  the animation plus a three-panel companion. `--check` prints the sweep and asserts what the
  figures claim, gated in `tests/test_example_checks.py`. Its crossing lands at `H = 7`,
  which is the same integer as 0.4.4's registered `H*` and is unrelated to it: different
  model, different backend, whole-state epistemic rather than node-restricted, and no
  search at all. The demo says so in its own output, and neither number should be quoted
  as the other.
- `examples/ffg/cue_maze.py`. The cue-maze task, built for any `n_dims >= 1`. A hidden
  context decides which of two goals pays, the prior points at the wrong one, and a cue
  elsewhere in the arena reads the context sharply while the agent stands on it.
  `axis_action_set` grows as `2·n·|magnitudes| + 1` rather than as a product grid, so two
  dimensions cost exactly what the corridor did. `enumeration_cost` prices a sweep before
  you commit to it, and `best_reachable_noise` catches the failure that motivated it. A cue
  the action lattice cannot land on is one no policy can read, and the null that produces
  is indistinguishable from "information is never worth the detour". It is pure geometry.
- `examples/gallery.py`. The shared machinery the demos were each copying: three palettes,
  the GIF writer, the figure and still writers, the bacillus glyph as a style object, the
  covariance-ellipse geometry, the EFE sweep and decomposition panel, the action grid, and
  the two-route agreement report. Roughly 390 lines of duplication removed. Eight scripts
  touched. Every committed asset re-renders byte-identical to its pre-refactor baseline.
  `tests/test_horizon_dimensions.py` builds its arena from `cue_maze` rather than inlining
  a copy of it.
- `mkdocs_hooks.py`. Expands `--8<--` snippet includes and rewrites the relative links
  inside them to site-local targets, so `examples/README.md` can carry links that work both
  on GitHub and on the published site without hard-coding either. `tests/test_docs_hooks.py`
  gates it, dead links included.

### Changed

- The README's crossover section says what the horizon result is for. It now cites
  arXiv:2607.20306, which establishes that `R(x)` makes the epistemic term non-constant,
  and separates that from the question this repo answers: the horizon at which the term
  starts changing which plan gets picked. The answer given is the one-dimensional `H* = 7`,
  with its warrant stated (exhaustive over the declared `{0, ±1, ±2}`, completeness
  certificate, upper bound because the grid clips the reach), and the plane figure is
  labelled as the readable illustration rather than the proof. `examples/README.md` carries
  the same distinction at length.
- `examples/bacillus_uncertain_food.py` is gated. `check_backend_agreement` — `KalmanBackend`
  against the FFG `ChainBackend` on one scripted input sequence, `atol=1e-7` — now runs in
  `tests/test_example_checks.py`. The README's flagship had no test of any kind before this.
- The `slow` pytest marker is a wall-clock rule now, not a description of one test.
  Anything past `conftest.SLOW_TEST_SECONDS` (20s) carries it and runs on merge-to-main
  and release rather than on pull requests. `conftest.py` measures each run and prints
  any unmarked test that has drifted over, so the rule does not depend on anyone
  remembering it. Two tests cross the line and are newly marked:
  `test_double_integrator_horizon.py::test_h2_choice_matches_brute_force_oracle` (~33s)
  and `test_example_checks.py::test_crossover_horizon_check` (~20s). Pull-request
  coverage is unaffected at 93.8% against the 80% gate.

### Removed

- The `## The journey` section of `examples/README.md`, and with it the gallery entries for
  `bacillus_seeking_food.py`, `bacillus_lqr.py`, `efe_collapse_figure.py` and
  `internal_noise_figure.py`. `efe_collapse_figure.py` and `internal_noise_figure.py` stay,
  keep their `check()` gates, and keep their figures, which `DECISIONS.md` still cites.
  `docs/assets/bacillus.gif` and `docs/assets/bacillus_lqr.gif` had no other referrer and go
  with the section, 4.2 MB of them.
- `examples/bacillus_lqr.py`. Every claim it demonstrated is asserted elsewhere and more
  strongly: the LQR reduction at exact array equality in
  `tests/test_agent.py::test_state_goal_path_is_byte_identical_to_lqr`, and the epistemic
  collapse three times over in `tests/test_efe.py`, `tests/test_theorem.py` and
  `efe_collapse_figure.py`'s gated `check()`. Nothing imported it.
- `examples/bacillus_seeking_food.py`. It had stopped being a demo and become an undeclared
  dependency: `bacillus_uncertain_food.py` imported its beacon falloff, its palette, its
  glyph and its precision field at module scope, so the README's flagship could not run
  without it and no test would have said so. The shared parts are now `gallery.beacon_noise`,
  `gallery.precision_field` and `gallery.SMALL_BACILLUS`; the beacon geometry moved into the
  flagship, which is the only thing that used it. The flagship's GIF re-renders byte-identical
  to the committed one across the move. Its own claims were already asserted in
  `tests/test_agent.py`, `tests/test_efe_selector.py` and `tests/test_efe.py`, whose
  `TestPrecisionControlsBalance` was written for it. ADR-008 and ADR-013 still name the path;
  `DECISIONS.md` is append-only, so those references stand as the historical record.

## [0.4.4] — 2026-08-01

The multi-step slice of expected free energy. Scoring one policy over a horizon was already
here; this release adds searching over a policy family and certifying that the search was
exhaustive, which is what the crossover measurement needed. The headline is `H* = 7`: the
horizon at which the best plan on the coupled-tree cue task stops being a direct reach and
becomes a two-phase sense-then-commit walk. Every horizon is a complete enumeration, so the
flip is decided rather than sampled.

Nothing on the H = 1 path moved. The existing suite passes unmodified and the one-step
arithmetic stays byte-identical to 0.4.3.

### Added

- `cpomdp.enumeration` — the exhaustive search family, deliberately a separate object from
  `EFESelector`'s continuous grid so the two warrants cannot be confused. `FiniteActionSet`
  is versioned, so an action added after results are seen shows up in the diff rather than
  in the prose. `EnumeratedEfeSearch` enumerates `A^H` and returns the argmin policy with
  the full `G` vector. `CompletenessCertificate` asserts expected against visited and raises
  `IncompleteEnumerationError` on a mismatch, which is what earns the search a `PROVED`
  warrant where a grid earns only `CORROBORATED` (ADR-030, ADR-031).
- `EnumeratedEfeSearch.over_backend` scores enumerated policies on an FFG backend instead of
  a flat model. Scoring is injected as a strategy, so the enumeration, the certificate and
  the cost accounting are shared rather than duplicated. A reduce-to-flat oracle gates it:
  on a coupling-free backend it reproduces the flat search exactly (ADR-034).
- `RecedingHorizonSelector` and `OpenLoopSelector`, both `ActionSelector`s wrapping the
  enumerated search. They stay separate because the two modes genuinely differ, and a
  measurement has to declare which one it used. Both report `cost_per_plan = |A|^H · H` and
  a `replan_interval`. Neither reports the grid's `n_candidates × horizon`, which would
  under-count the real work by a factor of `|A|^(H−1)`.
- `policy_efe_trace`, returning a `PolicyEfeTrace`: the per-step `g`, pragmatic, epistemic,
  μ⁺, Σ⁺, Σ_post and S that the rollout scan already computed and then discarded. It sits
  beside the hot path rather than in it, so the selector stays allocation-free. Its sums
  equal the scalars `policy_efe` returns under `assert_array_equal`, which is the proof that
  it is the same arithmetic and not a second implementation.
- `policy_efe_ffg` and `policy_efe_ffg_trace`, the H-step rollout on a coupling graph, with
  the epistemic term aimed at a named node at the coupling-resolved μ⁺. Gated from both
  ends: at H = 1 it is byte-identical to the one-step FFG kernel, and with no coupling it
  matches `policy_efe` (ADR-032).
- `cpomdp.crossover` — `crossover_statistic`, `crossover_horizon` and `CrossoverStatistic`.
  The horizon aggregation is the symmetric between-policy contrast `ΔG = Δc − Δε`. At H = 1
  it collapses to the anchors `Δε = 1.7232`, `Δc = 4.4910`, `ΔG = +2.7678` nats, pinned at a
  tolerance of `1e-4` (ADR-033).
- `diagnostics.rollout_conditioning` — per-step `cond(Σ⁺)`, `cond(S)`, `cond(Σ_post)`, the
  minimum eigenvalue of the posterior covariance, and a positive-definiteness flag, computed
  host-side. Three `S` re-inversions under `R(x)` against a contracting Σ is where orders of
  magnitude go missing quietly. The eigenvalue bar is an assertion and not a clamp, because
  a clamp launders the failure it was meant to catch.
- `examples/ffg/crossover.py` — the crossover measurement, gated. It prints the exhaustive
  argmin per horizon, the mechanism split (disclosed as post-selection, because the pair it
  scores was found by the search), an independent NumPy kernel on the headline number, the
  counterfactual showing the flip is epistemically driven, and the conditioning against its
  registered bars.
- `examples/ffg/crossover_sweep.py` — the constant-action null. A constant walk overshoots
  the cue and never exploits, so the constant family cannot express the two-phase plan at
  all. That null is the reason the exhaustive search is necessary, which is why it ships as
  a check rather than as a footnote.
- A `slow` pytest marker for the exhaustive gate. Pull requests run without it; merges to
  main and releases run with it.

### Changed

- `EFESelector` now reports a `warrant` of `CORROBORATED`. It samples a continuum, so it can
  corroborate a universal over the action space but never decide one. The enumerated
  search's `PROVED` prints in different words for exactly that reason.
- `examples/ffg/crossover.py --check` reports the four registered falsifiers one line each,
  in three-valued outcomes, rather than closing on a single `PASS`. A falsifier that is void
  by construction did not pass a test, and one the gate skips now says so instead of
  borrowing the write-up's answer (ADR-029).

### Deferred and unsupported

Named here so each boundary is a statement rather than an omission. Each has an issue.

- A closed-loop `FfgEfeSelector` above H = 1 stays unsupported, not untested. The rollout it
  would need now exists and the FFG-backed drivers would make it nearly free, but nothing in
  scope uses it. (#52)
- No pruning of the `|A|^H` search. A pruner that cannot prove what it discarded would
  destroy the completeness certificate, which is the only reason the search is worth having,
  so it waits for a design that starts at the certificate. (#53)
- No continuous or gradient search over varying sequences. This one is a deliberate wall:
  gradient ascent on a non-convex objective corroborates and cannot decide, so letting it
  near the enumerated evidence would downgrade the certificate that evidence exists to earn.
  It is registered as a separate, separately-labelled track. (#54)
- `Q(x)` above H = 1 remains undocumented and unclaimed. It may work; nothing says so and
  nothing tests it. (#56)

## [0.4.3] — 2026-07-27

Closes the code-side divergences found by auditing the library against the paper written
about it (`PAPER_DIVERGENCE.md`). The paper's own quoted numbers are unchanged and both
demonstrations' `--check` gates still pass; the paper-side items are tracked separately.

### Added

- `cpomdp.diagnostics` — the conditions a state-dependent sensor has to meet, asked over
  the states an action can actually reach. `probe_model(model, belief, actions)` predicts
  under each candidate action and reports a `SensorReport`: whether `C` has full row rank,
  whether `R` stays positive definite, whether `R` moves at all across the reachable
  means, and whether the epistemic value moves with it. `is_positive_definite`,
  `epistemic_value` and `loewner_order` are exposed alongside for use on their own.
  `probe_model` and `SensorReport` are re-exported at the top level. A declared `R(x)`
  that no action can move now has something that says so.
- `tests/test_theorem.py` — one test per published claim: the pinning result and its
  rank-deficient counterexample, the dual effect on both the posterior covariance and the
  gain, the impossibility of a fixed noise schedule, the scalar monotonicity, the
  zero-innovation realisation, the planning-reduction equivalence, the level-set
  construction where the covariance moves but the epistemic value does not, and the
  worked one-step numbers (gains 2/3 and 2/7, ½ln3 and ½ln(7/5)).
- `tests/test_diagnostics.py` covering the new module.
- `examples/efe_collapse_figure.py --check` now also asserts the dual effect directly —
  the posterior variance and the Kalman gain moving across the action grid — rather than
  only the epistemic scalar that rides on it.

### Fixed

- **The FFG epistemic term had no positive-definiteness guard.** `_efe_step` mapped a
  non-positive-definite `R` to NaN so it would lose the selector's argmin;
  `_state_info_gain`, the term the coupling-graph path actually scores, discarded the
  determinant signs and returned a finite, plausible-but-wrong value that could *win*.
  Both now share one `_logdet_pd` helper, which tests positive definiteness by Cholesky
  rather than by determinant sign — the sign passes any matrix with an even number of
  negative eigenvalues, `diag(-1, -2)` among them.
- **`CouplingGraphBackend.model` handed out a frozen `R`.** For a state-dependent graph
  the flat model carried the noise evaluated once at a representative state, so anything
  filtering with it silently used the wrong noise at every step. The model now carries a
  `_JointObservation` that linearizes each node's `R` at that node's slice of the mean,
  the same point the backend's own filter uses.
- **`Agent` could not see a graph's state-dependence.** Because that flat model declared
  no observation model, a coupled `R(x)` backend read as a fixed sensor and a `StateGoal`
  was accepted onto it, silently selecting actions with the certainty-equivalent LQR
  controller. It now raises, as it always did on the flat path.
- **The theorem's own model class could not be flattened.** A single chain with no
  couplings raised `NotImplementedError`; it now emits a faithful state-dependent flat
  model, verified to reproduce the native filter to ~1e-16. It is a state-dependent
  model, not a fixed one — no fixed one exists.
- `examples/ffg/epistemic_dissociation_figure.py` numbered its first two results the
  opposite way round to the write-up, and printed them out of order. Result 4 asserted
  only that the *best* action B rates below the pragmatic-only agent is cue-ward; it now
  asserts the claim actually made, that *every* such action is.

### Changed

- The `IncompatibleLinearizationError` message gains a closing sentence: the guard reads
  the sensor's *declared* state-dependence, and a declared `R(x)` that ignores the state
  is constant in fact and does flatten. The original two sentences are unchanged, so the
  message is now a strict extension of what came before. Whether `R` really varies over
  reachable means is what `cpomdp.diagnostics.probe_model` is for.
- Trimmed prose in `src/` that had gone stale: references to an `rfcs/` directory that
  does not exist, sensors and selectors described as arriving in a release that has
  shipped, and the unqualified claim that the predicted covariance is action-independent
  — true of a single step, false over a horizon under `R(x)` or `Q(x)`.

## [0.4.2] — 2026-07-18

Archival release accompanying the paper. No library code changed — the public API and
numerics are identical to 0.4.1. The changes are in the example check suite, which the
paper's reproducibility claims quote, and are now gated in CI.

### Added

- A single-chain theorem-instance check on the EFE-collapse demo
  (`examples/efe_collapse_figure.py --check`): the model class of the paper's section
  3.1 — one node, no couplings, additive control, a state-dependent `R(x)` sensor — run
  through the flat `KalmanBackend(CallableSensor)` route, asserting Theorem 1 clauses (i)
  and (iii): the one-step epistemic value varies across the action grid, its frozen-`R`
  twin is flat (the ADR-003 collapse), and `R(μ⁻)` traces a curve. Runs with no plotting
  deps.
- `tests/test_example_checks.py` runs both demos' `check()` gates in CI, so their
  paper-facing assertions fire on every test run rather than only when the script is run
  by hand.

### Changed

- Strengthened Result 4 of the epistemic-dissociation demo
  (`examples/ffg/epistemic_dissociation_figure.py --check`): `_boundary_scan` now also
  returns the pragmatic term, and Result 4 prints and asserts the horizon-1 ordering
  `epistemic pull < pragmatic gradient` — the "reaches without walking" claim, previously
  an unasserted diagnostic number, is now a checked fact.

## [0.4.1] — 2026-07-13

Maintenance release cut for the paper-1 Zenodo archive. No library code changed — the
public API and numerics are identical to 0.4.0.

### Changed

- Reworked the epistemic-dissociation flagship figure
  (`examples/ffg/epistemic_dissociation_figure.py`): renamed the candidate action to `u`
  to stop it clashing with "Agent A", moved the `R(mu+u)` label into a clean gap, and
  added paper-facing assertions that pin Result 2 at the geometry where only B's edge
  over A survives. Regenerated `docs/assets/epistemic_dissociation_boundary.png`.

## [0.4.0] — 2026-07-06

Forney factor-graph (FFG) message passing — the continuous-state generalisation of the
Kalman/EFE path to a branching model the chain cannot draw cleanly (ADR-012, ADR-014).
The chain is the degenerate case and stays validated byte-numerically against the Kalman
filter.

### Added

- FFG message passing in canonical/information form, owned from-scratch in JAX, reachable
  under `cpomdp.ffg.*` (the model-construction symbols are also re-exported at the top
  level — see the public-surface entry below, ADR-021):
  - `CanonicalGaussian` (`cpomdp.ffg.message`) — the `(Λ, h)` message payload; factor
    product is addition, marginalisation a Schur complement, moment form a readout view.
  - Tier-1 linear-Gaussian factor nodes (`cpomdp.ffg.factors.linear_gaussian`):
    `GaussianObservation`, `GaussianTransition`, `GaussianCoupling`.
  - `ChainBackend` (`cpomdp.ffg.chain`) — an `InferenceBackend` over a linear chain,
    numerically identical to `KalmanBackend` (atol 1e-7, the keystone gate), including
    state-dependent `R(x)`/`Q(x)` parity.
  - `CouplingGraph` (`cpomdp.ffg.graph`) — the v0.4 core representation: integer-indexed
    nodes coupled into a rooted tree with a shared node of degree > 2, inferred by a
    hand-authored deepest-first tree-collect schedule (only the root crosses moment form,
    so the inversion cost is per-root, not per-node). `levels()` is deferred (ADR-015).
  - `CouplingGraphBackend` (`cpomdp.backends.coupling`) — the branching FFG as a recursive
    Gaussian filter satisfying `InferenceBackend` (issue #25): driven-relaxation
    composition (each node's own dynamics + its structural parent's drive every slice,
    ADR-017), the exact full-joint carry (ADR-016), `marginal`/`readout` for any node, and
    `to_flat_model` (structural couplings as within-slice pseudo-observations). Validated
    at atol 1e-7 against an independent NumPy joint-precision oracle, `KalmanBackend`, and
    `RxInferBackend` on the flattened model.
  - Carry partition on `CouplingGraphBackend` (ADR-016): a `partition` of the node set
    controls which between-cluster correlations survive the time boundary — `[[all]]`
    (the default) keeps the exact full joint, singletons fully factor it. The carry
    factors the joint *covariance* (not the precision), so every node's marginal stays
    exact within a slice (ADR-017); only the cross-cluster correlation carried forward is
    dropped. A pure `partition_error` reports the severed mass (a covariance magnitude,
    not a rate), and `rollout` profiles it over a whole sequence in one traced `lax.scan`
    pass. A fully-factored (singleton) partition runs cheap two-pass tree belief
    propagation (`CouplingGraph.infer_all`, plus `GaussianCoupling.message_to_child` and
    `CanonicalGaussian.__sub__`) in place of the dense joint solve — O(tree) rather than
    O(n³), matching the dense path to atol 1e-7. The exact endpoint stays byte-identical
    under `[[all]]`.
- State-dependent sensing `R(x)` on the branching backend (issue #27, ADR-019): a
  `CallableGaussianObservation` factor and a two-pass linearisation that reads `R` at the
  coupling-resolved predictive mean μ⁺ — the dual effect (`observation_noise_at`,
  `predicted_belief`).
- `FfgEfeSelector` scores the per-candidate epistemic term over the FFG, so an `Agent` with
  an `ObservationGoal(info_target=…)` seeks information through a branch. `severed_efe_edges`
  / `InadmissiblePartitionError` reject a carry partition that would sever an EFE-relevant
  edge (ADR-018).
- `IncompatibleLinearizationError` (`cpomdp.backends.coupling`): `to_flat_model` raises it
  when a coupled `R(x)` model is asked to flatten. A mean-shifting coupling makes μ⁺ ≠ μ⁻, so
  no fixed linear-Gaussian model reproduces `R(μ⁺)` (ADR-019, ADR-020).
- Public surface (ADR-021): the branching construction symbols (`CouplingGraph`, `Coupling`,
  `CouplingGraphBackend`, `IncompatibleLinearizationError`, the four Gaussian factors) and the
  selector family (`ActionSelector`, `EFESelector`, `FfgEfeSelector`, `LQRSelector`) now sit in
  the top-level `cpomdp` namespace. `FfgEfeSelector` gains `EFESelector`'s `n_candidates` /
  `horizon` / `cost_per_cycle`.
- Examples: `epistemic_dissociation_figure.py`, the v0.4 flagship — two agents on one maze,
  differing only in a state-dependent vs. fixed cue sensor; the `R(x)` agent resolves the
  hidden context through a branch and crosses to the reward, the fixed-sensor agent collapses
  to LQR, and the `R(x)` model can't be flattened (`IncompatibleLinearizationError`).
  `bacillus_uncertain_food.py`, the instrumental-epistemics demo — the beacon resolves an
  explicit food *latent* rather than the agent's own position (ADR-013), on both
  `KalmanBackend` and `ChainBackend`. `coupling_graph_figure.py`, the difference demo — a
  branching tree resolved natively vs. the hand-flattened joint precision.
  `chemotaxis_figure.py`, the same on a real branching network (the E. coli chemotaxis
  pathway's shape, native FFG vs flattened Kalman) — illustrative only, no biophysics
  (ADR-020).

### Validation

- RxInfer oracle (behind the `rxinfer` marker) extended to the branching tree, alongside
  the existing chain checks; jit/grad/vmap smoke tests gate every new public inference
  entry point.

## [0.3.0] — 2026-06-22

Epistemic action selection. v0.2 could perceive and pursue a goal; v0.3 lets the agent *seek information* — it minimises Expected Free Energy `G = pragmatic − epistemic`, so it will detour to where its sensor is sharp, localise, then act. Under a fixed sensor this collapses exactly to the v0.2 LQR behaviour (ADR-003), so that path is unchanged.

### Added

- Expected Free Energy action selection: `expected_free_energy` (the one-step kernel) and `EFESelector` (a front-loaded candidate grid, `argmin G`). Observation-space cross-entropy pragmatic minus state info-gain epistemic (ADR-005).
- State-dependent sensing and dynamics: `CallableSensor` for `R(x)` and `CallableProcessNoise` for `Q(x)`, honoured in *both* the planner and the Kalman filter (ADR-006, ADR-008). The fixed-matrix path stays byte-identical.
- Typed objectives: `StateGoal` (state-space → LQR) and `ObservationGoal` (observation-space → EFE). The `Agent` dispatches on the objective's type, so an illegal pairing is unrepresentable rather than a runtime check (ADR-007).
- H-step horizon: `EFESelector(horizon=H)` scores constant-action policies over an H-step rollout (default `H=1`), making delayed consequences visible — e.g. acting on velocity to steer an observed position (ADR-009).
- `ModelStructure`: optional, declarable factor / Markov-blanket / sensory-channel metadata on a model, with inspection and an opt-in, experimental `validate()` that checks the declaration against the matrix sparsity (ADR-010).
- Public building blocks for the above: `ObservationModel`, `FixedSensor`, `DynamicsNoise`.

### Changed

- **Breaking (pre-1.0):** `Agent` takes a typed objective instead of `goal=`/`goal_precision=` keywords. `Agent(model, goal=[1.0, 0.0])` becomes `Agent(model, StateGoal([1.0, 0.0]))`; observation-seeking uses `ObservationGoal(...)` (ADR-007).

## [0.2.0] — 2026-06-16

The array backend moves from NumPy to JAX (ADR-004). v0.1 was a proof of concept; this is the groundwork for the autodiff and batching v0.3 is aiming at.

### Changed

- The core runs on `jax.numpy`. `Belief` and `LinearGaussianModel` now hold `jax.Array`s, and the Kalman filter and LQR are pure `jnp`. If you were reaching past the public API and expecting `numpy.ndarray` off `belief.mean`, you'll get a `jax.Array` now — both still hand off to NumPy, so most code won't notice.
- Importing `cpomdp` switches JAX into float64 mode (`jax_enable_x64`) process-wide. The library is validated to 1e-9 against the RxInfer oracle and JAX defaults to float32, so this keeps the numbers right — but it does change float behaviour for any other JAX code in the same process.

### Added

- `Belief` and `LinearGaussianModel` are registered JAX pytrees, so they flow through `jit`, `vmap`, and `grad` as data.
- The Kalman step is split into pure, `jit`-compiled kernels — one filter step now `vmap`s over a batch of beliefs.

### Dependencies

- Added `jax` and `jaxtyping`. NumPy stays: JAX pulls it in anyway, and the RxInfer backend still hands real NumPy arrays across the Julia bridge.

## [0.1.1] — 2026-06-15

A metadata-only re-release, functionally identical to 0.1.0. The 0.1.0 release has
been removed from PyPI, so use 0.1.1.

### Changed

- Trimmed the author entry in the package metadata.
- README: the DECISIONS.md link is now an absolute URL, so it resolves on PyPI.

## [0.1.0] — 2026-06-15

The first cut. Linear-Gaussian active inference, end to end: perceive with a Kalman filter, act with LQR, all behind a pymdp-style `Agent`.

### Added

- `Agent`, the stateful façade you actually drive. `infer_states` to perceive, `sample_action` to act, the same loop pymdp users know. It remembers the last action it took, so you don't have to thread that back in by hand. Build it without a goal and it's a pure tracker that perceives but won't act.
- `LinearGaussianModel`, the generative model. Matrices are named for their role (`dynamics`, `control`, `sensor_model`, `dynamics_noise`, `sensor_noise`), with the control-theory letters (`A`/`B`/`C`/`Q`/`R`) kept as aliases for when you're reading the maths.
- `Belief`, an immutable Gaussian belief: a mean and a covariance, validated on the way in.
- `KalmanBackend`, exact Kalman filtering. Has an optional steady-state mode that solves the gain once up front and reuses it.
- `LQRController`, steady-state LQR action selection. For a linear-Gaussian sensor this *is* the expected-free-energy-optimal action rather than a stand-in for it (the why is in DECISIONS.md, ADR-003).
- `RxInferBackend`, an optional [RxInfer](https://github.com/ReactiveBayes/RxInfer.jl) (Julia) backend. It re-derives the same filtering results through completely separate machinery and exists as a correctness oracle for the native path. Lives behind the `rxinfer` extra so the core install stays Julia-free.
- `InferenceBackend`, the protocol the backends satisfy, so you can drop in your own engine.

This is pre-alpha. The API works and is tested against the RxInfer oracle, but it can still move before 1.0.
