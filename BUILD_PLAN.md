# cpomdp build plan / progress tracker

Running checklist of what's built and what's next. Authoritative decisions live
in `DECISIONS.md` (ADRs); this file is the roadmap.

Conventions: `[x]` done, `[ ]` open, `[~]` partial.

---

## v0.4.4 — multi-step EFE preliminary window (item A)

Source: `.claude/v0.5_context/multistep_efe_preliminary_plan.md`. Items are
prefixed **M** because the repo's Workstream B (the rollout seam) and the p\*
build plan's item B (the scoring harness) already overload the letter.

**Why this sits in v0.4.4 and not v0.5.** This is the multi-step slice of the
main plan's item A (expected free energy, horizon > 1). It runs *parallel to*
the Certifiable Active Inference programme (the p\*-anchored scoring harness),
and it is **not blocking** on it: nothing here touches a grid, so the warrant
never routes through the reference filter and never meets v0.4.4's certified
discretisation bound. Item A is gate-independent in the same way item E is. It
belongs in the window for its own reason — the standing rule requires H\* (the
horizon at which the epistemic/pragmatic crossover appears) to be **measured
early and registered**. An H\* first seen during v0.5 drafting has no severity;
measuring and registering it here, then requiring v0.5 to reproduce it at H\*
and H\* − 1 with bars, is the only ordering under which D3 is a test at all. So
these objectives are not blocking, but they **must be present in the v0.5
release**.

Current state (v0.4.3, Workstream B): the rollout arithmetic landed and is
byte-locked at H = 1. `policy_efe` scores *a given* constant-action policy at
H > 1 (Tier B, on the agent's own recursion). It does **not** yet license any
claim about the *best* policy (the search family is constant-action), about
*what happens across* the horizon (only sums return), or any *proof* over a
policy set (no enumeration, no certificate). The gap is: the search (M3) and
the certificate (M3's completeness proof).

### The three things called "multi-step EFE"

- [x] **The rollout** — scoring one policy. Done, verified, byte-locked at H = 1
      (Workstream B1–B5).
- [ ] **The search** — choosing over a policy *family*. Today's selector searches
      |A| constant-action policies, not |A|^H sequences. R10 (detour-to-beacon
      then approach-goal) is a *varying* sequence the constant-action family
      **cannot express** — an H-sweep on today's selector produces a null that is
      a search-family artefact and looks exactly like D3's registered falsifier.
- [ ] **The certificate** — the warrant that a finite search was exhaustive.
      Without it "no policy flips sign" is Prover 3a (corroborates), not 3b
      (decides).

### Items

- [x] **M1. Per-step trace from the rollout.** *(Required, cheap, first —
      everything below reads it.)* Return the `lax.scan` `ys` currently discarded:
      per-step `(g, pragmatic, epistemic, μ⁺, Σ⁺, Σ_post, S)` as stacked arrays,
      behind a static flag / a separate `policy_efe_trace`, **not** in the hot
      path (under `vmap` over |A|^H policies, stacking H × n × n covariances is
      the memory driver; the selector stays allocation-free). *Gate:* the trace's
      sums equal the returned scalars under `assert_array_equal` — proof it is the
      same arithmetic, not a second implementation. Unblocks M2, M5, M6.
      **Done:** `policy_efe_trace` → `PolicyEfeTrace` NamedTuple, both driving a
      shared `_rollout_body`; `tests/test_policy_efe_trace.py` (19 tests). Notation
      follows `efe.py` (μ⁺/Σ⁺ predicted, Σ_post contracted); global unification is
      the follow-up checklist below.
- [x] **M2. The Σ(π) policy-dependence witness.** *(Required, cheap.)* Make the
      "planner manipulates the open-loop planning covariances Σ⁻ₖ(π), Σ⁺ₖ(π)"
      claim a check, not a doc sentence: under a fixed sensor two distinct
      policies give **byte-identical** Σ trajectories; under `R(x)` the same two
      separate by more than a declared margin. Theorem 1(i) stated at H > 1,
      checkable today with no grid; prices the cost driver in the same run.
      **Done:** `tests/test_sigma_policy_dependence.py` (4 tests) — fixed sensor
      `assert_array_equal` on Σ⁺/Σ_post/S across two policies (means still differ);
      `R(x)` separates Σ_post and `Q(x)` separates Σ⁺ past a declared 1e-2 margin.
- [x] **M3. Exhaustive enumeration over a declared finite action set.**
      *(Required. Largest item; the one R10 actually needs.)* Enumerate A^H over a
      **finite, declared, versioned** action set — a distinct object from
      `EFESelector`'s continuous grid; the API must not let them be confused.
      Emit the completeness certificate **now** (expected |A|^H, visited count,
      equality asserted — ADR-030 arriving early, ~ten lines). Action set lives in
      the versioned model spec so a later-added action shows in the diff. Print
      the two warrant classes in **different vocabulary** (standing rule 6):
      `PROVED (finite set, |A|^H = N, visited N)` for the enumerated search,
      `CORROBORATED (grid sample of a continuum)` for the action sweep — never
      both as `PASS`. Return the argmin policy **and** the full `G` vector.
      **Done:** new `cpomdp/enumeration.py` (internal seam) — `FiniteActionSet`
      (versioned), `EnumeratedEfeSearch.evaluate` → `(best_policy, G)`,
      `CompletenessCertificate` (asserts `visited == |A|^H`, raises
      `IncompleteEnumerationError`), `SearchWarrant` PROVED; `EFESelector.warrant`
      = CORROBORATED. Supports `p >= 1` and varying sequences; a varying policy
      provably wins on the beacon fixture. `tests/test_enumeration.py` (20 tests).
      ADR-030 + ADR-031 written.
- [x] **M4. Receding-horizon driver + honest cost accounting.** *(Required.)*
      Keep open-loop (apply the whole sequence) and receding-horizon (apply first,
      re-plan) as separate, both available — item E's matched-horizon bracket
      depends on the distinction and R10 must declare its mode before measurement.
      `cost_per_cycle` becomes |A|^H × H step-evals on the enumerated path; it
      must **not** silently keep reporting `n_candidates × horizon`. A cost number
      under-reporting by |A|^(H−1) is worse than none (RFC-001).
      **Done:** `RecedingHorizonSelector` and `OpenLoopSelector` in `enumeration.py`,
      both `ActionSelector`s wrapping `EnumeratedEfeSearch`, both drivable by `Agent`
      via `selector=`. Open-loop commits to the sequence and ignores interim beliefs
      (with `reset()`); receding re-plans each step. Honest cost: both expose
      `cost_per_plan = |A|^H × H` and `replan_interval` (1 / H); `cost_per_cycle` is
      the amortised per-cycle cost (|A|^H × H receding, |A|^H open-loop), never the
      grid's `n_candidates × horizon`. `tests/test_enumerated_drivers.py` (12 tests).
- [ ] **M5. The crossover statistic, defined before it is measured.** *(Required.
      The standing-rule item.)* Write down — in code and in the pre-registration —
      what "epistemic pull" and "pragmatic gradient" mean at H > 1. It must
      (i) reduce **exactly** to Paper 1's H = 1 anchors of 1.72 and 4.49 nats,
      (ii) be computable from M1's trace, (iii) have its sign convention asserted
      in a test, (iv) exist in writing before the sweep runs. The reach/walk
      policies are **named, declared members** of the enumerated set, never
      "whichever two the sweep surfaced". Encode D3's registered falsifiers (no
      crossover at feasible H; a flip not clean at H\* / H\* − 1; non-reproducible
      across seeds) plus a fourth this build makes cheap: **H\* stable under a
      declared refinement of the finite action set** — if inserting intermediate
      actions moves H\*, the crossover is a property of the enumeration, not the
      agent. Pre-register it as a falsifier rather than have a reviewer find it.
- [x] **M6. Numerical hygiene across the scan.** *(Required, cheap; gates the
      honesty of everything above.)* Three `S` re-inversions under `R(x)` against a
      contracting Σ are where twelve orders of magnitude go missing. Print
      `cond(Σ⁺ₖ)` and `cond(Sₖ)` per step from the trace and assert a declared
      ceiling; replace the symmetrise-only PSD guard with symmetrise **plus an
      asserted minimum eigenvalue** (assertion, not a clamp — a clamp launders the
      failure); **assert the `slogdet` sign** (both kernel and NumPy oracle
      currently discard it, so a non-PD matrix yields a meaningless logdet both
      agree on — return it as a diagnostic flag, assert outside the `vmap`);
      assert `jax_enable_x64` in the check suite (a silent float32 downcast is
      invisible at H = 1, fatal at H = 3).
      **Done:** `diagnostics.rollout_conditioning` (per-step cond(Σ⁺)/cond(S)/
      cond(Σ_post), min-eig, PD flag, host-side); `tests/test_rollout_hygiene.py`
      (7 tests) asserts a 1e8 cond ceiling, a 1e-9 min-eig floor (assert, not clamp),
      and x64-on (float64 is the default). **Correction:** the slogdet sign is
      *already* guarded in both the kernel (`_logdet_pd`, cholesky→NaN) and the oracle
      (`epistemic_value`, PD-checked), so that became a regression test, not a fix — no
      rollout-kernel change, byte-identity untouched.
- [ ] **M7. The H-sweep harness and the measured budget.** *(Required. Most
      justifies doing this early.)* One harness running the enumerated planner at
      H = 1…H_max, printing per H: |A|^H, wall time, peak memory, the M5 statistic
      with its bar, F4's three-valued outcome. **H_max is measured, not chosen.**
      If H\* lies beyond the feasible budget that is a D3 falsifier — learning it
      now costs a week; learning it in v0.5 costs the headline result. M5 must be
      *written* before M7 is *run*.
- [ ] **M8. FFG H-step rollout — register, do not build.** *(Deferred, explicit.)*
      `FfgEfeSelector` raises above H = 1 and keeps raising; R10's crossover model
      is a flat corridor, so nothing needs it. Release notes say **unsupported**,
      not untested (standing rule 5).
- [ ] **M9. What must not be built in this window.** Pruning (defer, don't
      half-build). Any continuous varying-sequence search **folded into R10's
      enumerated evidence** — `GradientEfeSelector` is a 3a licence and must never
      contaminate M3's PROVED cells. It is not banned outright: it ships as a
      separate, clearly-labelled corroboration track (see "Continuous-action
      corroboration track" below), walled off from the crossover decision. Any
      documentation of `Q(x)` at H > 1, however cleanly it falls out.

### Gate for the window (soft — all five hold together)

- [ ] Every H = 1 path byte-identical to v0.4.3 (`assert_array_equal`); whole
      existing suite passes **unmodified**.
- [ ] Completeness certificate holds at H = 2 and H = 3 on the declared action set.
- [ ] Σ(π) witness separates fixed-R from `R(x)` at the declared margin, and shows
      byte-identity in fixed-R.
- [ ] Sweep harness prints H_max_feasible with its cost table and condition-number
      ceilings.
- [ ] Crossover statistic defined, reduces exactly to 1.72 / 4.49 at H = 1, and
      **H\* measured and written into the pre-registration** — after which Phase 2's
      job is to reproduce it at H\* / H\* − 1 with bars, not to find it.

### Critical path

`M1 → M3 → M4 → M7` serial; M2 and M6 run alongside; M5 needs M1's trace and M3's
declared policy names and must be *written* before M7 is *run*. Nothing on this
path touches a grid, so nothing waits on the v0.4.4 certified-discretisation gate.

### ADRs

- [x] **ADR-030** (enumeration completeness certificate) — lands here, not v0.6;
      same content, earlier.
- [x] **ADR-031** (new) — the search-family seam: enumerated finite set vs
      continuous grid, their two warrant classes and two output vocabularies, and
      cost attribution at H > 1.
- [ ] **ADR-029** (three-valued check outcomes) — consumed by M7; exists before it
      runs.

### Notation unification (follow-up, non-blocking)

Surfaced while writing the rollout trace: the `μ`/`Σ` predict/update superscript
convention is inconsistent across the codebase. Three variants coexist today:

1. `efe.py` and its neighbours (`observation.py`, `dynamics.py`, `selection.py`,
   `ffg/factors/`, `backends/base.py`): `μ⁺`/`Σ⁺` = the **predicted** (post-dynamics,
   pre-observation) moment; the post-observation covariance is `Σ_post`.
2. `kalman.py`, `ffg/chain.py`, `diagnostics.py`: standard Kalman — `μ⁻`/`Σ⁻` for
   predicted, `post` for updated.
3. `coupling.py`: uses **both** `μ⁻` and `μ⁺`, but there they mean pre-coupling vs
   coupling-resolved *predicted* mean (ADR-019), a distinction orthogonal to
   predict/update.

Not folded into the trace work: it is a cross-file semantics change, not an additive
feature, and variant 3 is a trap — a naive `μ⁺`→`μ⁻` sweep would corrupt ADR-019's
meaning. The trace matches variant 1 (its own file) so it adds no new divergence.

- [ ] Pick one canonical convention (standard Kalman `Σ⁻`/`Σ⁺` is the textbook default
      and what external readers expect) and record it as an ADR — it touches ADR-003's
      epistemic-collapse wording and ADR-019.
- [ ] Give `coupling.py`'s pre-coupling vs coupling-resolved means a **separate**
      disambiguator (not the predict/update superscript), so ADR-019's distinction
      survives the rename.
- [ ] Sweep comments/docstrings across `src/` to the chosen convention; verify it is a
      pure doc/comment change (arithmetic byte-identical, whole suite unmodified,
      `ruff`/`ty` clean).

### Continuous-action corroboration track (v0.5 preliminary, parallel, non-blocking)

A continuous-state agent should also exercise genuinely continuous action spaces, not
only declared finite repertoires. `GradientEfeSelector` — gradient ascent on the
differentiable `policy_efe` over a continuous action box — is that selector. This track
covers the continuous-action regime, kept honest about what it can and cannot claim.

**Warrant: 3a / CORROBORATED only.** Gradient ascent finds a *local* optimum of a
non-convex objective; like the grid it searches a continuum without exhausting it, so it
can never *decide* a universal over the action space. Every result it produces carries
the CORROBORATED label — never PROVED, never a bare PASS (standing rule 6).

**Home: the self-acting regimes, not the p\* scoring harness.** The p\* harness (item B)
runs in exogenous action mode — it *severs* the control loop and drives a common
control sequence into every agent — so action *selection* is not part of the
decomposition (continuous action *values* already are). `GradientEfeSelector` belongs to
the self-acting brackets: a corroborating companion to item A (the crossover) and,
principally, item E (the control bracket), where a self-acting agent under `R(x)` steers
toward low-noise regions and changes its own gap.

**Wall: strictly separated from M3's PROVED evidence.** R10's crossover decision rides
on M3's finite enumeration. A gradient-selected policy may corroborate alongside but
must never enter the decisive cells, or the 3b certificate M3 earns is contaminated back
to 3a (M9).

- [ ] `GradientEfeSelector` — gradient ascent on `policy_efe` over a continuous action
      box (`p >= 1`); returns the optimized sequence and its `G`. Labelled CORROBORATED.
- [ ] Warrant label travels with every continuous-action result (printed and asserted),
      so a corroboration is never read as a certification.
- [ ] Walled off from M3: no gradient result enters R10's enumerated evidence; the two
      families' outputs stay separately labelled in any shared harness.

**Not this track — register, do not build.** *Certified* continuous-action coverage
(deciding "no action in the compact box flips") is Prover 3c — validated numerics, a
certified branch-and-bound with Lipschitz/interval bounds on `policy_efe` over the
box. That is a distinct, larger, later (Paper 2-scale) workstream; `GradientEfeSelector`
does not deliver it and nothing here should imply it does.
