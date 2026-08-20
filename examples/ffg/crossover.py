"""Where the crossover lives: the horizon at which the best plan stops reaching.

At a short horizon the best plan on the coupled-tree cue task is a direct reach for the
goal. At a long enough horizon it becomes a two-phase walk: drive to the cue, sense the
hidden context, reverse, and commit. This demo locates the horizon H* where the switch
happens and shows why.

Two measurements, and the distinction between them is the whole point.

- **The decisive one is selection-free.** For each horizon the *exhaustive* argmin over
  every ``A^H`` action sequence is computed by ``EnumeratedEfeSearch.over_backend``. It
  is prior-ward (a reach) through H = 6 and cue-ward (a two-phase walk) at H = 7, and
  each horizon is an exact finite enumeration, so no policy is chosen with knowledge of
  the outcome. This is what carries the result.

- **The mechanism curve is exposition, and it is post-selection.** The specific walk the
  search picks out at H = 7, ``[+1,-2,-2,0,...]``, is scored back across the horizon
  against the coast-to-stop reach ``[-2,-1,0,...]`` to read out the epistemic and
  pragmatic split. That pair was *found by* the search, so its crossing is shown to
  explain the flip, never to establish it.

The mechanism is the surprising part. The epistemic pull is flat (~1.7 nats, the
one-step value); it does not accumulate. What moves is the pragmatic gradient: after
sensing at the cue the contracted context belief lowers the *commit* channel's expected
ambiguity by ~0.67 nats per step, and that saving accumulates until it overtakes the
one-time detour cost. So the reach's advantage decays with the horizon and the walk wins
at H = 7 -- the gradient decays below the constant pull, not the pull past the gradient.

``--check`` asserts the flip, the mechanism split, an independent NumPy oracle on the
headline number, and the kernel's numerical inertness, with no plotting deps::

    uv run --no-sync python examples/ffg/crossover.py --check
    uv run --no-sync python examples/ffg/crossover.py
"""

from __future__ import annotations

import itertools
import sys
from typing import NamedTuple

import epistemic_dissociation_figure as demo
import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.crossover import crossover_statistic
from cpomdp.diagnostics import logdet_pd, rollout_conditioning
from cpomdp.efe import policy_efe_ffg, policy_efe_ffg_trace
from cpomdp.enumeration import (
    CompletenessCertificate,
    EnumeratedEfeSearch,
    FiniteActionSet,
)
from cpomdp.selection import Preference
from cpomdp.types import Belief
from cpomdp.warrant import (
    CheckReport,
    Outcome,
    Provenance,
    Tier,
    Warrant,
    check_summary,
)

# The rollout-hygiene bars this shares with tests/test_rollout_hygiene.py: a floor under
# min-eigenvalue(Σ_post) and a ceiling on cond(Σ⁺), cond(S), cond(Σ_post).
MIN_EIG_FLOOR = 1e-9
COND_CEILING = 1e8

# The registered action set the anchors were fixed against. It clips the reach at the
# grid edge -2, so it needs two steps to reach the goal at -3. A set containing -3,
# which reaches the goal in one step from the start, flips one horizon sooner (see the
# +edge measurement below). H* on this set is therefore an upper bound, and the honest
# headline number. Wider sets are unmeasured: -3 is the one-step reach from the start,
# not an established optimum, since the walk arrives at the cue at +1.
V1 = [-2.0, -1.0, 0.0, 1.0, 2.0]  # the registered action set (clips the reach at -2)
V1_EDGE = [-3.0, *V1]  # + the one-step reach -3; flips one horizon sooner
ACTION_SET = FiniteActionSet([[a] for a in V1], version="v1")
EDGE_SET = FiniteActionSet([[a] for a in V1_EDGE], version="v1-edge")
FLIP_H = 7  # the crossover horizon on ACTION_SET
MAX_H = 7  # the exhaustive sweep budget (feasibility is |A|^H * H, printed below)
# Declared feasibility bound: enumeration is feasible to H_MAX (5^9 * 9 = 17.6M scored
# steps, measured); the crossover at H=7 sits well inside it and is cue-ward through 9.
H_MAX = 9
EPS_PLATEAU = 0.2  # the epistemic pull varies by less than this: flat, not growing


def _setup():
    """``(backend, belief, preference, info_block)`` for the coupled-tree cue task."""
    backend = demo.build_backend(epistemic_alive=True, cue_x=demo.CUE_DETOUR_X)
    belief = demo.start_belief()
    preference = Preference(
        goal=[0.0, 0.0],
        precision=[[demo.GOAL_PRECISION, 0.0], [0.0, demo.INFO_PRECISION]],
    )
    info_block = tuple(backend.block(demo.CONTEXT))
    return backend, belief, preference, info_block


def _cue_ward(policy) -> bool:
    """A plan visits the cue if its position (cumulative action from x=0) reaches it."""
    return bool(np.cumsum(np.asarray(policy).ravel()).max() >= 0.5)


def _walk(horizon: int):
    """The two-phase walk the search selects at H*: cue, sense, reverse, then park."""
    actions = [1.0, -2.0, -2.0][: min(horizon, 3)] + [0.0] * max(0, horizon - 3)
    return jnp.asarray(actions).reshape(-1, 1)


def _reach(horizon: int):
    """The coast-to-stop reach: decelerate onto the goal at x=-3 and hold."""
    actions = [-2.0, -1.0][: min(horizon, 2)] + [0.0] * max(0, horizon - 2)
    return jnp.asarray(actions).reshape(-1, 1)


# --- 1. the decisive, selection-free measurement -----------------------------------
def argmin_at(horizon: int, action_set=ACTION_SET):
    """The exhaustive argmin at a horizon: ``(n_policies, Gmin, argmin, cue_ward)``."""
    backend, belief, preference, info_block = _setup()
    search = EnumeratedEfeSearch.over_backend(
        backend, action_set, info_block=info_block, horizon=horizon
    )
    result = search.evaluate(belief, preference)
    policy = np.asarray(result.best_policy).ravel()
    return search.n_policies, float(result.g.min()), policy, _cue_ward(policy)


def exhaustive_flip(action_set=ACTION_SET, max_horizon=MAX_H):
    """Per-horizon exhaustive argmin ``(H, n_policies, Gmin, argmin, cue_ward)``."""
    return [(h, *argmin_at(h, action_set)) for h in range(1, max_horizon + 1)]


class FlipMeasurement(NamedTuple):
    """One horizon's exhaustive enumeration, as the falsifiers read it.

    Separating the measurement from the report is what lets the reporting logic be
    exercised on a refuting result. Nothing in a live run can produce one, and a branch
    no run can reach is a branch no run has tested.

    Attributes:
        horizon: the horizon enumerated.
        certificate: that enumeration's completeness certificate.
        delta_g: ``G(walk) - G(reach)`` at this horizon. Negative means the walk wins.
        bound: the error ``COND_CEILING`` allows on that difference.
        cue_ward: whether the *exhaustive* argmin visits the cue. The direction the
            falsifier asks about, from the enumeration rather than from the pair.
    """

    horizon: int
    certificate: CompletenessCertificate
    delta_g: float
    bound: float
    cue_ward: bool

    @property
    def separated(self) -> bool:
        """Whether the margin clears the error bound, so the ordering is decidable."""
        return abs(self.delta_g) > self.bound


def measure_flip(horizon: int) -> FlipMeasurement:
    """Enumerate at ``horizon`` and read off everything the falsifiers need."""
    n_policies, _, _, cue_ward = argmin_at(horizon)
    backend, _, _, info_block = _setup()
    certificate = EnumeratedEfeSearch.over_backend(
        backend, ACTION_SET, info_block=info_block, horizon=horizon
    ).certificate
    walk, reach = _numpy_score(_walk(horizon)), _numpy_score(_reach(horizon))
    assert certificate.expected == n_policies  # the two routes enumerate the same set
    return FlipMeasurement(
        horizon=horizon,
        certificate=certificate,
        delta_g=walk - reach,
        bound=flip_margin_error(walk, reach),
        cue_ward=cue_ward,
    )


# --- 2. the mechanism split (post-selection exposition) ----------------------------
def mechanism_curve(max_horizon=MAX_H):
    """Per-horizon ``(H, Δε, Δc, ΔG)`` for the selected walk against the coast reach."""
    backend, belief, preference, info_block = _setup()
    rows = []
    for horizon in range(1, max_horizon + 1):
        stat = crossover_statistic(
            backend,
            belief,
            _walk(horizon),
            _reach(horizon),
            preference,
            info_block=info_block,
        )
        rows.append(
            (
                horizon,
                float(stat.delta_epsilon),
                float(stat.delta_c),
                float(stat.delta_g),
            )
        )
    return rows


# --- 3. conditioning + an independent NumPy oracle on the headline number ----------
def _numpy_score(policy) -> float:
    """Pure-NumPy predict-then-contract score.

    Reuses the backend's belief propagation (separately oracle-tested) but recomputes
    the EFE kernel -- pragmatic mean, ambiguity, node-restricted info gain, contraction
    -- independently of ``_ffg_efe_step``. So agreement is a cross-implementation check
    of the scoring kernel, not of shared plumbing.
    """
    backend, belief, preference, info_block = _setup()
    observation_matrix = np.asarray(backend.observation_model[0])  # C
    precision = np.asarray(preference.precision)  # Lambda
    goal = np.asarray(preference.goal)
    idx = np.array(info_block)

    carry = belief
    total = 0.0
    for action in np.asarray(policy).ravel():
        predicted = backend.predicted_belief(carry, jnp.array([float(action)]))
        mean = np.asarray(predicted.mean)  # mu+
        cov = np.asarray(predicted.cov)  # Sigma+
        noise = np.asarray(backend.observation_noise_at(predicted.mean))  # R(mu+)
        obs = observation_matrix @ mean  # o+ = C mu+
        innovation = observation_matrix @ cov @ observation_matrix.T + noise  # S
        pragmatic = 0.5 * (obs - goal) @ precision @ (obs - goal) + 0.5 * np.trace(
            precision @ innovation
        )
        post_info = (
            np.linalg.inv(cov)
            + observation_matrix.T @ np.linalg.inv(noise) @ observation_matrix
        )
        cov_post_info = np.linalg.inv(post_info)  # info-form posterior, the epistemic
        # `logdet_pd` is the host guard: slogdet behind a Cholesky, so a block with an
        # even number of negative eigenvalues returns NaN rather than a plausible
        # number. The kernel's `_logdet_pd` is a separate implementation that reads its
        # answer off a Cholesky factor under `jit`. They stay distinct so the agreement
        # assertion in `check()` compares two routines rather than one.
        epistemic = 0.5 * (
            logdet_pd(cov[np.ix_(idx, idx)])
            - logdet_pd(cov_post_info[np.ix_(idx, idx)])
        )
        total += pragmatic - epistemic
        # Kalman-form contraction for the carry; the mean stays predict-only.
        cov_post = (
            cov
            - cov
            @ observation_matrix.T
            @ np.linalg.inv(innovation)
            @ observation_matrix
            @ cov
        )
        carry = Belief(mean=predicted.mean, cov=jnp.asarray(cov_post))
    return total


def flip_margin_error(g_one: float, g_two: float) -> float:
    """The error the declared conditioning ceiling allows on ``g_one - g_two``.

    A float64 solve at condition number ``k`` returns a result whose relative error is
    about ``k * eps``. The rollout-hygiene discipline declares ``cond <= COND_CEILING``
    and ``tests/test_rollout_hygiene.py`` gates every ``Σ⁺``, ``S`` and ``Σ_post``
    against it, so each score is good to ``|G| * COND_CEILING * eps`` and their
    difference to twice that.

    This is the bar the flip is measured against. It is derived from a ceiling that was
    already declared and already enforced, not chosen to fit the measured margin. The
    measured conditioning is far below the ceiling (``max cond = 1003`` on the H* walk),
    so the true error is some five orders smaller again and this bound is loose in the
    safe direction.

    Args:
        g_one: one policy's summed EFE.
        g_two: the other's.

    Returns:
        A bound on the numerical error in their difference, in nats.
    """
    eps = float(np.finfo(float).eps)
    return 2.0 * max(abs(g_one), abs(g_two)) * COND_CEILING * eps


def conditioning(policy):
    """The registered rollout conditioning for a plan.

    Delegates to ``cpomdp.diagnostics.rollout_conditioning``, so the reported quantities
    -- ``cond(Σ⁺)``, ``cond(S)``, ``cond(Σ_post)``, ``min_eig(Σ_post)``, and the all-PD
    flag -- and their bars are exactly the ones ``tests/test_rollout_hygiene.py`` gates
    on. The headline margin is small in relative terms, so this is where it is shown
    benign. ``min_eig(Σ_post)`` staying above the floor means neither Cholesky guard
    fires, the kernel's ``_logdet_pd`` or this module's ``logdet_pd``, and the epistemic
    never goes NaN.
    """
    backend, belief, preference, info_block = _setup()
    trace = policy_efe_ffg_trace(
        backend,
        belief,
        jnp.asarray(policy).reshape(-1, 1),
        preference,
        info_block=info_block,
    )
    return rollout_conditioning(trace)


def epistemic_counterfactual(horizon=FLIP_H, actions=None):
    """Is the flip epistemic? Compare the argmin WITH the epistemic term to WITHOUT it.

    Enumerates at ``horizon`` and returns ``(g_cue_ward, pragmatic_cue_ward,
    all_g_finite)``. When the full-``G`` argmin is cue-ward while the pragmatic-only
    argmin is still prior-ward, the epistemic term is what causes the flip here -- the
    crossover is epistemic, not a pragmatic phenomenon wearing an epistemic label.
    ``all_g_finite`` doubles as the guard check: no enumerated policy scores NaN, so the
    NaN-safe argmin is never exercised here.
    """
    actions = V1 if actions is None else actions
    backend, belief, preference, info_block = _setup()
    combos = np.array(list(itertools.product(actions, repeat=horizon)), dtype=float)

    @jax.jit
    def batch(pols):
        def one(pol):
            g, parts = policy_efe_ffg(
                backend, belief, pol, preference, info_block=info_block
            )
            return g, parts["pragmatic"]

        return jax.vmap(one)(pols)

    gs, prags = [], []
    for start in range(0, combos.shape[0], 16384):
        chunk = jnp.asarray(combos[start : start + 16384]).reshape(-1, horizon, 1)
        g, prag = batch(chunk)
        gs.append(np.asarray(g))
        prags.append(np.asarray(prag))
    g_all = np.concatenate(gs)
    prag_all = np.concatenate(prags)

    def cue_ward(scores):  # NaN-safe argmin, matching EnumeratedEfeSearch
        best = int(np.argmin(np.where(np.isnan(scores), np.inf, scores)))
        return bool(np.cumsum(combos[best]).max() >= 0.5)

    return cue_ward(g_all), cue_ward(prag_all), bool(np.all(np.isfinite(g_all)))


# --- the registered falsifiers, reported by both the bare run and the gate ----------
def _bar(*measurements: FlipMeasurement) -> str:
    """The margins and the bound each was read against, one clause per horizon."""
    return "; ".join(
        f"|ΔG({m.horizon})| = {abs(m.delta_g):.4f} vs bound {m.bound:.2e} "
        f"({abs(m.delta_g) / m.bound:.0e}x)"
        for m in measurements
    )


def falsifiers(
    at_prior: FlipMeasurement | None = None,
    at_flip: FlipMeasurement | None = None,
) -> tuple[CheckReport, ...]:
    """The four D3 falsifiers registered for this crossover, one report each.

    A falsifier does not pass. ``NOT TRIGGERED`` is "it ran and the condition did not
    obtain", so the claim survives it. ``NOT APPLICABLE`` is void by construction, so
    it is evidence for nothing and is not a survivor. ``NOT RUN HERE`` was measured
    elsewhere. Neither of the last two ran, so neither carries a warrant: the prover
    cell is ``—``, because attributing one would claim evidence that was never produced.

    Rows 1 and 2 read on both axes at once. On the prover axis they are ``PROVED``,
    resting on exhaustive enumeration and carrying its certificates. On the tier axis
    they are ``B``, the margin read against the error ``COND_CEILING`` allows on a
    difference of two scores.

    Every outcome and every detail here is computed from the measurements. The direction
    comes from ``cue_ward``, off the exhaustive argmin, so a reversed result reports
    ``FIRED`` instead of printing the old sentence beside a cleared magnitude test. A
    margin inside its bound reports ``NOT RESOLVED``, because the ordering would then be
    genuinely undetermined and the honest report of a tie is a tie.

    Row 2 quantifies over two horizons, so it takes both measurements and both
    certificates. Reading it off H* alone would let it survive on the H* margin while
    ``ΔG(H*-1)`` sat inside its own bound.

    Args:
        at_prior: the measurement at ``H*-1``. Enumerated live when omitted.
        at_flip: the measurement at ``H*``. Enumerated live when omitted.
    """
    at_prior = measure_flip(FLIP_H - 1) if at_prior is None else at_prior
    at_flip = measure_flip(FLIP_H) if at_flip is None else at_flip

    def crossover_exists() -> Outcome:
        """Fires when the exhaustive argmin is not cue-ward at H*."""
        if not at_flip.separated:
            return Outcome.NOT_RESOLVED
        return Outcome.NOT_TRIGGERED if at_flip.cue_ward else Outcome.FIRED

    def flip_is_clean() -> Outcome:
        """Fires unless the argmin is prior-ward at H*-1 and cue-ward at H*."""
        if not (at_prior.separated and at_flip.separated):
            return Outcome.NOT_RESOLVED
        clean = at_flip.cue_ward and not at_prior.cue_ward
        return Outcome.NOT_TRIGGERED if clean else Outcome.FIRED

    def where(measurement: FlipMeasurement) -> str:
        return "cue-ward" if measurement.cue_ward else "prior-ward"

    # Travels with the number everywhere it is quoted: the declared set clips the reach
    # at -2 while -3 reaches the goal in one step, so 7 is an upper bound on H*, and a
    # set containing -3 flips at 6. The error bound certifies the arithmetic, which was
    # never the exposure here.
    upper = (
        f"H* = {FLIP_H} is an upper bound "
        "(set clips the reach at -2, one-step reach -3)"
    )
    # One commit derived the separation bar from the declared conditioning ceiling and
    # reported these falsifiers against it. Registering and measuring together is what
    # happened, so the two refs are one and the render says the history orders nothing.
    registration = Provenance(
        registered_at="efc43e2",
        measured_at="efc43e2",
        registered="the flip separation bar, derived from the conditioning ceiling",
    )
    return (
        CheckReport(
            name="1. no crossover at feasible H",
            warrant=Warrant.PROVED,
            outcome=crossover_exists(),
            tier=Tier.BOUNDED,
            detail=(
                f"argmin is {where(at_flip)} at H = {at_flip.horizon}, inside "
                f"H_MAX = {H_MAX}. {upper}. {_bar(at_flip)}"
            ),
            evidence=(at_flip.certificate,),
            provenance=(registration,),
        ),
        CheckReport(
            name="2. flip not clean at H*/H*-1",
            warrant=Warrant.PROVED,
            outcome=flip_is_clean(),
            tier=Tier.BOUNDED,
            detail=(
                f"argmin is {where(at_prior)} at H = {at_prior.horizon} and "
                f"{where(at_flip)} at H = {at_flip.horizon}. {upper}. "
                f"{_bar(at_prior, at_flip)}"
            ),
            evidence=(at_prior.certificate, at_flip.certificate),
            provenance=(registration,),
        ),
        CheckReport(
            name="3. not reproducible across seeds",
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.COMPUTED,
            detail="void by construction: no observation draw to vary across seeds",
        ),
        CheckReport(
            name="4. H* unstable under refinement",
            warrant=None,
            outcome=Outcome.NOT_RUN_HERE,
            tier=Tier.COMPUTED,
            detail=(
                f"step-0.5 refinement costs {9**7 * 7} steps. Recorded in the "
                "write-up, and the live exposure on this number"
            ),
        ),
    )


def _print_falsifiers(reports: tuple[CheckReport, ...]) -> None:
    """Print the registered falsifiers on both axes, then the run's counts.

    Prover and tier print as separate columns because they answer separate questions.
    A row can be decided and have nothing stated behind its margin, or measured against
    a tight bar and only sampled. Collapsing them into one verdict loses whichever half
    the reader needed.

    Takes the reports rather than building them, so the caller's enumerations are paid
    for once.
    """
    print(f"\nRegistered D3 falsifiers ({len(reports)} registered):")
    print(f"   {'':<33} {'outcome':<15} {'prover':<13} {'tier':<5} why")
    for report in reports:
        prover = report.warrant.value if report.warrant else "—"
        print(
            f"   {report.name:<33} {report.outcome.value:<15} "
            f"{prover:<13} {report.tier.value:<5} {report.detail}"
        )
    print(f"\n{check_summary(reports)}")
    print("   full accounting: research/r10_open_loop_crossover.md, chapter 7")


# --- the four measurement tables ----------------------------------------------------
def _print_tables() -> None:
    flip = exhaustive_flip()
    mech = mechanism_curve()
    edge = exhaustive_flip(action_set=EDGE_SET, max_horizon=FLIP_H - 1)

    print("1. Exhaustive argmin per horizon (selection-free, exact enumeration):")
    print(
        f"   {'H':>2} {'|A|^H':>7} {'cost |A|^H*H':>12} {'Gmin':>10} "
        f"{'argmin':<22} plan"
    )
    for horizon, n, gmin, policy, cue in flip:
        kind = "walk (cue-ward)" if cue else "reach (prior-ward)"
        print(
            f"   {horizon:>2} {n:>7} {n * horizon:>12} {gmin:>10.4f} "
            f"{policy!s:<22} {kind}"
        )
    h_star = next((h for h, _, _, _, cue in flip if cue), None)
    print(f"   -> the argmin flips reach -> walk at H* = {h_star}\n")

    print("2. Mechanism (post-selection: the walk the search picked, scored back):")
    print(f"   {'H':>2} {'Δε (pull)':>11} {'Δc (grad)':>11} {'ΔG':>10}   reading")
    for horizon, de, dc, dg in mech:
        print(
            f"   {horizon:>2} {de:>11.4f} {dc:>11.4f} {dg:>10.4f}   "
            f"{'walk wins' if dg < 0 else 'reach wins'}"
        )
    pulls = [de for _, de, _, _ in mech]
    print(
        f"   -> pull flat (range {max(pulls) - min(pulls):.3f}); gradient decays "
        f"{mech[0][2]:.2f} -> {mech[-1][2]:.2f}, crossing it at H* = {FLIP_H}\n"
    )

    print(
        f"3. Registered rollout conditioning along the H={FLIP_H} walk "
        f"(bars: min_eig(Σ_post) > {MIN_EIG_FLOOR:.0e}, cond < {COND_CEILING:.0e}):"
    )
    rc = conditioning(_walk(FLIP_H))
    print(
        f"   {'k':>2} {'cond(Σ+)':>10} {'cond(S)':>9} {'cond(Σ_post)':>13} "
        f"{'minEig(Σ_post)':>15}"
    )
    for k in range(FLIP_H):
        print(
            f"   {k:>2} {rc.cond_sigma_pred[k]:>10.1f} {rc.cond_s[k]:>9.1f} "
            f"{rc.cond_sigma_post[k]:>13.1f} {rc.min_eig_sigma_post[k]:>15.3e}"
        )
    max_cond = max(rc.cond_sigma_pred.max(), rc.cond_s.max(), rc.cond_sigma_post.max())
    min_eig = float(rc.min_eig_sigma_post.min())
    print(
        f"   all PD = {rc.all_positive_definite}; min eig(Σ_post) = {min_eig:.3e} "
        f"({min_eig / MIN_EIG_FLOOR:.0e}x the floor); max cond = {max_cond:.0f}"
    )
    backend, belief, preference, info_block = _setup()
    walk_ship = float(
        policy_efe_ffg(
            backend, belief, _walk(FLIP_H), preference, info_block=info_block
        )[0]
    )
    walk_np, reach_np = _numpy_score(_walk(FLIP_H)), _numpy_score(_reach(FLIP_H))
    print(
        f"   NumPy oracle: ΔG = {walk_np - reach_np:+.6f} "
        f"(|ΔG|/|G| = {abs(walk_np - reach_np) / reach_np:.1e}); "
        f"|G_walk shipped-numpy| = {abs(walk_ship - walk_np):.1e} < atol 1e-9"
    )
    print("   (ln det differs: kernel Cholesky, oracle slogdet, both PD-guarded)")

    g_cue, prag_cue, finite = epistemic_counterfactual(FLIP_H)
    print(
        f"\n4. Epistemic counterfactual at H={FLIP_H} (all {5**FLIP_H} G finite = "
        f"{finite}):"
    )
    print(f"   argmin cue-ward -- with epistemic: {g_cue}; without: {prag_cue}")
    print("   -> the flat ~1.7-nat epistemic pull is what flips it at H=7; zero the")
    print("      epistemic term and the reach still wins, crossing only at H~10.")

    edge_star = next((h for h, _, _, _, cue in edge if cue), None)
    refine_cost = 9**7 * 7
    hmax_cost = 5**H_MAX * H_MAX
    print("\n5. Action-set dependence and feasibility:")
    print(f"   one-step reach -3 -> H* = {edge_star}; the registered set clips the")
    print(f"   reach to -2, so H* = {FLIP_H} is an upper bound.")
    print(f"   recorded, not re-run here (cost {refine_cost} steps): a step-0.5")
    print("   refinement of the same range left the H=6 and H=7 argmins unchanged,")
    print("   no intermediate action scoring lower G (a subset check; step-0.25")
    print("   dropped on cost).")
    print("   analytic bound: relief <= 0.77/step vs 2.77 detour -> H* >= 6.")
    print(f"   feasibility: declared to H_MAX = {H_MAX} (cost {hmax_cost} scored")
    print("   steps); the argmin is cue-ward at H = 7, 8, 9.")
    _print_falsifiers(falsifiers())


# --- the gate ----------------------------------------------------------------------
def check() -> None:
    """Assert the flip, the mechanism split, the oracle, and the kernel's inertness.

    The exhaustive enumerations are scoped to the flip boundary and H* (the horizons the
    assertions bear on), so the gate enumerates ~230k policies, not the full sweep.
    """
    # 1. Decisive: the exhaustive argmin is a reach at H*-1 and a cue-ward walk at H*.
    #    Measured once here and handed to `falsifiers()` below, so the two enumerations
    #    are paid for once rather than per consumer.
    at_prior, at_flip = measure_flip(FLIP_H - 1), measure_flip(FLIP_H)
    assert at_prior.cue_ward is False, "argmin should still be a reach at H*-1"
    assert at_flip.cue_ward is True, "argmin should be a cue-ward walk at H*"

    # 2. Mechanism: pull flat, gradient decays below it, clean sign change at H*/H*-1.
    mech = mechanism_curve()
    pulls = [de for _, de, _, _ in mech]
    assert max(pulls) - min(pulls) < EPS_PLATEAU  # epistemic does not accumulate
    dg = {h: g for h, _, _, g in mech}
    assert dg[FLIP_H - 1] > 0  # reach still wins at H*-1
    assert dg[FLIP_H] < 0  # walk wins at H* -- a clean single sign change
    # No registered falsifier fired. A margin inside its bound reports NOT RESOLVED and
    # a reversed argmin reports FIRED, both from the measurements above, so this reads
    # the outcomes rather than re-deriving them. NOT RESOLVED and NOT APPLICABLE are
    # findings to report; only FIRED is a gate failure.
    reports = falsifiers(at_prior, at_flip)
    fired = [r.name for r in reports if r.outcome is Outcome.FIRED]
    assert not fired, f"registered falsifiers fired: {fired}"
    for report in reports:
        if report.outcome is Outcome.NOT_RESOLVED:
            print(f"   NOT RESOLVED: {report.name} — {report.detail}")
    grad = {h: c for h, _, c, _ in mech}
    assert grad[1] > pulls[0]  # the gradient starts above the pull
    assert grad[FLIP_H] < pulls[-1]  # ... and has decayed below it by H*

    # 3. The headline number matches an independent NumPy kernel at H* (atol 1e-9). The
    #    oracle takes its ln det magnitude from slogdet, the shipped kernel off a
    #    Cholesky factor. Both reject a non-PD argument (see the audit tests).
    backend, belief, preference, info_block = _setup()
    for policy in (_walk(FLIP_H), _reach(FLIP_H)):
        shipped = float(
            policy_efe_ffg(backend, belief, policy, preference, info_block=info_block)[
                0
            ]
        )
        assert abs(shipped - _numpy_score(policy)) < 1e-9

    # 4. The flip is epistemic, and the NaN guard is inert. With the epistemic term the
    #    argmin is cue-ward at H*; with it zeroed the argmin is still prior-ward, so the
    #    epistemic term is what flips it. Every enumerated G is finite.
    g_cue_ward, pragmatic_cue_ward, all_finite = epistemic_counterfactual(FLIP_H)
    assert all_finite, "some enumerated G is NaN -- the guard would be load-bearing"
    assert g_cue_ward is True  # with the epistemic term, the walk wins at H*
    assert pragmatic_cue_ward is False  # without it, the reach still wins -> epistemic

    # 5. The registered rollout conditioning clears its bars: Sigma_post stays positive
    #    definite and off the floor, so the Cholesky guard never fires (margin is real).
    rc = conditioning(_walk(FLIP_H))
    assert rc.all_positive_definite
    assert rc.min_eig_sigma_post.min() > MIN_EIG_FLOOR
    assert rc.cond_sigma_pred.max() < COND_CEILING
    assert rc.cond_s.max() < COND_CEILING
    assert rc.cond_sigma_post.max() < COND_CEILING

    # The equal-billing measurement (H* one sooner with the one-step reach -3) and the
    # refinement-stability check are heavier enumerations; they print in the bare run
    # and are recorded in the write-up rather than gated on every CI pass.
    print(f"Crossover at H* = {FLIP_H} on the registered set ({FLIP_H - 1} with the")
    print("one-step reach): the exhaustive argmin flips reach -> two-phase walk. The")
    print("gradient decays below a flat ~1.7-nat epistemic pull; zero it and the")
    print("flip moves to H~10, so the epistemic is load-bearing. Oracle- and")
    print("conditioning-confirmed.")
    _print_falsifiers(reports)


def main():
    """``--check`` asserts; the bare command prints the four measurement tables."""
    if "--check" in sys.argv:
        check()
        return
    _print_tables()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
