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

import epistemic_dissociation_figure as demo
import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.crossover import crossover_statistic
from cpomdp.diagnostics import logdet_pd, rollout_conditioning
from cpomdp.efe import policy_efe_ffg, policy_efe_ffg_trace
from cpomdp.enumeration import EnumeratedEfeSearch, FiniteActionSet
from cpomdp.selection import Preference
from cpomdp.types import Belief
from cpomdp.warrant import CheckReport, Outcome, Tier, Warrant, check_summary

# The rollout-hygiene bars this shares with tests/test_rollout_hygiene.py: a floor under
# min-eigenvalue(Σ_post) and a ceiling on cond(Σ⁺), cond(S), cond(Σ_post).
MIN_EIG_FLOOR = 1e-9
COND_CEILING = 1e8

# The registered action set the anchors were fixed against. It clips the reach at the
# grid edge -2, so it needs two steps to reach the goal at -3; a set containing the
# optimal reach -3 flips one horizon sooner (see the +edge measurement below). H* on
# this set is therefore an upper bound, and the honest headline number.
V1 = [-2.0, -1.0, 0.0, 1.0, 2.0]  # the registered action set (clips the reach at -2)
V1_EDGE = [-3.0, *V1]  # + the optimal reach -3; flips one horizon sooner
ACTION_SET = FiniteActionSet([[a] for a in V1], version="v1")
EDGE_SET = FiniteActionSet([[a] for a in V1_EDGE], version="v1-edge")
FLIP_H = 7  # the crossover horizon on ACTION_SET
MAX_H = 7  # the exhaustive sweep budget (feasibility is |A|^H * H, printed below)
# Declared feasibility bound: enumeration is feasible to H_MAX (5^9 * 9 = 17.6M scored
# steps, measured); the crossover at H=7 sits well inside it and is cue-ward through 9.
H_MAX = 9
EPS_PLATEAU = 0.2  # the epistemic pull varies by less than this: flat, not growing


def _setup():
    """``(backend, belief, preference, target)`` for the coupled-tree cue task."""
    backend = demo.build_backend(epistemic_alive=True, cue_x=demo.CUE_DETOUR_X)
    belief = demo.start_belief()
    preference = Preference(
        goal=[0.0, 0.0],
        precision=[[demo.GOAL_PRECISION, 0.0], [0.0, demo.INFO_PRECISION]],
    )
    target = tuple(backend.block(demo.CONTEXT))
    return backend, belief, preference, target


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
    backend, belief, preference, target = _setup()
    search = EnumeratedEfeSearch.over_backend(
        backend, action_set, target=target, horizon=horizon
    )
    result = search.evaluate(belief, preference)
    policy = np.asarray(result.best_policy).ravel()
    return search.n_policies, float(result.g.min()), policy, _cue_ward(policy)


def exhaustive_flip(action_set=ACTION_SET, max_horizon=MAX_H):
    """Per-horizon exhaustive argmin ``(H, n_policies, Gmin, argmin, cue_ward)``."""
    return [(h, *argmin_at(h, action_set)) for h in range(1, max_horizon + 1)]


# --- 2. the mechanism split (post-selection exposition) ----------------------------
def mechanism_curve(max_horizon=MAX_H):
    """Per-horizon ``(H, Δε, Δc, ΔG)`` for the selected walk against the coast reach."""
    backend, belief, preference, target = _setup()
    rows = []
    for horizon in range(1, max_horizon + 1):
        stat = crossover_statistic(
            backend, belief, _walk(horizon), _reach(horizon), preference, target=target
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
    backend, belief, preference, target = _setup()
    sensor_model = np.asarray(backend.observation_model[0])  # C
    precision = np.asarray(preference.precision)  # Lambda
    goal = np.asarray(preference.goal)
    idx = np.array(target)

    carry = belief
    total = 0.0
    for action in np.asarray(policy).ravel():
        predicted = backend.predicted_belief(carry, jnp.array([float(action)]))
        mean = np.asarray(predicted.mean)  # mu+
        cov = np.asarray(predicted.cov)  # Sigma+
        noise = np.asarray(backend.observation_noise_at(predicted.mean))  # R(mu+)
        obs = sensor_model @ mean  # o+ = C mu+
        innovation = sensor_model @ cov @ sensor_model.T + noise  # S
        pragmatic = 0.5 * (obs - goal) @ precision @ (obs - goal) + 0.5 * np.trace(
            precision @ innovation
        )
        post_info = (
            np.linalg.inv(cov) + sensor_model.T @ np.linalg.inv(noise) @ sensor_model
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
            cov - cov @ sensor_model.T @ np.linalg.inv(innovation) @ sensor_model @ cov
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
    backend, belief, preference, target = _setup()
    trace = policy_efe_ffg_trace(
        backend, belief, jnp.asarray(policy).reshape(-1, 1), preference, target=target
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
    backend, belief, preference, target = _setup()
    combos = np.array(list(itertools.product(actions, repeat=horizon)), dtype=float)

    @jax.jit
    def batch(pols):
        def one(pol):
            g, parts = policy_efe_ffg(backend, belief, pol, preference, target=target)
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
def falsifiers() -> tuple[CheckReport, ...]:
    """The four D3 falsifiers registered for this crossover, one report each.

    A falsifier does not pass. ``NOT TRIGGERED`` is "it ran and the condition did not
    obtain", so the claim survives it. ``NOT APPLICABLE`` is void by construction, so
    it is evidence for nothing and is not a survivor. ``NOT RUN HERE`` was measured
    elsewhere. Neither of the last two ran, so neither carries a warrant: the prover
    cell is ``—``, because attributing one would claim evidence that was never produced.

    Rows 1 and 2 read on both axes at once, which is the point of carrying them
    separately. On the prover axis they are ``PROVED``, each resting on the exhaustive
    per-horizon enumeration and carrying its completeness certificate, so "no policy in
    the declared set flips earlier" is decided rather than sampled. On the tier axis
    they are ``B``: the margin is measured against the error ``COND_CEILING`` allows on
    a difference of two scores, named in the detail with the ``H*`` qualifier beside it.

    A margin inside that bound is ``NOT RESOLVED``, not a failure. The ordering would
    genuinely be undetermined, and the honest report of a tie is a tie.
    """
    backend, _, _, target = _setup()
    certificate = EnumeratedEfeSearch.over_backend(
        backend, ACTION_SET, target=target, horizon=FLIP_H
    ).certificate
    walk, reach = _numpy_score(_walk(FLIP_H)), _numpy_score(_reach(FLIP_H))
    bound = flip_margin_error(walk, reach)
    separated = abs(walk - reach) > bound
    # A tie routes to NOT RESOLVED rather than to an exception. The bound is a bar the
    # margin is read against, and "these two are too close to order" is a finding.
    flip = Outcome.NOT_TRIGGERED if separated else Outcome.NOT_RESOLVED
    bar = (
        f"|ΔG| = {abs(walk - reach):.4f} vs error bound {bound:.2e} "
        f"(cond ≤ {COND_CEILING:.0e}), {abs(walk - reach) / bound:.0e}x"
    )
    # Travels with the number everywhere it is quoted: the declared set clips the reach
    # at -2 while the unconstrained optimum is -3, so 7 is an upper bound on H*, and a
    # set containing -3 flips at 6. The bound above certifies the arithmetic, which was
    # never the exposure here.
    upper = f"H* = {FLIP_H} is an upper bound (set clips the reach at -2, optimum -3)"
    return (
        CheckReport(
            name="1. no crossover at feasible H",
            warrant=Warrant.PROVED,
            outcome=flip,
            tier=Tier.B,
            detail=(
                f"argmin is cue-ward at H* = {FLIP_H}, inside H_MAX = {H_MAX}. "
                f"{upper}. {bar}"
            ),
            evidence=certificate,
        ),
        CheckReport(
            name="2. flip not clean at H*/H*-1",
            warrant=Warrant.PROVED,
            outcome=flip,
            tier=Tier.B,
            detail=(
                f"ΔG changes sign once, argmin flips at the same step. {upper}. {bar}"
            ),
            evidence=certificate,
        ),
        CheckReport(
            name="3. not reproducible across seeds",
            warrant=None,
            outcome=Outcome.NOT_APPLICABLE,
            tier=Tier.C,
            detail="void by construction: no observation draw to vary across seeds",
        ),
        CheckReport(
            name="4. H* unstable under refinement",
            warrant=None,
            outcome=Outcome.NOT_RUN_HERE,
            tier=Tier.C,
            detail=(
                f"step-0.5 refinement costs {9**7 * 7} steps. Recorded in the "
                "write-up, and the live exposure on this number"
            ),
        ),
    )


def _print_falsifiers() -> None:
    """Print the registered falsifiers on both axes, then the run's counts.

    Prover and tier print as separate columns because they answer separate questions.
    A row can be decided and have nothing stated behind its margin, or measured against
    a tight bar and only sampled. Collapsing them into one verdict loses whichever half
    the reader needed.
    """
    reports = falsifiers()
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
    backend, belief, preference, target = _setup()
    walk_ship = float(
        policy_efe_ffg(backend, belief, _walk(FLIP_H), preference, target=target)[0]
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
    print(f"   optimal reach -3 -> H* = {edge_star}; the registered set clips the")
    print(f"   reach to -2, so H* = {FLIP_H} is an upper bound.")
    print(f"   recorded, not re-run here (cost {refine_cost} steps): a step-0.5")
    print("   refinement of the same range left the H=6 and H=7 argmins unchanged,")
    print("   no intermediate action scoring lower G (a subset check; step-0.25")
    print("   dropped on cost).")
    print("   analytic bound: relief <= 0.77/step vs 2.77 detour -> H* >= 6.")
    print(f"   feasibility: declared to H_MAX = {H_MAX} (cost {hmax_cost} scored")
    print("   steps); the argmin is cue-ward at H = 7, 8, 9.")
    _print_falsifiers()


# --- the gate ----------------------------------------------------------------------
def check() -> None:
    """Assert the flip, the mechanism split, the oracle, and the kernel's inertness.

    The exhaustive enumerations are scoped to the flip boundary and H* (the horizons the
    assertions bear on), so the gate enumerates ~230k policies, not the full sweep.
    """
    # 1. Decisive: the exhaustive argmin is a reach at H*-1 and a cue-ward walk at H*.
    assert argmin_at(FLIP_H - 1)[3] is False, "argmin should still be a reach at H*-1"
    assert argmin_at(FLIP_H)[3] is True, "argmin should be a cue-ward walk at H*"

    # 2. Mechanism: pull flat, gradient decays below it, clean sign change at H*/H*-1.
    mech = mechanism_curve()
    pulls = [de for _, de, _, _ in mech]
    assert max(pulls) - min(pulls) < EPS_PLATEAU  # epistemic does not accumulate
    dg = {h: g for h, _, _, g in mech}
    assert dg[FLIP_H - 1] > 0  # reach still wins at H*-1
    assert dg[FLIP_H] < 0  # walk wins at H* -- a clean single sign change
    # Both sign changes clear the error COND_CEILING allows on a difference of two
    # scores, so neither is a numerical near-tie read as a decision. A margin inside
    # the bound is reported as NOT RESOLVED by `falsifiers()`, not raised here: a tie
    # is a finding about the measurement, and an exception would erase it.
    for horizon in (FLIP_H - 1, FLIP_H):
        walk_g, reach_g = _numpy_score(_walk(horizon)), _numpy_score(_reach(horizon))
        if abs(dg[horizon]) <= flip_margin_error(walk_g, reach_g):
            print(f"   NOT RESOLVED: ΔG({horizon}) sits inside the error bound")

    # No registered falsifier fired. NOT RESOLVED and NOT APPLICABLE are outcomes to
    # report, so only FIRED is a gate failure.
    fired = [r.name for r in falsifiers() if r.outcome is Outcome.FIRED]
    assert not fired, f"registered falsifiers fired: {fired}"
    grad = {h: c for h, _, c, _ in mech}
    assert grad[1] > pulls[0]  # the gradient starts above the pull
    assert grad[FLIP_H] < pulls[-1]  # ... and has decayed below it by H*

    # 3. The headline number matches an independent NumPy kernel at H* (atol 1e-9). The
    #    oracle takes its ln det magnitude from slogdet, the shipped kernel off a
    #    Cholesky factor. Both reject a non-PD argument (see the audit tests).
    backend, belief, preference, target = _setup()
    for policy in (_walk(FLIP_H), _reach(FLIP_H)):
        shipped = float(
            policy_efe_ffg(backend, belief, policy, preference, target=target)[0]
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

    # The equal-billing measurement (H* one sooner with the optimal reach -3) and the
    # refinement-stability check are heavier enumerations; they print in the bare run
    # and are recorded in the write-up rather than gated on every CI pass.
    print(f"Crossover at H* = {FLIP_H} on the registered set ({FLIP_H - 1} with the")
    print("optimal reach): the exhaustive argmin flips reach -> two-phase walk. The")
    print("gradient decays below a flat ~1.7-nat epistemic pull; zero it and the")
    print("flip moves to H~10, so the epistemic is load-bearing. Oracle- and")
    print("conditioning-confirmed.")
    _print_falsifiers()


def main():
    """``--check`` asserts; the bare command prints the four measurement tables."""
    if "--check" in sys.argv:
        check()
        return
    _print_tables()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    main()
