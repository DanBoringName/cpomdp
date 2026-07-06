"""The expressiveness boundary: a branch-coupled R(x) can't be flattened.

The v0.4 flagship. Two agents run the same continuous T-maze on the same factor
graph; the only difference is the cue sensor.

- Agent B: the cue noise is state-dependent, R(x), sharp only near the cue.
  Reading the cue sharpens it, so R(mu+) moves with the action -- the dual effect --
  and B's epistemic EFE term is live. So B values reading the cue.
- Agent A: same topology, fixed cue noise R(prior). Under fixed noise the epistemic
  term is constant (Koudahl-Kouw-de Vries 2021), so A can't value the cue and falls
  back to the prior-arm (LQR) choice.

Both believe the reward is in one arm and set off there together. B reads the cue,
learns it's in the other arm, and crosses. A can't read the cue and stays. They end
in opposite arms.

The headline is the raised error. B's model -- a mean-shifting coupling plus R(x) --
can't be flattened to a fixed linear-Gaussian model. A flat Kalman linearizes the
noise at the prior mean mu-; the factor graph linearizes at the coupling-resolved
predictive mean mu+; the coupling makes those differ, so no fixed model reproduces
R(mu+). Ask cpomdp to flatten it and it raises ``IncompatibleLinearizationError``.
A's fixed sensor flattens fine on the same topology, so the clash is R(x)-plus-
coupling, not the branching. The boundary panel makes that visible -- it plots
R(mu+) sliding with the candidate action against A's one fixed R, and no horizontal
line traces the curve.

The raised error witnesses that *cpomdp's* linear-Gaussian flattening fails at mu+, a
concrete necessary-condition tripwire -- not a proof of the general (infinite-
dimensional) filtering impossibility.

Not a biology model (ADR-020): the abstract T-maze is the canonical epistemic task
(Friston et al. 2015), here continuous-state on a factor graph.

Needs the ``examples`` extra (matplotlib + pillow). ``--check`` asserts the three
results with no plotting deps; the bare command writes the GIF, its poster still, the
boundary panel, and a static triptych for the paper::

    uv run --extra examples python examples/ffg/epistemic_dissociation_figure.py --check
    uv run --extra examples python examples/ffg/epistemic_dissociation_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# The whole demo builds off the documented top-level surface (ADR-021) -- one
# import block, no reaching into cpomdp internals.
from cpomdp import (
    Agent,
    Belief,
    CallableGaussianObservation,
    Coupling,
    CouplingGraph,
    CouplingGraphBackend,
    GaussianCoupling,
    GaussianObservation,
    GaussianTransition,
    IncompatibleLinearizationError,
    ObservationGoal,
)

# --- the T-maze, as one rooted tree -------------------------------------------
# Node 0 is the hidden CONTEXT c (which arm pays) -- never observed, resolved only
# through its coupling to node 1. Node 1 is [position, arm]: the action-driven
# position x, and a perceived-arm f the coupling keeps equal to the context (f ~= c).
# One sensor on node 1 reads the displacement o = f - x, so it sees the context
# through f -- one rooted tree carrying independent position plus a cue on a coupled
# context.
CONTEXT, ARM_NODE = 0, 1  # node indices; joint state is [c, x, f]

# geometry along the arm axis x. The prior points at the WRONG arm and the cue sits
# there, so belief-chasing routes BOTH agents to the wrong arm; B reads the cue,
# learns the truth, and reverses to the correct arm.
X_START = 0.0  # start at the junction
CUE_X = -3.0  # cue location = the wrong arm (where the prior points)
REWARD_TRUE = 3.0  # true context c: the correct arm is to the RIGHT
PRIOR_ARM = -3.0  # a-priori guess: the WRONG (left) arm -- routes both agents left
PRIOR_COV_CONTEXT = 5.0  # context loosely known: must detour to learn it
PRIOR_COV_POSITION = 0.05  # the agent knows its own position (dead reckoning)
PRIOR_COV_ARM = 5.0  # perceived-arm inherits the context uncertainty

# the R(x) precision well on the cue (info) channel -- a bacillus-style falloff keyed
# on position x: sharp (R_LO) at the cue, near-useless (R_HI) far away.
R_LO, R_HI, R_WIDTH = 0.02, 200.0, 1.2
R_GOAL = 200.0  # the fixed "commit" channel: dull, but carries the pragmatic pull

# process noise (all strictly positive; the information form inverts them)
Q_CONTEXT, Q_POSITION, Q_ARM = 1e-2, 1e-4, 1e-2
# coupling 0->1 noise: f tracks context tightly, x is left essentially free
COUPLE_Q_POSITION, COUPLE_Q_ARM = 1e3, 1e-2

ACTION_BOUNDS = (-2.0, 2.0)
GRID_N = 41
N_STEPS = 20
GOAL_PRECISION = 0.6  # Lambda on the commit channel (robust across ~0.1-4.0)
INFO_PRECISION = 1e-4  # ~0 on the info channel: decouples pragmatic from epistemic


def cue_noise(state, params):
    """R for node 1's two-channel sensor: ``diag([R_goal, R_well(x)])``, x = state[0].

    The fixed goal channel carries the pragmatic pull (its ambiguity is
    action-invariant, so it never traps the agent at the cue); the R(x) info channel
    is sharp near the cue and makes the epistemic term action-dependent -- R.
    """
    x = state[0]
    gap2 = (x - params["x0"]) ** 2
    falloff = 1.0 - jnp.exp(-gap2 / (2.0 * params["width"] ** 2))
    r_info = params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff
    return jnp.diag(jnp.array([params["r_goal"], r_info]))


def _cue_params():
    return {"x0": CUE_X, "r_lo": R_LO, "r_hi": R_HI, "width": R_WIDTH, "r_goal": R_GOAL}


def _fixed_noise_at_prior():
    """Agent A's fixed cue noise = R at the prior position (far from cue -> dull)."""
    return np.asarray(cue_noise(jnp.array([X_START, PRIOR_ARM]), _cue_params()))


def build_backend(*, epistemic_alive: bool) -> CouplingGraphBackend:
    """The shared branching T-maze; the cue's noise is the only A/B difference."""
    sensor_model = [[-1.0, 1.0], [-1.0, 1.0]]  # C: both channels read o = f - x
    cue = (
        CallableGaussianObservation(sensor_model, cue_noise, _cue_params())
        if epistemic_alive
        else GaussianObservation(sensor_model, _fixed_noise_at_prior())
    )
    context_to_arm = Coupling(
        parent=CONTEXT,
        child=ARM_NODE,
        factor=GaussianCoupling(
            coupling=[[0.0], [1.0]],  # W: the context drives the perceived arm f only
            coupling_noise=[[COUPLE_Q_POSITION, 0.0], [0.0, COUPLE_Q_ARM]],
        ),
        tau=1.0,
        efe_relevant=True,  # the covariance path the instrumental epistemic rides
    )
    graph = CouplingGraph(
        root=CONTEXT,
        dims=(1, 2),
        couplings=(context_to_arm,),
        observations={ARM_NODE: cue},
    )
    transitions = (
        GaussianTransition([[1.0]], [[Q_CONTEXT]]),  # node 0: context, near-static
        GaussianTransition(  # node 1: [position, arm]
            [[1.0, 0.0], [0.0, 1.0]], [[Q_POSITION, 0.0], [0.0, Q_ARM]]
        ),
    )
    control = [[0.0], [1.0], [0.0]]  # B: the 1-D action drives position (joint idx 1)
    return CouplingGraphBackend(graph, transitions, control=control)


def start_belief() -> Belief:
    """Joint prior over ``[c, x, f]``; the perceived arm mirrors the context guess."""
    mean = jnp.array([PRIOR_ARM, X_START, PRIOR_ARM])
    cov = jnp.diag(jnp.array([PRIOR_COV_CONTEXT, PRIOR_COV_POSITION, PRIOR_COV_ARM]))
    return Belief(mean=mean, cov=cov)


def build_agent(backend: CouplingGraphBackend) -> Agent:
    """An Agent whose static ``o = 0`` goal chases the context belief through f.

    Because the predicted commit reading is ``E[f]+ - x+`` and f tracks the context,
    "observe zero displacement" drives ``x -> E[context]`` -- resolving the context
    changes where the agent heads. ``info_target`` aims the epistemic at the context.
    """
    objective = ObservationGoal(
        target=[0.0, 0.0],  # observe zero displacement on both channels
        action_bounds=ACTION_BOUNDS,
        precision=[[GOAL_PRECISION, 0.0], [0.0, INFO_PRECISION]],  # weight commit only
        n_candidates=GRID_N,
        horizon=1,
        info_target=CONTEXT,
    )
    agent = Agent(objective=objective, backend=backend)
    agent.belief = start_belief()  # the FFG backend's model.prior is a placeholder
    return agent


def simulate(*, epistemic_alive: bool, seed: int = 7) -> dict:
    """Run one agent's closed perceive -> act loop; return its trajectory + beliefs."""
    rng = np.random.default_rng(seed)
    backend = build_backend(epistemic_alive=epistemic_alive)
    agent = build_agent(backend)
    params = _cue_params()

    x_true = float(X_START)
    positions = [x_true]
    actions, context_means, context_covs, arm_means = [], [], [], []

    for _ in range(N_STEPS):
        noise_cov = np.asarray(cue_noise(jnp.array([x_true, REWARD_TRUE]), params))
        displacement = REWARD_TRUE - x_true  # f_true - x_true, f_true = context truth
        draw = np.linalg.cholesky(noise_cov) @ rng.standard_normal(2)
        observation = np.array([displacement, displacement]) + draw

        agent.infer_states(observation)
        context = backend.marginal(CONTEXT, agent.belief)
        context_means.append(float(context.mean[0]))
        context_covs.append(float(context.cov[0, 0]))
        arm_means.append(float(agent.belief.mean[2]))

        action = float(np.asarray(agent.sample_action())[0])
        actions.append(action)
        x_true += action
        positions.append(x_true)

    return {
        "positions": np.array(positions),
        "actions": np.array(actions),
        "context_means": np.array(context_means),
        "context_covs": np.array(context_covs),
        "arm_means": np.array(arm_means),
        "backend": backend,
    }


def _boundary_scan(alive: bool) -> dict:
    """The action scan behind Result 2 and the boundary panel -- one shared source.

    Sweep the candidate-action grid; at each action read the info-channel noise
    ``R_info(mu+) = R`` at the coupling-resolved predictive mean (entry ``[1, 1]``; the
    ``[0, 0]`` goal channel is action-invariant) and the FFG epistemic term. ``check()``
    asserts on the spread, the boundary panel plots ``r_info_mu_plus`` -- same numbers
    from the same call, so the gate and the figure can't diverge.

    Returns arrays over the grid: ``grid`` (the actions), ``epistemic`` (the EFE
    epistemic value), ``r_info_mu_plus`` (``R_info(mu+(a))``), and ``mu_plus_x`` (the
    position component of node 1's *local* predictive mean -- the vector
    ``observation_noise_at`` hands to ``cue_noise``, which reads ``x = state[0]``).
    """
    from cpomdp.efe import _ffg_efe_step

    backend = build_backend(epistemic_alive=alive)
    sensor_model, _ = backend.observation_model
    arm_block = list(backend.block(ARM_NODE))  # node 1's joint indices: [position, arm]
    goal = jnp.array([0.0, 0.0])
    precision = jnp.array([[GOAL_PRECISION, 0.0], [0.0, INFO_PRECISION]])
    grid = np.linspace(*ACTION_BOUNDS, GRID_N)

    epistemic, r_info_mu_plus, mu_plus_x = [], [], []
    for action in grid:
        predicted = backend.predicted_belief(start_belief(), jnp.array([action]))
        noise = backend.observation_noise_at(predicted.mean)  # R(mu+)
        _, parts = _ffg_efe_step(
            predicted.mean,
            predicted.cov,
            sensor_model,
            noise,
            goal,
            precision,
            (CONTEXT,),
        )
        epistemic.append(float(parts["epistemic"]))
        r_info_mu_plus.append(float(np.asarray(noise)[1, 1]))  # info channel [1, 1]
        local = np.asarray(predicted.mean)[arm_block]  # cue_noise reads x = local[0]
        mu_plus_x.append(float(local[0]))
    return {
        "grid": grid,
        "epistemic": np.array(epistemic),
        "r_info_mu_plus": np.array(r_info_mu_plus),
        "mu_plus_x": np.array(mu_plus_x),
    }


# --- the three demonstrated results (assert them; --check prints, no plotting) ---
def check() -> None:
    """Assert the three results; the ``IncompatibleLinearizationError`` is the lead."""
    run_b = simulate(epistemic_alive=True)
    run_a = simulate(epistemic_alive=False)

    # Result 3 (the headline) first: the flat route cannot express B's model.
    print("Result 3 -- the expressiveness boundary (the headline):")
    backend_b = build_backend(epistemic_alive=True)
    backend_a = build_backend(epistemic_alive=False)
    try:
        backend_b.to_flat_model()
    except IncompatibleLinearizationError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected IncompatibleLinearizationError for Agent B")
    assert "state-dependent R(x)" in message
    print("  B: to_flat_model() raised IncompatibleLinearizationError -- PASS")
    backend_a.to_flat_model()  # fixed sensor, mu- = mu+ : flattens fine, no raise
    print("  A: to_flat_model() flattened fine (fixed sensor) -- PASS")
    print(
        "  a mean-shifting coupling makes mu+ != mu-, so no fixed flat model "
        "reproduces R(mu+)\n"
    )

    # Result 1: B resolves the hidden context and commits to the correct arm; A cannot.
    print("Result 1 -- B resolves the latent through the branch and acts on it:")
    var_b, var_a = run_b["context_covs"][-1], run_a["context_covs"][-1]
    assert var_b < var_a
    assert np.abs(run_b["actions"] - run_a["actions"]).max() > 1e-3
    assert abs(run_b["context_means"][-1] - REWARD_TRUE) < 1.0  # B ~ correct arm
    assert (run_b["positions"][-1] - X_START) * (REWARD_TRUE - X_START) > 0  # B right
    assert (run_a["positions"][-1] - X_START) * (REWARD_TRUE - X_START) < 0  # A wrong
    print(
        f"  final context var    B={var_b:.3f}  A={var_a:.3f}  "
        f"(B {var_a / var_b:.0f}x tighter)"
    )
    print(
        f"  context belief end   B={run_b['context_means'][-1]:+.2f}  "
        f"A={run_a['context_means'][-1]:+.2f}  (true {REWARD_TRUE:+.0f})"
    )
    print(
        f"  commit position end  B={run_b['positions'][-1]:+.2f} (correct arm)  "
        f"A={run_a['positions'][-1]:+.2f} (wrong arm) -- PASS\n"
    )

    # Result 2: B's epistemic term moves with the action; A's is flat (ADR-003/019).
    # Both the spread and the R(mu+) curve come from _boundary_scan -- the same call the
    # boundary panel plots, so the gate and the figure can never disagree.
    print("Result 2 -- the dual effect: B's epistemic is action-dependent, A's flat:")
    for tag, alive in (("B", True), ("A", False)):
        scan = _boundary_scan(alive)
        spread = float(np.ptp(scan["epistemic"]))
        r_spread = float(np.ptp(scan["r_info_mu_plus"]))
        if alive:
            assert spread > 1e-6
            assert r_spread > 1.0  # R(mu+) genuinely moves -> no single fixed R fits it
        else:
            assert spread < 1e-9
            assert r_spread < 1e-9  # fixed sensor: R is constant across every action
        print(
            f"  {tag}: epistemic range = {spread:.2e}   R(mu+) range = {r_spread:.1f}"
            f"   ({'VARIES' if spread > 1e-6 else 'flat'})"
        )
    print("\nAll three results PASS.")


# --- rendering ------------------------------------------------------------------
# The science above is frozen; everything below is presentation. The figure tells
# the dissociation as three held beats -- set off together, the cue read, the
# crossing -- interpolated with easing so the belief visibly glides and sharpens
# rather than jumping. Nothing here feeds back into the simulation.
BG = "#F6F6F3"  # page
PANEL = "#FFFFFF"  # panel fill
INK = "#22262B"  # near-black text
GRID = "#E2E2DE"  # hairlines / maze walls
FAINT = "#8B9095"  # secondary text
CUE_COLOR = "#0B6FB0"  # blue -- the cue and its sensing zone
REWARD_COLOR = "#D5581C"  # vermillion -- the reward and the belief about it
ALIVE = "#0E9E76"  # green -- Agent B, epistemic alive
DEAD = "#9AA0A6"  # grey  -- Agent A, epistemic dead
ARM = 3.9  # half-length of the drawn arm axis
Y0 = 0.60  # baseline of the belief-density strip above the corridor


def _ease(t: float) -> float:
    """Smoothstep easing -- eases both ends so motion starts and stops gently."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# The story as a handful of stops sampled from the run, each held for `hold` frames
# and reached over `travel` eased frames. `reveal` fades the true reward in as B
# commits. This is where the pacing lives -- the sim has 20 steps but the narrative
# is these six beats; the static tail is dropped.
_STOPS = (
    {
        "i": 0,
        "hold": 10,
        "travel": 8,
        "beat": "both believe the reward is on the left — they set off together",
    },
    {"i": 1, "hold": 2, "travel": 8, "beat": "closing on the cue"},
    {
        "i": 2,
        "hold": 18,
        "travel": 9,
        "reveal": 0.0,
        "beat": "B reads the cue: the reward is on the right — A reads nothing",
    },
    {"i": 3, "hold": 2, "travel": 7, "beat": "B reverses"},
    {"i": 4, "hold": 2, "travel": 8, "reveal": 1.0, "beat": "… and crosses the maze"},
    {
        "i": 5,
        "hold": 20,
        "travel": 0,
        "reveal": 1.0,
        "beat": "B is on the reward; A is stranded at the wrong arm",
    },
)


def _story_frames(run_a: dict, run_b: dict) -> list[dict]:
    """Expand the stops into per-frame states, interpolating positions and beliefs."""
    n_belief = len(run_b["context_means"])

    def sample(run, i):
        j = min(i, n_belief - 1)
        return {
            "pos": float(run["positions"][i]),
            "mean": float(run["context_means"][j]),
            "var": float(run["context_covs"][j]),
        }

    pos_a = [run_a["positions"][s["i"]] for s in _STOPS]
    pos_b = [run_b["positions"][s["i"]] for s in _STOPS]
    reveal_of = [s.get("reveal", 0.0) for s in _STOPS]

    def frame(k, e, extra_a, extra_b):
        prev, nxt = _STOPS[k], _STOPS[min(k + 1, len(_STOPS) - 1)]
        a0, a1 = sample(run_a, prev["i"]), sample(run_a, nxt["i"])
        b0, b1 = sample(run_b, prev["i"]), sample(run_b, nxt["i"])
        reveal = _lerp(reveal_of[k], reveal_of[min(k + 1, len(_STOPS) - 1)], e)
        return {
            "a": {key: _lerp(a0[key], a1[key], e) for key in a0},
            "b": {key: _lerp(b0[key], b1[key], e) for key in b0},
            "reveal": reveal,
            "beat": prev["beat"] if e < 0.5 else nxt["beat"],
            "step": prev["i"] if e < 0.5 else nxt["i"],
            "trail_a": extra_a,
            "trail_b": extra_b,
        }

    frames: list[dict] = []
    for k, stop in enumerate(_STOPS):
        trail_a = list(pos_a[: k + 1])
        trail_b = list(pos_b[: k + 1])
        frames.extend(frame(k, 0.0, trail_a, trail_b) for _ in range(stop["hold"]))
        if k + 1 < len(_STOPS):
            span = stop["travel"]
            for t in range(1, span + 1):
                e = _ease(t / span)
                ta = [*trail_a, _lerp(pos_a[k], pos_a[k + 1], e)]
                tb = [*trail_b, _lerp(pos_b[k], pos_b[k + 1], e)]
                frames.append(frame(k, e, ta, tb))
    return frames


def _draw_maze(ax, *, reveal: float) -> None:
    """The corridor: the two arms of the T, cue at one end, reward at the other."""
    from matplotlib.patches import Circle, FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (-ARM, -0.34),
            2 * ARM,
            0.68,
            boxstyle="round,pad=0.02,rounding_size=0.34",
            facecolor=PANEL,
            edgecolor=GRID,
            lw=1.5,
            zorder=2,
        )
    )
    for end in (CUE_X, REWARD_TRUE):
        ax.add_patch(
            Circle((end, 0), 0.52, facecolor=BG, edgecolor=GRID, lw=1.2, zorder=2)
        )

    # the cue: a blue diamond with a soft sensing glow
    ax.add_patch(
        Circle(
            (CUE_X, 0), 0.9, facecolor=CUE_COLOR, alpha=0.10, zorder=2, edgecolor="none"
        )
    )
    ax.plot(
        CUE_X, 0, marker="D", ms=12, color=CUE_COLOR, mec="white", mew=1.4, zorder=6
    )
    ax.text(
        CUE_X,
        -0.72,
        "cue",
        ha="center",
        va="top",
        color=CUE_COLOR,
        fontsize=8.5,
        fontweight="bold",
    )

    # the true reward: a hidden marker until B commits, then a bright star
    hidden = reveal < 0.05
    ax.plot(
        REWARD_TRUE,
        0,
        marker="*",
        ms=_lerp(0, 24, reveal) + 8,
        color=REWARD_COLOR,
        mec=INK,
        mew=0.8 * reveal + 0.2,
        alpha=_lerp(0.0, 1.0, reveal),
        zorder=7,
    )
    if hidden:
        ax.text(
            REWARD_TRUE,
            0,
            "?",
            ha="center",
            va="center",
            color=FAINT,
            fontsize=13,
            fontweight="bold",
            zorder=7,
        )
    if reveal > 0.5:
        ax.text(
            REWARD_TRUE,
            -0.72,
            "reward",
            ha="center",
            va="top",
            color=REWARD_COLOR,
            fontsize=8.5,
            fontweight="bold",
            alpha=(reveal - 0.5) * 2,
        )


def _draw_belief_strip(ax, mean: float, var: float) -> None:
    """P(reward location) as a filled Gaussian density curve above the corridor.

    Broad, low hump when the reward arm is unknown; a sharp, tall spike once the
    cue resolves it. This is the epistemic state made visible -- the same quantity
    whose action-dependence is dead for Agent A and alive for Agent B.
    """
    resolved = var < 0.6
    xs = np.linspace(-ARM, ARM, 200)
    sigma = float(np.clip(np.sqrt(max(var, 1e-6)), 0.15, 3.4))
    amp = float(np.clip(0.92 / (0.55 + sigma), 0.30, 0.74))
    curve = Y0 + amp * np.exp(-((xs - mean) ** 2) / (2.0 * sigma**2))
    peak = float(np.clip(mean, -ARM, ARM))

    ax.plot([-ARM, ARM], [Y0, Y0], color=GRID, lw=1.0, zorder=3)
    ax.fill_between(
        xs,
        Y0,
        curve,
        color=REWARD_COLOR,
        alpha=0.24 if resolved else 0.13,
        lw=0,
        zorder=3,
    )
    ax.plot(
        xs, curve, color=REWARD_COLOR, lw=1.4, alpha=0.9 if resolved else 0.5, zorder=4
    )
    ax.plot(
        [peak, peak],
        [0.34, Y0],
        color=REWARD_COLOR,
        lw=0.9,
        ls=":",
        alpha=0.55 if resolved else 0.3,
        zorder=3,
    )
    ax.plot(
        peak,
        Y0 + amp,
        marker="*",
        ms=11 if resolved else 8,
        color=REWARD_COLOR,
        mec="white",
        mew=0.9,
        alpha=0.95 if resolved else 0.55,
        zorder=6,
    )


def _panel_status(which: str) -> tuple[str, str]:
    """Header + subtitle, leading with whether the model flattens to a Kalman filter."""
    if which == "A":
        return "A   fixed sensor", "flattens to a Kalman filter · epistemic dead"
    return "B   R(x) sensor", "won't flatten — FFG-only · epistemic alive"


def _draw_panel(ax, which, state, reveal, *, accent, arrived) -> None:
    from matplotlib.patches import Circle

    ax.set_xlim(-ARM - 0.7, ARM + 0.7)
    ax.set_ylim(-1.05, 1.75)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(ALIVE if arrived else GRID)
        spine.set_linewidth(2.4 if arrived else 1.0)

    _draw_maze(ax, reveal=reveal)
    _draw_belief_strip(ax, state["mean"], state["var"])

    trail = np.asarray(state["trail"])
    ax.plot(trail, np.zeros_like(trail), color=accent, lw=2.4, alpha=0.35, zorder=4)
    ax.plot(
        state["pos"], 0, marker="o", ms=14, color=accent, mec="white", mew=1.6, zorder=9
    )
    if arrived:
        ax.add_patch(
            Circle(
                (state["pos"], 0),
                0.42,
                facecolor="none",
                edgecolor=ALIVE,
                lw=1.6,
                alpha=0.7,
                zorder=8,
            )
        )

    ax.text(
        0.0,
        (0.34 + Y0) / 2,
        "belief: P(reward arm)",
        ha="center",
        va="center",
        color=FAINT,
        fontsize=6.5,
        style="italic",
        zorder=5,
    )
    header, status = _panel_status(which)
    ax.text(
        0.025,
        0.955,
        header,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=accent,
        fontsize=11,
        fontweight="bold",
        family="monospace",
    )
    ax.text(
        0.025,
        0.855,
        status,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=FAINT,
        fontsize=8.2,
    )


def _figure(fr):
    """Compose the two-panel figure (A | B) for one frame state."""
    import matplotlib.pyplot as plt

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.6, 3.55), dpi=100)
    fig.patch.set_facecolor(BG)
    arrived = fr["reveal"] > 0.5 and fr["b"]["pos"] > 1.5

    _draw_panel(
        ax_a,
        "A",
        {**fr["a"], "trail": fr["trail_a"]},
        fr["reveal"],
        accent=DEAD,
        arrived=False,
    )
    _draw_panel(
        ax_b,
        "B",
        {**fr["b"], "trail": fr["trail_b"]},
        fr["reveal"],
        accent=ALIVE,
        arrived=arrived,
    )

    fig.suptitle(
        "why a factor graph — B's model can't be flattened to a Kalman filter, A's can",
        color=INK,
        fontsize=13,
        fontweight="bold",
        y=0.965,
    )
    fig.text(0.5, 0.115, fr["beat"], ha="center", va="center", color=INK, fontsize=9.5)
    fig.text(
        0.5,
        0.035,
        "so a flat loop can't run B at all — on the FFG it reads the "
        "cue and crosses to the reward, while A collapses to LQR and stays",
        ha="center",
        va="center",
        color=FAINT,
        fontsize=7.6,
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.87, bottom=0.17, wspace=0.05)
    return fig, plt


def render_gif(frames, out_path, *, fps=14):
    """Render the frame states to a looping GIF; slow fps keeps the beats readable."""
    import matplotlib as mpl

    mpl.use("Agg")
    from PIL import Image

    images = []
    for fr in frames:
        fig, plt = _figure(fr)
        fig.canvas.draw()
        images.append(
            Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
        )
        plt.close(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return out_path


def render_still(frame, out_path):
    """Render one frame state to a high-DPI PNG (the hero still is the last frame)."""
    import matplotlib as mpl

    mpl.use("Agg")
    fig, plt = _figure(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG, dpi=200)
    plt.close(fig)
    return out_path


def _draw_boundary_panel(ax, scan_b: dict, a_fixed_r_info: float) -> None:
    """R_info(mu+) vs A's one fixed R -- the impossibility at a glance.

    Green: B's info-channel noise ``R(mu+(a))``, which slides as the action moves the
    linearisation point mu+. Grey dashed: A's single fixed R. No horizontal line traces
    the curve, so no fixed linear-Gaussian model reproduces ``R(mu+)`` -- that is the
    ``IncompatibleLinearizationError``. The two meet only at the no-drift action, where
    mu+ = mu-; the shaded gap everywhere else is the model the flat route can't hold. A
    faint second axis carries the *cause*: mu+'s position drifting with the action.
    """
    grid, curve = scan_b["grid"], scan_b["r_info_mu_plus"]
    span = float(np.ptp(curve))
    i0 = int(np.argmin(np.abs(grid)))  # the no-drift action, mu+ = mu-

    ax.set_facecolor(PANEL)
    ax.fill_between(
        grid, curve, a_fixed_r_info, color=ALIVE, alpha=0.13, lw=0, zorder=2
    )
    ax.axhline(a_fixed_r_info, color=DEAD, lw=1.8, ls="--", zorder=4)
    ax.plot(grid, curve, color=ALIVE, lw=2.6, zorder=5)

    # the cause, faint on a twin axis: the linearisation point mu+ slides with a
    ax2 = ax.twinx()
    ax2.plot(
        grid,
        scan_b["mu_plus_x"] - scan_b["mu_plus_x"][i0],
        color=FAINT,
        lw=1.0,
        ls=":",
        alpha=0.8,
        zorder=3,
    )
    ax2.set_ylabel("mu+ position drift (the cause)", color=FAINT, fontsize=7.5)
    ax2.tick_params(colors=FAINT, labelsize=6.5)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_xlabel("candidate action  a", color=INK, fontsize=9)
    ax.set_ylabel("info-channel noise  R(mu+)", color=ALIVE, fontsize=9)
    ax.tick_params(colors=FAINT, labelsize=7.5)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.set_title(
        "the boundary: R(mu+) moves with the action — no fixed R can match it",
        color=INK,
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    ax.text(
        grid[2],
        curve[2],
        "B: R(mu+(a))",
        color=ALIVE,
        fontsize=9.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax.text(
        grid[0] + 0.06,
        a_fixed_r_info + span * 0.015,
        "A: one fixed R",
        color=DEAD,
        fontsize=9.5,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    kmax = int(np.argmax(np.abs(curve - a_fixed_r_info)))  # the widest gap
    ax.annotate(
        "no fixed R reproduces R(mu+)\n→ IncompatibleLinearizationError",
        xy=(grid[kmax], (curve[kmax] + a_fixed_r_info) / 2),
        xytext=(0.60, 0.28),
        textcoords="axes fraction",
        color=INK,
        fontsize=8.5,
        ha="center",
        va="center",
        arrowprops={"arrowstyle": "-", "color": FAINT, "lw": 1.0},
    )
    ax.plot(
        grid[i0],
        curve[i0],
        marker="o",
        ms=6,
        color=DEAD,
        mec="white",
        mew=1.0,
        zorder=6,
    )
    ax.text(
        grid[i0] - 0.08,
        curve[i0] + span * 0.05,
        "agree only where mu+ = mu-",
        color=FAINT,
        fontsize=7,
        ha="right",
        va="bottom",
    )


def render_boundary(out_path: Path) -> Path:
    """Draw the boundary panel (R(mu+) vs A's fixed R) and write it to ``out_path``."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    scan_b = _boundary_scan(alive=True)
    a_fixed_r_info = float(np.asarray(_fixed_noise_at_prior())[1, 1])
    assert np.ptp(scan_b["r_info_mu_plus"]) > 1.0  # the curve must genuinely move

    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=150)
    fig.patch.set_facecolor(BG)
    _draw_boundary_panel(ax, scan_b, a_fixed_r_info)
    fig.text(
        0.5,
        0.015,
        "ask cpomdp to flatten B → IncompatibleLinearizationError    "
        "·    A's fixed sensor flattens fine",
        ha="center",
        va="bottom",
        color=FAINT,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.11, right=0.89, top=0.88, bottom=0.17)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    return out_path


# The three held beats the triptych freezes: (sim step, reward reveal, caption).
_TRIPTYCH_BEATS = (
    (0, 0.0, "both believe left — they set off together"),
    (2, 0.0, "B reads the cue: reward is right — A reads nothing"),
    (5, 1.0, "B crosses to the reward — A is stranded left"),
)


def _beat_state(run: dict, i: int) -> dict:
    """A single-frame panel state sampled straight from the run at sim step ``i``."""
    j = min(i, len(run["context_means"]) - 1)
    return {
        "pos": float(run["positions"][i]),
        "mean": float(run["context_means"][j]),
        "var": float(run["context_covs"][j]),
        "trail": list(run["positions"][: i + 1]),
    }


def render_triptych(run_a: dict, run_b: dict, out_path: Path) -> Path:
    """Three held maze beats side by side (A over B) -- the static story for docs."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 4.9), dpi=150)
    fig.patch.set_facecolor(BG)
    for col, (i, reveal, caption) in enumerate(_TRIPTYCH_BEATS):
        a_state, b_state = _beat_state(run_a, i), _beat_state(run_b, i)
        arrived = reveal > 0.5 and b_state["pos"] > 1.5
        _draw_panel(axes[0, col], "A", a_state, reveal, accent=DEAD, arrived=False)
        _draw_panel(axes[1, col], "B", b_state, reveal, accent=ALIVE, arrived=arrived)
        axes[0, col].set_title(caption, color=INK, fontsize=9, pad=6)

    fig.suptitle(
        "why a factor graph — B's model can't be flattened to a Kalman filter, A's can",
        color=INK,
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.025,
        "ask cpomdp to flatten B → IncompatibleLinearizationError    "
        "·    B reads the state-dependent cue and crosses; A collapses to LQR and "
        "stays",
        ha="center",
        va="bottom",
        color=FAINT,
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.88, bottom=0.075, wspace=0.04, hspace=0.16
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    return out_path


def main():
    """``--check`` asserts the three results; otherwise render the figure set."""
    if "--check" in sys.argv:
        check()
        return
    out = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("docs/assets/epistemic_dissociation.gif")
    )
    run_b = simulate(epistemic_alive=True)
    run_a = simulate(epistemic_alive=False)
    frames = _story_frames(run_a, run_b)
    gif = render_gif(frames, out)
    still = render_still(frames[-1], out.with_suffix(".png"))
    boundary = render_boundary(out.with_name("epistemic_dissociation_boundary.png"))
    triptych = render_triptych(
        run_a, run_b, out.with_name("epistemic_dissociation_triptych.png")
    )
    print(
        f"wrote {gif} ({len(frames)} frames)\nwrote {still}\nwrote {boundary}\n"
        f"wrote {triptych}"
    )


if __name__ == "__main__":
    main()
