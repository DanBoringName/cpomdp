"""Render an animated GIF of a bacillus seeking food via continuous active inference.

The original v0.2 demo — kept in the gallery as the start of the journey. It is
the pure-LQR, fixed-sensor case (the epistemic term collapses to nothing, ADR-003);
the flagship ``bacillus_seeking_food.py`` is the v0.3 successor that switches the
epistemic term back on with a state-dependent sensor. See ``examples/README.md``.

The continuous-state answer to pymdp's mouse-seeking-cheese demo. A single
rod-shaped agent (a "bacillus") lives in a 2-D continuous plane. Its true
position is the hidden state; it only ever sees a noisy reading of where it is,
so it has to *infer* its own location while *acting* to reach a stationary food
particle (the goal the generative model prefers).

What each visual element maps onto in the model:

- **bacillus body** -- the true hidden state (position), rendered as a capsule
  with a wiggling flagellum so it reads as a swimming microbe rather than a dot.
- **belief marker (+)** -- the posterior mean ``agent.belief.mean``, where the
  agent *thinks* it is.
- **uncertainty ellipse** -- the positional posterior covariance
  ``agent.belief.cov``, a 2-sigma contour that shrinks as the filter grows
  confident.
- **food particle** -- the goal / prior preference the LQR controller steers
  toward.

Run it (``RUN`` = ``uv run --with matplotlib --with pillow python``)::

    RUN examples/bacillus_lqr.py            # -> docs/assets/bacillus_lqr.gif
    RUN examples/bacillus_lqr.py out.gif    # custom path

Needs ``matplotlib`` and ``pillow`` on top of cpomdp
(``pip install "cpomdp[examples]"``); neither is a runtime dependency of the library.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gallery
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from PIL import Image

from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal

gallery.use_headless_backend()  # render to a buffer, never open a window

# --- Okabe-Ito colourblind-safe palette --------------------------------------
BG, INK, GRID = gallery.FIELD.bg, gallery.FIELD.ink, gallery.FIELD.grid
AGENT = gallery.GREEN  # bluish-green -- the true bacillus
BELIEF = gallery.ORANGE  # orange       -- the belief mean (mu)
SIGMA = gallery.SKY  # sky blue     -- the uncertainty ellipse (Sigma)
FOOD = gallery.VERMILLION  # vermillion   -- the food / goal

# This demo's bacillus: the shared glyph's default proportions.
BACILLUS = gallery.BacillusStyle(body=AGENT, ink=INK)

VERSION_TAG = "cpomdp v0.2.0"


def build_model(dt: float) -> tuple[LinearGaussianModel, np.ndarray, np.ndarray]:
    """A 2-D point-swimmer: state ``[x, y, vx, vy]``, action pushes velocity.

    Velocity is lightly damped so that ``[fx, fy, 0, 0]`` is a genuine
    zero-action equilibrium (what the LQR controller needs in a goal). The agent
    senses position only -- never velocity -- so the filter has to recover the
    velocity from how the (noisy) position moves, exactly the quickstart story
    lifted into two dimensions.

    Returns the model plus the true start state and the prior belief mean, which
    are deliberately offset from each other so the belief is seen converging onto
    the truth.
    """
    damp = 0.92  # velocity decay per step -> v=0 is the only equilibrium
    dynamics = [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, damp, 0],
        [0, 0, 0, damp],
    ]
    control = [
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt],
    ]
    sensor_model = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]

    true_start = np.array([-3.6, -2.2, 0.0, 0.0])
    # The belief starts off-target and *uncertain*. The offset is deliberately
    # *lateral* to the food direction: the agent first commits toward where it
    # wrongly believes it is, then the filter drags the belief onto the truth, so
    # the trajectory curves as perception corrects action. Velocity is unknown
    # and the positional covariance is wide.
    belief_mean = np.array([-4.3, -1.0, 0.0, 0.0])
    prior_cov = np.diag([2.0, 2.0, 0.8, 0.8])

    model = LinearGaussianModel(
        dynamics=dynamics,
        control=control,
        sensor_model=sensor_model,
        dynamics_noise=np.diag([1e-5, 1e-5, 1e-4, 1e-4]),
        # Moderately noisy position sensor: the belief takes several steps to
        # lock on, so the uncertainty ellipse is seen *shrinking*, not snapping.
        sensor_noise=np.diag([0.22, 0.22]),
        prior=Belief(mean=belief_mean, cov=prior_cov),
    )
    return model, true_start, belief_mean


def simulate(n_steps: int, dt: float, seed: int = 7):
    """Run the perceive -> act loop, recording everything needed to draw it.

    Returns parallel lists: true states, belief means, positional 2x2
    covariances, and the food/goal position.
    """
    rng = np.random.default_rng(seed)
    model, true_state, _ = build_model(dt)
    food = np.array([2.6, 1.8])
    goal = np.array([food[0], food[1], 0.0, 0.0])

    # Softer effort penalty than the identity -> a swimmer that commits to the
    # food rather than creeping, giving a trajectory with visible curvature.
    agent = Agent(model, StateGoal(goal, effort=np.eye(2) * 3.0))

    # Frame 0 is the prior, before any observation: the wide opening ellipse.
    true_states = [true_state.copy()]
    means = [agent.belief.mean.copy()]
    covs = [agent.belief.cov[:2, :2].copy()]
    sensor_chol = np.linalg.cholesky(model.sensor_noise)
    control = model.control
    assert control is not None  # the swimmer is built with one, just above

    for _ in range(n_steps):
        # The agent sees a noisy reading of its true position, then perceives.
        obs = model.sensor_model @ true_state + sensor_chol @ rng.standard_normal(2)
        agent.infer_states(obs)
        action = agent.sample_action()

        true_states.append(true_state.copy())
        means.append(agent.belief.mean.copy())
        covs.append(agent.belief.cov[:2, :2].copy())  # positional block only

        # Advance the true plant with the action the agent actually applied.
        true_state = model.dynamics @ true_state + control @ action

    return true_states, means, covs, food


def _draw_bacillus(ax, pos, heading, phase):
    """This demo's bacillus glyph: the shared one, in this demo's colours."""
    gallery.draw_bacillus(ax, pos, heading, phase, BACILLUS)


def render(true_states, means, covs, food, out_path: Path, dt: float, fps: int = 20):
    """Draw every frame and write the looping GIF."""
    xs = [s[0] for s in true_states] + [food[0]]
    ys = [s[1] for s in true_states] + [food[1]]
    pad = 1.4
    xlim = (min(xs) - pad, max(xs) + pad)
    ylim = (min(ys) - pad, max(ys) + pad)

    frames: list[Image.Image] = []
    n = len(true_states)

    for i in range(n):
        fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors="#B0B0B0", labelsize=7)

        true_pos = np.array(true_states[i][:2])
        mean_pos = np.array(means[i][:2])
        cov = covs[i]

        # --- food / goal -----------------------------------------------------
        ax.plot(
            food[0],
            food[1],
            "*",
            color=FOOD,
            ms=26,
            markeredgecolor=INK,
            markeredgewidth=0.8,
            zorder=4,
        )

        # --- true trajectory so far -----------------------------------------
        if i > 0:
            tr = np.array(true_states[: i + 1])
            ax.plot(tr[:, 0], tr[:, 1], color=AGENT, lw=1.3, alpha=0.35, zorder=2)

        # --- uncertainty ellipse (2-sigma of the positional covariance) ------
        w, h, ell_angle = gallery.covariance_ellipse(cov)  # 2-sigma diameters
        ax.add_patch(
            Ellipse(
                gallery.xy(mean_pos),
                w,
                h,
                angle=ell_angle,
                facecolor=SIGMA,
                edgecolor=SIGMA,
                alpha=0.18,
                lw=1.2,
                zorder=3,
            )
        )
        ax.add_patch(
            Ellipse(
                gallery.xy(mean_pos),
                w,
                h,
                angle=ell_angle,
                facecolor="none",
                edgecolor=SIGMA,
                alpha=0.55,
                lw=1.2,
                zorder=3,
            )
        )

        # --- belief mean (mu) ------------------------------------------------
        ax.plot(
            mean_pos[0],
            mean_pos[1],
            "+",
            color=BELIEF,
            ms=13,
            mew=2.4,
            zorder=8,
        )

        # --- the bacillus at the TRUE state ----------------------------------
        vel = np.array(true_states[i][2:])
        heading = food - true_pos if np.linalg.norm(vel) < 0.001 else vel
        _draw_bacillus(ax, true_pos, heading, phase=i * 0.9)

        # --- legend ----------------------------------------------------------
        handles = [
            plt.Line2D(
                [],
                [],
                marker="o",
                color=AGENT,
                ls="none",
                ms=9,
                mec=INK,
                label="agent  (true state)",
            ),
            plt.Line2D(
                [],
                [],
                marker="+",
                color=BELIEF,
                ls="none",
                ms=11,
                mew=2.4,
                label="belief  (μ)",
            ),
            plt.Line2D(
                [],
                [],
                marker="o",
                color=SIGMA,
                ls="none",
                ms=9,
                alpha=0.5,
                label="uncertainty  (Σ)",
            ),
            plt.Line2D(
                [],
                [],
                marker="*",
                color=FOOD,
                ls="none",
                ms=13,
                mec=INK,
                label="food  (goal)",
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper left",
            framealpha=0.9,
            edgecolor=GRID,
            fontsize=8.5,
            labelcolor=INK,
        )

        ax.set_title(
            "bacillus seeking food  ·  continuous active inference",
            color=INK,
            fontsize=11,
            pad=10,
        )
        # error between belief and truth, shrinking as the filter locks on
        err = np.linalg.norm(mean_pos - true_pos)
        ax.text(
            0.985,
            0.02,
            f"step {i:>2d}/{n - 1}   |μ−x| = {err:4.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="#8A8A8A",
            fontsize=8,
            family="monospace",
        )
        ax.text(
            0.015,
            0.02,
            VERSION_TAG,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color="#8A8A8A",
            fontsize=8,
            family="monospace",
        )

        fig.tight_layout()
        frames.append(gallery.figure_frame(fig))
        plt.close(fig)

    # Hold the final frame a beat, then loop cleanly.
    return gallery.write_gif(frames, out_path, fps=fps, hold_seconds=1.2)


def main() -> None:
    """Main body for demo."""
    dt = 0.12
    n_steps = 60
    out = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/assets/bacillus_lqr.gif")
    )

    true_states, means, covs, food = simulate(n_steps, dt)
    path = render(true_states, means, covs, food, out, dt)

    final_err = np.linalg.norm(np.array(means[-1][:2]) - np.array(true_states[-1][:2]))
    reached = np.linalg.norm(np.array(true_states[-1][:2]) - food)
    print(f"wrote {path}  ({len(true_states)} steps)")
    print(f"  final belief error |μ−x| = {final_err:.3f}")
    print(f"  final distance to food   = {reached:.3f}")


if __name__ == "__main__":
    main()
