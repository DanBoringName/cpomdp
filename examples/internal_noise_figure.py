"""Figure: the epistemic collapse breaks from the *inside* — through process noise.

The companion to examples/efe_collapse_figure.py. There, a state-dependent sensor
R(x) made the epistemic term action-dependent. Here the sensor noise R is held
FIXED and the action-dependence comes entirely from state-dependent internal
process noise Q(x) — the route RFC-001 chapter 8 cares about, where the binding
precision constraint lives in internal processing rather than the sensor.

- LEFT (fixed Q): epistemic is flat across actions — the ADR-003 collapse, same as
  a fixed sensor. R and Q are both constant, so Σ⁺, S, and the info gain do not
  move with the action.
- RIGHT (state-dependent Q(x)): epistemic curves. Q is evaluated at μ⁺ (the
  arrived-at state), so it depends on the action; Σ⁺ = AΣAᵀ + Q(μ⁺) and the info
  gain move with it. R is the SAME constant as on the left — the whole effect is
  internal.

Run: `uv run python examples/internal_noise_figure.py`
Output: docs/assets/internal_noise.png
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cpomdp.dynamics import CallableProcessNoise
from cpomdp.efe import expected_free_energy
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "internal_noise.png"

# 1-D single integrator: μ⁺ = μ + a, so the action moves the state directly and a
# one-step sweep is clean. R (sensor noise) is fixed throughout — the only thing
# that varies between the panels is Q.
DYNAMICS = [[1.0]]
CONTROL = [[1.0]]
SENSOR = [[1.0]]
FIXED_R = [[0.3]]  # the SAME on both panels
FIXED_Q = [[0.2]]
BELIEF = Belief(mean=[0.0], cov=[[0.4]])
GOAL = Preference(goal=[0.0], precision=[[0.15]])
ACTIONS = jnp.linspace(-2.5, 3.5, 400)
BUMP = 1.2  # Q(x) is noisiest here — internal processing is least precise


def _bump_q(x, params):
    """Q(x): a bump of high internal noise at params['bump']. Module-level fn."""
    gap = x[0] - params["bump"]
    return jnp.array(
        [
            [
                params["base"]
                + params["amp"] * jnp.exp(-(gap**2) / (2.0 * params["width"] ** 2))
            ]
        ]
    )


def _model(process_noise=None):
    return LinearGaussianModel(
        dynamics=DYNAMICS,
        sensor_model=SENSOR,
        dynamics_noise=FIXED_Q,
        sensor_noise=FIXED_R,
        prior=BELIEF,
        control=CONTROL,
        process_noise=process_noise,
    )


def _sweep(model):
    prag, epi, g = [], [], []
    for a in ACTIONS:
        g_a, parts = expected_free_energy(model, BELIEF, jnp.array([a]), GOAL)
        g.append(float(g_a))
        prag.append(float(parts["pragmatic"]))
        epi.append(float(parts["epistemic"]))
    return np.array(prag), np.array(epi), np.array(g)


def _panel(ax, prag, epi, g, title, note):
    actions = np.asarray(ACTIONS)
    ax.plot(actions, prag, color="#e8833a", lw=2.2, label="pragmatic (goal cost)")
    ax.plot(actions, epi, color="#2ca6a4", lw=2.2, label="epistemic (info gain)")
    ax.plot(actions, g, color="#5b3a8a", lw=3.0, label="G = pragmatic − epistemic")
    a_g = actions[int(np.argmin(g))]
    a_prag = actions[int(np.argmin(prag))]
    ax.axvline(a_prag, color="#e8833a", ls=":", lw=1.6)
    ax.axvline(a_g, color="#5b3a8a", ls="--", lw=1.6)
    ax.scatter([a_g], [g.min()], color="#5b3a8a", zorder=5, s=45)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("action  (one-step move of state)")
    ax.annotate(
        note,
        xy=(0.5, -0.30),
        xycoords="axes fraction",
        ha="center",
        va="top",
        fontsize=9.5,
        color="#333333",
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)


def main():
    """Render the two-panel internal-noise figure to docs/assets/."""
    prag_f, epi_f, g_f = _sweep(_model())  # fixed Q (process_noise=None)
    q_well = CallableProcessNoise(
        _bump_q,
        {
            "bump": BUMP,
            "base": jnp.array(0.05),
            "amp": jnp.array(1.6),
            "width": jnp.array(0.6),
        },
    )
    prag_s, epi_s, g_s = _sweep(_model(process_noise=q_well))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    fig.suptitle(
        "Internal process noise breaks the collapse from the inside (R held fixed)",
        fontsize=13.5,
        fontweight="bold",
    )

    a_g_f = np.asarray(ACTIONS)[int(np.argmin(g_f))]
    _panel(
        ax_l,
        prag_f,
        epi_f,
        g_f,
        "Fixed Q — collapse (ADR-003)",
        f"epistemic is flat → argmin G ({a_g_f:.2f}) tracks the goal.\n"
        "Σ⁺ and S do not move with the action.",
    )
    ax_l.set_ylabel("nats")

    a_g_s = np.asarray(ACTIONS)[int(np.argmin(g_s))]
    ax_r.axvline(BUMP, color="#999999", ls="-", lw=1.0, alpha=0.7)
    _panel(
        ax_r,
        prag_s,
        epi_s,
        g_s,
        "State-dependent Q(x) — the internal route",
        f"epistemic curves through Q(μ⁺) (grey = noisiest x) → argmin G ({a_g_s:.2f}) "
        "shifts.\nR is the same constant as on the left; the effect is internal.",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
