"""Figure: the EFE epistemic collapse, and how a state-dependent sensor breaks it.

Sweeps a one-step action and plots the EFE decomposition `G = pragmatic − epistemic`
for two sensors:

- LEFT (fixed sensor): the epistemic term is dead-flat across actions, so `G`'s
  minimum is driven entirely by the pragmatic term — `argmin G == argmin pragmatic`.
  This is ADR-003's collapse: under a fixed linear-Gaussian sensor, EFE selection
  reduces to LQR and there is no information-seeking.
- RIGHT (state-dependent sensor): a "precision well" makes the sensor sharp near a
  beacon away from the goal. The epistemic term now *curves* (peaks at the beacon),
  and `G`'s minimum is pulled off the goal toward the information — the agent
  detours. This is why v0.3 exists.

The right-hand sensor is a real `CallableSensor` (Phase 2a) supplying a
position-dependent `R(x)` to `expected_free_energy` through the `gaussianize` seam.

Run: `uv run python examples/efe_collapse_figure.py`
Output: docs/assets/efe_collapse.png
"""

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from cpomdp.efe import expected_free_energy
from cpomdp.observation import CallableSensor
from cpomdp.selection import Preference
from cpomdp.types import Belief, LinearGaussianModel

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "efe_collapse.png"

# A 1-D single integrator: action moves the observed position directly, so a
# one-step sweep tells a clean story (μ⁺ = μ + a, observation = position).
DYNAMICS = [[1.0]]
CONTROL = [[1.0]]
SENSOR = [[1.0]]
PROCESS_NOISE = [[0.01]]
FIXED_NOISE = [[0.3]]
BELIEF = Belief(mean=[0.0], cov=[[2.0]])
GOAL = Preference(goal=[0.0], precision=[[0.4]])  # prefer to observe position 0
ACTIONS = jnp.linspace(-2.0, 4.0, 400)
BEACON = 1.5  # the sensor is sharpest here — away from the goal at 0


def _precision_well_noise(x, params):
    """R(x) for a 'precision well': dips to ``r_lo`` at the beacon, rises to ``r_hi``.

    Module-level (jit-safe) so it can ride in ``CallableSensor``'s static aux; all
    tunables live in ``params``.
    """
    pos = x[0]
    falloff = 1.0 - jnp.exp(
        -((pos - params["beacon"]) ** 2) / (2.0 * params["width"] ** 2)
    )
    r = params["r_lo"] + (params["r_hi"] - params["r_lo"]) * falloff
    return jnp.array([[r]])


def _model(observation=None):
    return LinearGaussianModel(
        dynamics=DYNAMICS,
        sensor_model=SENSOR,
        dynamics_noise=PROCESS_NOISE,
        sensor_noise=FIXED_NOISE,
        prior=BELIEF,
        control=CONTROL,
        observation=observation,
    )


def _sweep(model):
    """Pragmatic, epistemic, and G over the action sweep (eager, no jit)."""
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
    ax.set_xlabel("action  (one-step move of observed position)")
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
    """Render the two-panel collapse/detour figure to docs/assets/."""
    prag_f, epi_f, g_f = _sweep(_model())  # fixed sensor (observation=None)
    well = CallableSensor(
        sensor_model=[[1.0]],
        noise_fn=_precision_well_noise,
        params={"beacon": BEACON, "width": 0.6, "r_lo": 0.02, "r_hi": 0.8},
    )
    prag_s, epi_s, g_s = _sweep(_model(observation=well))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12.5, 5.2), sharex=True)
    fig.suptitle(
        "Expected Free Energy: epistemic collapse, and how a state-dependent "
        "sensor breaks it",
        fontsize=13.5,
        fontweight="bold",
    )

    a_prag_f = np.asarray(ACTIONS)[int(np.argmin(prag_f))]
    a_g_f = np.asarray(ACTIONS)[int(np.argmin(g_f))]
    _panel(
        ax_l,
        prag_f,
        epi_f,
        g_f,
        "Fixed sensor — collapse (ADR-003)",
        f"epistemic is flat → argmin G ({a_g_f:.2f}) == argmin pragmatic "
        f"({a_prag_f:.2f}).\nEFE reduces to LQR; no information-seeking.",
    )
    ax_l.set_ylabel("nats")

    a_prag_s = np.asarray(ACTIONS)[int(np.argmin(prag_s))]
    a_g_s = np.asarray(ACTIONS)[int(np.argmin(g_s))]
    ax_r.axvline(BEACON, color="#999999", ls="-", lw=1.0, alpha=0.7)
    _panel(
        ax_r,
        prag_s,
        epi_s,
        g_s,
        "State-dependent sensor — the detour",
        f"epistemic peaks at the beacon (grey) → argmin G ({a_g_s:.2f}) detours "
        f"off the goal\n(argmin pragmatic {a_prag_s:.2f}) toward information.",
    )

    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
