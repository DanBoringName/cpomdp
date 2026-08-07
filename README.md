# cpomdp — Continuous Active Inference for Python

[![PyPI](https://img.shields.io/pypi/v/cpomdp.svg)](https://pypi.org/project/cpomdp/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21334562-blue.svg)](https://doi.org/10.5281/zenodo.21334562)
[![Python](https://img.shields.io/pypi/pyversions/cpomdp.svg)](https://pypi.org/project/cpomdp/)
[![CI](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml/badge.svg)](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml)
[![coverage](https://cpomdp.inferogenesis.com/assets/coverage.svg)](https://github.com/inferogenesis/cpomdp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/inferogenesis/cpomdp/blob/main/LICENSE)

Continuous active inference for Python. The continuous-state sibling of [pymdp](https://github.com/infer-actively/pymdp).

pymdp is great, but it speaks in discrete states. A lot of the world isn't discrete. Positions, velocities, temperatures, the kinds of things you'd actually want an agent to track and steer, don't come in neat little categories. cpomdp fills that gap. You hand it a linear-Gaussian model of how the world moves and what you can see of it, and you get back an agent that perceives and acts in the same `infer_states` / `sample_action` loop pymdp users already know.

That's the whole idea: keep the pymdp muscle memory, swap the discrete machinery underneath for continuous.

Full documentation — API reference and guides — lives at [cpomdp.inferogenesis.com](https://cpomdp.inferogenesis.com/).

## What cpomdp does

**Exact continuous perception.** The mean dynamics are linear and the noise is Gaussian, so the filter is an exact Kalman filter. No sampling and no variational gap. No approximation to tune. The same closed form runs whether you are tracking or acting.

**Epistemics that survive the linear-Gaussian collapse.** Under a *fixed* linear-Gaussian sensor the epistemic term of expected free energy is identical for every policy (Koudahl, Kouw & de Vries 2021). cpomdp deviates from that model by the smallest amount that avoids the collapse. It lets the *noise* depend on the state. The mean stays linear. That noise is a state-dependent sensor `R(x)` (Corva 2026), or state-dependent process noise `Q(x)`. The action then reaches the posterior covariance, so the epistemic term moves with it. Information-seeking behaviour is available in a regime where it is proved away.

**Multi-step expected free energy, searched and certified.** EFE over an H-step horizon (`cpomdp.efe.policy_efe`), plus an exhaustive search across the entire space of action sequences `A^H` (`cpomdp.enumeration.EnumeratedEfeSearch`) that hands back a `CompletenessCertificate`. So "this is the best plan" is *decided* over the declared action set rather than sampled from it, and the object that says so is checkable. Receding-horizon and open-loop selectors wrap the search, and both report the honest per-cycle cost `|A|^H · H` rather than a grid's much smaller number.

With `R(x)` alive a short-horizon agent walks straight past the information. Stretch the horizon and it detours to collect it. Curiosity needs both.

## Example

Four bacilli seeking food in the same world — the continuous-state answer to pymdp's mouse-seeking-cheese, now with the **epistemic** term v0.3 adds. The twist: the food's position is **hidden**, and a **beacon** marks where the agent can *see* it. Visiting the beacon doesn't sharpen where the agent thinks *it* is — it sharpens where it thinks the *food* is, which it can't act on directly. That makes the information genuinely **instrumental**: resolving it changes where the agent then heads. Each body sits at its **true** hidden state; the blue `+` is where it believes it is, the diamond is where it believes the food is (both with their uncertainty ellipses), and the star is the food's true, hidden location. The four differ in **one number only** — the **goal precision Λ** each is built with. They all minimise the same Expected Free Energy `G = pragmatic − epistemic`; because the pragmatic (goal) term scales with Λ while the epistemic (information) term doesn't, Λ alone tips the balance: **classic LQR** and a **sharp Λ** beeline to the agent's current food guess and never detour; a **balanced Λ** detours to the beacon, learns where the food really is, *then* heads there with confidence; a **weak Λ** is so over-curious it parks at the beacon and never eats. One real knob — the precision you'd actually pass — four behaviours.

![Four bacilli learning where the food is, under different goal precisions Λ, via continuous active inference](docs/assets/bacillus_uncertain_food.gif)

Reproduce it with [`examples/bacillus_uncertain_food.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/bacillus_uncertain_food.py) (`pip install "cpomdp[examples]"`).

### How far ahead before information is worth a detour?

An agent on an open plane wants a goal it cannot locate. Its prior points the wrong way. A beacon well off that line is the only thing that can say where the goal really is. Walking there costs ground.

Run the same world once per planning horizon. At `H = 2` the agent walks straight to the spot it already believed in and settles there. It never checks. At `H = 14` it walks *away* from that spot, reads the beacon, finds out it was wrong, and then goes to the real goal. Only `H` changed.

![One agent run once per planning horizon: at short horizons it settles where its prior said, and past a crossing it detours to the beacon, learns where the goal really is, and goes there](docs/assets/crossover_horizon.gif)

The margin between the two plans, `ΔG(H) = G(detour) − G(direct)` in nats, crosses zero exactly once. The epistemic pull is flat, because sensing once is worth what sensing once is worth. The pragmatic gradient decays under it. Freeze `R` at a constant and the sweep never crosses at any horizon, so the behaviour belongs to the state-dependent sensor and the horizon together.

Reproduce it with [`examples/crossover_horizon_figure.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/crossover_horizon_figure.py).

**Why the horizon is the question.** [State-dependent observation noise reintroduces epistemic value in linear-Gaussian active inference](https://arxiv.org/abs/2607.20306) (Corva 2026) establishes that `R(x)` makes the epistemic term non-constant, so a linear-Gaussian agent can be curious at all. Non-constant is not the same as decision-changing. The horizon at which curiosity starts changing which plan an agent picks is a separate question, and it has a measured answer.

**The answer is one-dimensional, and it is proven.** On the corridor cue task, `H* = 7`: the first horizon whose argmin is a two-phase sense-then-commit walk rather than a direct reach. That comes from enumerating every sequence in the declared action set `{0, ±1, ±2}` to depth `H`, so it decides the flip rather than sampling for it, and the search returns a `CompletenessCertificate` saying how many policies it was obliged to visit and how many it did. Zero the epistemic term and the crossing moves out to `H ≈ 10`, which is what makes the pull load-bearing rather than incidental. The numbers are registered in [`warrant_numbers.md`](https://github.com/inferogenesis/cpomdp/blob/main/warrant_numbers.md), the model is [`examples/ffg/crossover.py`](https://github.com/inferogenesis/cpomdp/blob/main/examples/ffg/crossover.py), and `tests/test_example_checks.py::test_crossover_check` asserts it on every merge and release. The `7` is an upper bound, because the declared set clips the reach at `−2` while reaching the goal from the start takes `−3`. On the wider `{−3…2}` the flip is at 6. That wider number is computed rather than certified. No completeness certificate reaches a report and no test asserts it, so it is stated here as a qualifier on the `7` rather than as a result of its own. Certifying it, and measuring whether wider sets still move it, is [issue #65](https://github.com/inferogenesis/cpomdp/issues/65).

**The plane above is the readable version, not the proof.** It contrasts two named plans instead of searching, so its crossing is an exact statement about those two plans and says nothing about the argmin over all plans. Its `H = 7` and the corridor's `H* = 7` are different quantities on different models that happen to coincide.

More in the [examples gallery](https://github.com/inferogenesis/cpomdp/blob/main/examples/README.md), including the [FFG examples](https://github.com/inferogenesis/cpomdp/blob/main/examples/ffg/README.md), where a branch-coupled state-dependent sensor resolves a hidden context and can't be flattened to a Kalman filter.

## Install

```bash
pip install cpomdp
```

Or the latest from source:

```bash
pip install git+https://github.com/inferogenesis/cpomdp
```

That's all you need for normal use. There's also an optional RxInfer (Julia) backend that the test suite leans on as a correctness oracle. You almost certainly don't need it, but if you want it:

```bash
pip install "cpomdp[rxinfer]"
```

It pulls in a Julia bridge and bootstraps itself the first time you use it.

## Quickstart

Here's an agent steering a point mass to a target. It can push the mass and it can see where the mass is, but it never sees the velocity. The filter has to work that out from how the position moves.

```python
import jax.numpy as jnp
from cpomdp import Agent, Belief, LinearGaussianModel, StateGoal

# State is [position, velocity]. A push changes velocity, velocity carries
# position along, and we only ever observe position (through a noisy sensor).
dt = 0.1
model = LinearGaussianModel(
    dynamics=[[1, dt], [0, 1]],          # velocity carries position along
    control=[[0], [dt]],                 # a push nudges velocity
    sensor_model=[[1, 0]],               # we observe position only
    dynamics_noise=jnp.eye(2) * 1e-6,
    sensor_noise=[[1e-2]],
    prior=Belief(mean=[0, 0], cov=jnp.eye(2)),
)

# Tell it where to go: sit still at position 1.
agent = Agent(model, StateGoal([1.0, 0.0]))

true_state = jnp.array([0.0, 0.0])
for _ in range(100):
    obs = model.sensor_model @ true_state            # what the agent gets to see
    agent.infer_states(obs)                           # perceive
    action = agent.sample_action()                    # act
    true_state = model.dynamics @ true_state + model.control @ action

print(jnp.round(agent.belief.mean, 3))   # ≈ [1, 0]
```

Run that and the belief lands on `[1, 0]`. The agent worked out it was at position 1 and sitting still, which is exactly where we asked it to go, and it did it without ever seeing the velocity it had to control.

## The pymdp parallel

If you've used pymdp, the loop is the same and most of the names are too. Four carry over verbatim:

> `Agent` · `qs` · `infer_states` · `sample_action`

(`qs` is a read-only alias for `belief`, cpomdp's canonical name — so `agent.qs` and `agent.belief` are the same posterior. Use whichever your fingers reach for.)

Only two things are spelled differently:

| pymdp | cpomdp                          | what it is                           |
| ----- | ------------------------------- | ------------------------------------ |
| `C`   | `StateGoal` / `ObservationGoal` | the goal you pursue, and how sharply |
| `D`   | `model.prior`                   | belief before you've seen anything   |

One honest difference in behaviour. `sample_action` here is deterministic, not a sample from a policy posterior. For a linear-Gaussian sensor the action that minimises expected free energy turns out to be exactly the LQR optimum, so there's a single best action and that's what comes back. Same loop, exact answer. The reasoning is in [DECISIONS.md](https://github.com/inferogenesis/cpomdp/blob/main/DECISIONS.md) (ADR-003) if you want it.

## Just want to track, not act?

A model with no `control` matrix is a pure tracker. Drop the goal and `infer_states` still folds in observations and sharpens the belief, while `sample_action` stops you — there's nothing to steer toward, and nothing to steer with.

```python
tracker = LinearGaussianModel(        # no control matrix -> pure tracking
    dynamics=[[1, dt], [0, 1]],
    sensor_model=[[1, 0]],
    dynamics_noise=jnp.eye(2) * 1e-6,
    sensor_noise=[[1e-2]],
    prior=Belief(mean=[0, 0], cov=jnp.eye(2)),
)
agent = Agent(tracker)                # no objective
agent.infer_states([0.5])             # perceiving is fine
agent.sample_action()                 # ValueError: this Agent has no objective ...
```

## What's in the box

| you want to | reach for |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| perceive, exactly | `Agent.infer_states`, `KalmanBackend` |
| reach a target state | `StateGoal` with `LQRSelector` |
| act on expected free energy, one step | `ObservationGoal` with `EFESelector`, `expected_free_energy` |
| ...over an H-step horizon | `cpomdp.efe.policy_efe`, `policy_efe_trace` |
| ...searched exhaustively, with a certificate | `cpomdp.enumeration`, `EnumeratedEfeSearch`, `RecedingHorizonSelector`, `OpenLoopSelector` |
| sense more sharply in some places than others | `CallableSensor`, state-dependent `R(x)` |
| diffuse more in some states than others | `CallableProcessNoise`, state-dependent `Q(x)` |
| declare a branching model | `CouplingGraph`, `Coupling`, `CouplingGraphBackend` |
| ask whether a state-dependent sensor earns its keep | `probe_model` → `SensorReport` |
| check a rollout stayed well conditioned | `cpomdp.diagnostics.rollout_conditioning` |
| perceive but never act | a model with no `control`, `Agent(model)` alone |
| re-plan every step, closed loop | `cpomdp.enumeration.RecedingHorizonSelector` (`replan_interval = 1`) |
| commit to one plan and execute it | `cpomdp.enumeration.OpenLoopSelector` (`replan_interval = H`) |
| swap the inference engine | the `InferenceBackend` protocol, or `cpomdp.backends.rxinfer.RxInferBackend` |

The state-dependence is in the *noise*. The mean stays linear. Genuinely **nonlinear sensors**, a curved `g(x)` needing a second-order moment match, are the next step and are not here yet.

## Swappable backends

You can swap the inference engine if you want to. `KalmanBackend` is the default and does the real work; `RxInferBackend` re-derives the same answers through Julia and exists mainly so the fast path has something independent to check itself against. Both sit behind the `InferenceBackend` protocol, so you can write your own.

## Status

Still pre-1.0. v0.4.4 is the current release. It added the multi-step slice of expected free energy: horizon rollouts, exhaustive search with completeness certificates, and the horizon at which a planner stops reaching and starts sensing. The demos were rebuilt on top of it. The crossover now also runs on the flat Kalman/EFE route, the cue-maze task works in any number of dimensions, and the shared plotting machinery lives in `examples/gallery.py`. The `Agent` / `infer_states` / `sample_action` surface has been stable since v0.3 and is what I am trying to hold steady. If you have a request or a suggestion that would make that front-facing API more usable, please open a GitHub issue. Until 1.0 a minor version is where breaking changes can land.

## Development

I designed and built cpomdp — the architecture, the conditionally-linear-Gaussian formulation, the API, and every decision in [DECISIONS.md](https://github.com/inferogenesis/cpomdp/blob/main/DECISIONS.md) are mine. The design draws on my day-to-day work as a full-time software engineer and on hands-on expertise integrating and developing large machine-learning models at scale using event-driven microservice architecture.

I used an AI coding assistant (Claude Opus-4.8) as a tool under close review: to draft docstrings, probe for edge cases and candidate bugs, and expand the test suite, including adversarial ones. Everything it produced I read, checked, and approved before it landed. None of it is taken on trust — the numbers are validated independently against the RxInfer (Julia) and analytic NumPy oracles described above. Correctness rests on those checks, not on the tool that helped write the code.

## Contributions

If you would like to contribute either your dev time or help steer the direction of the toolbox, please add a GitHub issue or discussion thread. I am monitoring this repository closely and would love to collaborate. [CONTRIBUTING.md](CONTRIBUTING.md) covers the setup, the hooks, and the bar a PR has to clear.

If you notice a better method in something I've already done or are just curious and want to chat I am more than happy to talk through my decision processes and improve on my work. I intend to blog my construction of cpomdp provided it doesn't interfere with developing it.

## Acknowledgements

Thanks to **Kevin Backhouse** (Postgraduate Researcher in Cognitive Neuroscience, Durham University) for guidance on the active-inference formulation, collaboration on related discrete generative-model projects, and for being a consistent sounding board throughout the design of this work.
