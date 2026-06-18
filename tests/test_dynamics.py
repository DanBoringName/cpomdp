"""CallableProcessNoise: state-dependent process noise Q(x) (Phase 2d).

The internal-noise dual of CallableSensor. ``params`` is a pytree leaf (grad-able),
``q_fn`` is static aux. Unlike CallableSensor it can't probe its own ``(n, n)`` shape
at construction (it has no dimension info); the *model* validates that when it knows
``n``. See test_types.py for the model-level shape rejection.
"""

import jax
import jax.numpy as jnp
import numpy as np

from cpomdp.dynamics import CallableProcessNoise, DynamicsNoise


def _q_well(x, params):
    """Q(x): internal process noise, low near 0, growing with position²."""
    return jnp.array([[params["base"] + params["slope"] * x[0] ** 2]])


def _params():
    return {"base": jnp.array(0.05), "slope": jnp.array(0.4)}


def _process_noise():
    return CallableProcessNoise(q_fn=_q_well, q_params=_params())


class TestCallableProcessNoise:
    def test_at_varies_with_state(self):
        pn = _process_noise()
        q0 = pn.noise_at(jnp.array([0.0]))
        q1 = pn.noise_at(jnp.array([2.0]))
        assert float(q0[0, 0]) < float(
            q1[0, 0]
        )  # noisier internal dynamics away from 0
        assert pn.is_fixed is False

    def test_satisfies_protocol(self):
        assert isinstance(_process_noise(), DynamicsNoise)

    def test_pytree_round_trip_preserves_q_fn(self):
        pn = _process_noise()
        leaves, treedef = jax.tree_util.tree_flatten(pn)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, CallableProcessNoise)
        np.testing.assert_array_equal(
            restored.noise_at(jnp.array([1.0])), pn.noise_at(jnp.array([1.0]))
        )

    def test_grad_wrt_params_flows(self):
        def scalar(params):
            return jnp.trace(
                CallableProcessNoise(_q_well, params).noise_at(jnp.array([1.0]))
            )

        grads = jax.tree_util.tree_leaves(jax.grad(scalar)(_params()))
        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
