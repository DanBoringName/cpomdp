import numpy as np
from numpy.typing import ArrayLike

from cpomdp.types import LinearGaussianModel


class LQRController:
    """Steady-state LQR action selection — the action-side dual of the filter.

    Where the Kalman filter front-loads perception (solve the estimation Riccati
    once for the steady-state gain ``K∞``, then ``mean += K∞·prediction_error``),
    this front-loads action: solve the dual *control* Riccati once for ``L∞``,
    then ``action = -L∞·(mean − goal)``. Both gains are data-independent, both are
    computed at construction, and together they are LQG (see RESEARCH.md).

    The load-bearing claim (ADR-003) is that LQR *is* active inference here, not a
    substitute for it. For a fixed linear-Gaussian sensor the covariance recursion
    is control-independent, so Expected Free Energy's epistemic term is identical
    for every action and drops out of the argmin; EFE-minimising selection reduces
    to its pragmatic term, and the pragmatic term under a Gaussian preference is a
    quadratic cost whose optimum is exactly LQR. The epistemic term only re-enters
    once sensing depends on the state or action — out of scope for v0.1.

    The two cost matrices are named by role, not by LQR's traditional ``Q``/``R``:
    those letters already mean the noise covariances on the model
    (``dynamics_noise``/``sensor_noise``), and reusing them here is the exact
    collision ADR-003 warns about.

    Args:
        model: The linear-Gaussian model to act in. Must carry a ``control``
            matrix — there is nothing to act with otherwise.
        state_cost: How much deviation from the goal costs, an ``(n, n)`` matrix.
            Heavier ``state_cost`` buys a more aggressive controller. (LQR's
            ``Q``; here the precision of the preference over states.)
        control_cost: How much action costs, a ``(p, p)`` matrix. Heavier
            ``control_cost`` buys a gentler one. (LQR's ``R``; the agent's effort
            penalty.)
        tol: Absolute tolerance on successive cost-to-go iterates; convergence is
            declared when they stop moving by more than this.
        max_iter: Iteration cap before the Riccati recursion is declared to have
            failed to converge.

    Raises:
        ValueError: If the model has no ``control`` matrix, or a cost matrix does
            not match the state/action dimensions.
        RuntimeError: If the control Riccati does not converge within ``max_iter``
            — typically because ``(dynamics, control)`` is not stabilisable.
    """

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        state_cost: ArrayLike,
        control_cost: ArrayLike,
        tol: float = 1e-12,
        max_iter: int = 1000,
    ) -> None:
        if model.control is None:
            raise ValueError(
                "LQR needs an action channel: the model has no control matrix, "
                "so there is nothing to act with."
            )
        self.model = model
        self._state_cost = np.asarray(state_cost, dtype=float)
        self._control_cost = np.asarray(control_cost, dtype=float)

        n, p = model.n_states, model.n_controls
        if self._state_cost.shape != (n, n):
            raise ValueError(
                f"state_cost must be {n}x{n} to match the {n}-D state, "
                f"got shape {self._state_cost.shape}"
            )
        if self._control_cost.shape != (p, p):
            raise ValueError(
                f"control_cost must be {p}x{p} to match the {p}-D action, "
                f"got shape {self._control_cost.shape}"
            )

        self._gain = self._converge_to_steady_state(tol, max_iter)

    @property
    def gain(self) -> np.ndarray:
        """The steady-state feedback gain L∞, shape (p, n)."""
        return self._gain

    def action(self, mean: np.ndarray, goal: ArrayLike) -> np.ndarray:
        """The action that drives the estimated state toward ``goal``.

        One matrix-vector product, ``-L∞·(mean − goal)`` — all the work was
        front-loaded into ``L∞`` at construction, so there is no optimisation in
        the loop. The ``mean − goal`` shift turns the regulator (which drives its
        state to zero) into a controller that drives the state to ``goal``.

        Args:
            mean: The current belief mean — the best estimate of the state,
                shape ``(n,)``.
            goal: The state to steer toward, shape ``(n,)``. It must be an
                equilibrium the dynamics can hold at zero action; aim at a
                non-equilibrium and a steady-state offset is left behind.

        Returns:
            The action, shape ``(p,)``.

        Raises:
            ValueError: If ``goal`` is not a 1-D vector of length ``n``.
        """
        # self._gain : (p, n) L∞;  mean, goal : (n,);  returns (p,)
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (self.model.n_states,):
            raise ValueError(
                f"goal must be a 1-D vector of length {self.model.n_states} "
                f"(the state dimension), got shape {goal.shape}"
            )
        return -self._gain @ (mean - goal)

    def _converge_to_steady_state(self, tol: float, max_iter: int) -> np.ndarray:
        """Iterate the control Riccati recursion to its fixed point for ``L∞``.

        The exact dual of ``KalmanBackend._converge_to_steady_state``. The filter
        iterates a *covariance* forward until it stops moving; this iterates a
        *cost-to-go* — the matrix ``P`` of the quadratic value function
        ``V(state) = stateᵀ·P·state`` — until it stops moving. Starting from
        ``state_cost``, each step applies Bellman's equation::

            P ← state_cost + Aᵀ P A − (Aᵀ P B)(control_cost + Bᵀ P B)⁻¹(Bᵀ P A)

        "the cost from here = what I pay now + the cost from wherever the dynamics
        carry me, minus what acting optimally buys back." For a stabilisable
        ``(A, B)`` this converges to the unique fixed point ``P∞`` (the solution
        of the discrete algebraic Riccati equation), from which the steady-state
        gain follows::

            L∞ = (control_cost + Bᵀ P∞ B)⁻¹ (Bᵀ P∞ A)

        (A=dynamics, B=control.) The ``(control_cost + Bᵀ P B)`` term is solved
        against with ``np.linalg.solve`` rather than inverted explicitly, for the
        same numerical reason the filter solves against its innovation covariance.

        Returns:
            The steady-state gain ``L∞``, shape ``(p, n)``.

        Raises:
            RuntimeError: If the recursion has not converged within ``max_iter``.
        """
        dynamics = self.model.dynamics  # A  (n×n)
        assert self.model.control is not None  # guard lives in __init__; narrows type
        control = self.model.control  # B  (n×p)
        cost_to_go = self._state_cost  # P, starting at the running state cost (n×n)

        for _ in range(max_iter):
            # Bellman's equation, one sweep.
            dyn_cost_ctrl = dynamics.T @ cost_to_go @ control  # Aᵀ P B  (n×p)
            # curvature of the action cost — the dual of the Kalman innovation
            # covariance S, the denominator the gain is solved against (p×p)
            inner = self._control_cost + control.T @ cost_to_go @ control
            next_cost_to_go = (
                self._state_cost  # pay now
                + dynamics.T @ cost_to_go @ dynamics  # cost the dynamics carry forward
                - dyn_cost_ctrl
                @ np.linalg.solve(
                    inner, dyn_cost_ctrl.T
                )  # what optimal action buys back
            )

            if np.allclose(cost_to_go, next_cost_to_go, atol=tol, rtol=0.0):
                cost_to_go = next_cost_to_go
                break
            cost_to_go = next_cost_to_go
        else:
            raise RuntimeError(
                f"control Riccati did not converge in {max_iter} iterations; "
                "(dynamics, control) may not be stabilisable, so no steady-state "
                "gain exists."
            )

        inner = self._control_cost + control.T @ cost_to_go @ control
        return np.linalg.solve(inner, control.T @ cost_to_go @ dynamics)  # L∞  (p×n)
