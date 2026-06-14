import numpy as np

from cpomdp.types import Belief, LinearGaussianModel


class KalmanBackend:
    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        steady_state: bool = False,
        tol: float = 1e-12,
        max_iter: int = 1000,
    ) -> None:
        self.model = model
        self.steady_state = steady_state
        if steady_state:
            self._steady_gain, self._steady_cov = self._converge_to_steady_state(
                tol, max_iter
            )

    def infer_states(
        self,
        observation: np.ndarray,
        prior: Belief,
        action: np.ndarray | None = None,
    ) -> Belief:
        model = self.model

        # Control (action) handling:
        #   no control matrix     -> pure filtering, action is ignored
        #   control but no action -> caller error
        #   control and action    -> action pushes the predicted state
        if model.control is None:
            control_term = 0.0
        elif action is None:
            raise ValueError(
                "this model has a control matrix; infer_states requires an action"
            )
        else:
            control_term = model.control @ np.asarray(action, dtype=float)

        if self.steady_state:
            gain, cov_post = self._steady_gain, self._steady_cov  # frozen
        else:
            gain, cov_post = self._gain_and_posterior_cov(prior.cov)

        mean_pred = model.dynamics @ prior.mean + control_term

        # prediction_error: observation minus predicted observation
        # ("innovation" in Kalman terms). Its covariance is the gain denominator.
        prediction_error = observation - model.sensor_model @ mean_pred

        mean_post = mean_pred + gain @ prediction_error

        return Belief(mean=mean_post, cov=cov_post)

    def _gain_and_posterior_cov(
        self, prior_cov: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """One step's covariance maths: given a prior covariance, return
        ``(gain, posterior_cov)``. Data-independent — shared by the per-step
        filter and the steady-state warmup."""
        model = self.model
        cov_pred = model.dynamics @ prior_cov @ model.dynamics.T + model.dynamics_noise
        prediction_error_cov = (
            model.sensor_model @ cov_pred @ model.sensor_model.T + model.sensor_noise
        )
        gain = np.linalg.solve(prediction_error_cov, b=model.sensor_model @ cov_pred).T
        cov_post = (np.eye(model.n_states) - gain @ model.sensor_model) @ cov_pred

        return gain, cov_post

    def _converge_to_steady_state(
        self, tol: float, max_iter: int
    ) -> tuple[np.ndarray, np.ndarray]:
        model = self.model
        cov = model.prior.cov
        for _ in range(max_iter):
            gain, cov_post = self._gain_and_posterior_cov(cov)
            if np.allclose(cov, cov_post, atol=tol, rtol=0.0):
                return gain, cov
            cov = cov_post

        raise RuntimeError(
            f"steady-state covariance did not converge in {max_iter} iterations; "
            "the model may not be stabilisable/detectable. "
            "Use steady_state=False for the full per-step filter."
        )
