import numpy as np

from cpomdp.types import Belief, LinearGaussianModel


class KalmanBackend:
    def __init__(self, model: LinearGaussianModel) -> None:
        self.model = model
        # self.K_inf = solve_riccati(model.A, model.C, model.C, model.Q, model.R)

    def infer_states(self, observation, prior, action=None) -> Belief:
        m = self.model
        identity = np.eye(m.n_states)

        # Control (action) handling:
        #   no control matrix     -> pure filtering, action is ignored
        #   control but no action -> caller error
        #   control and action    -> action pushes the predicted state
        if m.control is None:
            control_term = 0.0
        elif action is None:
            raise ValueError(
                "this model has a control matrix; infer_states requires an action"
            )
        else:
            control_term = m.control @ np.asarray(action, dtype=float)

        mean_pred = m.dynamics @ prior.mean + control_term
        cov_pred = m.dynamics @ prior.cov @ m.dynamics.T + m.dynamics_noise

        # prediction_error: observation minus predicted observation
        # ("innovation" in Kalman terms). Its covariance is the gain denominator.
        prediction_error = observation - m.sensor_model @ mean_pred
        prediction_error_cov = (
            m.sensor_model @ cov_pred @ m.sensor_model.T + m.sensor_noise
        )
        gain = np.linalg.solve(prediction_error_cov, m.sensor_model @ cov_pred).T
        mean_post = mean_pred + gain @ prediction_error
        cov_post = (identity - gain @ m.sensor_model) @ cov_pred

        return Belief(mean=mean_post, cov=cov_post)