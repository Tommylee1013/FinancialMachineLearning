import numpy as np
import statsmodels.api as sm
import pandas as pd
from filterpy.kalman import FixedLagSmoother, KalmanFilter
from filterpy.common import Q_discrete_white_noise
class LocalLinearTrend(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        k_states = k_posdef = 2
        super(LocalLinearTrend, self).__init__(
            endog, k_states = k_states, k_posdef = k_posdef,
            initialization = "approximate_diffuse",
            loglikelihood_burn = k_states
        )
        self.ssm['design'] = np.array([1, 0])
        self.ssm['transition'] = np.array([[1, 1],
                                           [0, 1]])
        self.ssm['selection'] = np.eye(k_states)
        self._state_cov_idx = ("state_cov",) + np.diag_indices(k_posdef)
    @property
    def param_names(self):
        return ["sigma2.measurement", "sigma2.level", "sigma2.trend"]
    @property
    def start_params(self):
        return [np.std(self.endog)]*3
    def transform_params(self, unconstrained):
        return unconstrained ** 2
    def untransform_params(self, constrained):
        return constrained ** 0.5
    def update(self, params, *args, **kwargs):
        params = super(LocalLinearTrend, self).update(params, *args, **kwargs)
        # Observation covariance
        self.ssm['obs_cov',0,0] = params[0]
        # State covariance
        self.ssm[self._state_cov_idx] = params[1:]
def kalman_filter(data, noise : int = 1 , Q : float = 0.001) -> pd.DataFrame :

    fk = KalmanFilter(dim_x=2, dim_z=1)

    fk.x = np.array([0., 1.])

    fk.F = np.array([[1., 1.],
                     [0., 1.]])

    fk.H = np.array([[1., 0.]])
    fk.P*= 10.
    fk.R = noise
    fk.Q = Q_discrete_white_noise(dim=2, dt=1., var=Q)
    zs = data
    mu, cov, _, _ = fk.batch_filter(zs)
    M, P, C, _ = fk.rts_smoother(mu, cov)

    result = pd.DataFrame({"Measurement": zs,
                           "RTS Smoother": M[:, 0],
                           "Kalman Filter": mu[:, 0]})
    return result
def kalman_smoother(data, N : int = 4) -> pd.DataFrame :
    fls = FixedLagSmoother(dim_x=2, dim_z=1, N=N)
    fls.x = np.array([0., .5])
    fls.F = np.array([[1., 1.],
                      [0., 1.]])

    fls.H = np.array([[1., 0.]])
    fls.P *= 200
    fls.R *= 5.
    fls.Q *= 0.001

    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., .5])
    kf.F = np.array([[1., 1.],
                     [0., 1.]])
    kf.H = np.array([[1., 0.]])
    kf.P *= 200
    kf.R *= 5.
    kf.Q = Q_discrete_white_noise(dim=2, dt=1., var=0.001)

    zs = data
    nom = np.array([t / 2. for t in range(len(zs))])

    for z in zs:
        fls.smooth(z)

    kf_x, _, _, _ = kf.batch_filter(zs)
    x_smooth = np.array(fls.xSmooth)[:, 0]

    fls_res = abs(x_smooth - nom)
    kf_res = abs(kf_x[:, 0] - nom)

    result = pd.DataFrame({"Measurement": zs,
                           "FL Smoother": x_smooth,
                           "Kalman Filter": kf_x[:, 0]},
                          index=data.index)
    return result