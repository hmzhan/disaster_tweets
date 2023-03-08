import numpy as np
import scipy.optimize as opt
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score
)


class SimulateSurvival:
    def __init__(self, n_samples, hazard_ratio, baseline_hazard, rnd):
        self.n_samples = n_samples
        self.hazard_ratio = hazard_ratio
        self.baseline_hazard = baseline_hazard
        self.rnd = rnd

    def _generate_marker(self):
        """
        Generate markers for simulation study
        :return: simulated data
        """
        X = self.rnd.randn(self.n_samples)
        hazard_ratio = np.array([self.hazard_ratio])
        logits = np.dot(X, np.log(hazard_ratio))

        u = self.rnd.uniform(size=self.n_samples)
        time_event = -np.log(u) / (self.baseline_hazard * np.exp(logits))

        X = np.squeeze(X)
        actual = concordance_index_censored(np.ones(self.n_samples, dtype=bool), time_event, X)
        return X, time_event, actual[0]

    def get_observed_time(self, time_event, x):
        rnd_cens = np.random.RandomState(0)
        time_censor = rnd_cens.uniform(high=x, size=self.n_samples)
        event = time_event < time_censor
        time = np.where(event, time_event, time_censor)
        return event, time

    def censoring_amount(self, time_event, percentage_cens, x):
        event, _ = self.get_observed_time(time_event, x)
        cens = 1.0 - event.sum() / event.shape[0]
        return (cens - percentage_cens)**2

    def generate_survival_data(self):
        X, time_event, actual_c = self._generate_marker()
        res = opt.minimize_scalar(censoring_amount,
                                  method="bounded",
                                  bounds=(0, time_event.max()))

        event, time = get_observed_time(res.x)

        # upper time limit such that the probability
        # of being censored is non-zero for `t > tau`
        tau = time[event].max()
        y = Surv.from_arrays(event=event, time=time)
        mask = time < tau
        X_test = X[mask]
        y_test = y[mask]
        return X_test, y_test, y, actual_c










