import numpy as np
import scipy.optimize as opt
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score
)


class SimulateSurvival:
    def __init__(self, n_sample, hazard_ratio, baseline_hazard, rnd):
        self.n_sample = n_sample
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






