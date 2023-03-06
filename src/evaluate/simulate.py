import numpy as np
import scipy.optimize as opt
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score
)


def generate_marker(n_samples, hazard_ratio, baseline_hazard, rnd):
    """
    Generate markers for simulation study
    :param n_samples: number of samples
    :param hazard_ratio: hazard ratio
    :param baseline_hazard: baseline hazard
    :param rnd: random number
    :return: simulated data
    """
    X = rnd.randn(n_samples)
    hazard_ratio = np.array([hazard_ratio])
    logits = np.dot(X, np.log(hazard_ratio))

    u = rnd.uniform(size=n_samples)
    time_event = -np.log(u) / (baseline_hazard * np.exp(logits))

    X = np.squeeze(X)
    actual = concordance_index_censored(np.ones(n_samples, dtype=bool), time_event, X)
    return X, time_event, actual[0]

